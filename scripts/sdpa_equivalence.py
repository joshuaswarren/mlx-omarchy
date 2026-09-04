#!/usr/bin/env python3
"""Value-level SDPA equivalence suite for the f16-scores rework.

Compares mx.fast.scaled_dot_product_attention (the primitive under test,
running the backend's f16-scores path) against the validated f32 composed
path it replaces, on the same build and device. The composed reference
mirrors the pre-rework ScaledDotProductAttention::eval_gpu exactly:
cast q/k/v to f32, scale as an f32 multiply, GQA via 5-D reshape, f32
scores matmul, additive mask, f32 softmax, f32 probs@v matmul, and the
5-D result flattened back to 4-D.

The f16-storage tolerance model is also checked directly: the rework's
scores are stored as f16 with f32 accumulation, so the suite emulates
that storage (f16 operands upcast to f32 for the dot, f16 store of the
scaled scores) and requires scores/probs within 2e-3 of the f32
reference and outputs within 1e-3, before comparing the real primitive.

Coverage per the census addendum 3 equivalence bar:
  - decode shapes [B, heads, 1, head_dim]
  - non-contiguous cache slices at offsets 1, 7, 41, 256
  - GQA repeat factors 1, 2, 7
  - head_dim 48 (not a power-of-two multiple)
  - q_len 1 and q_len > 1
  - causal and explicit-additive mask legs

Usage (llvmpipe dev box):
  MLX_OMARCHY_ALLOW_NON_APPLE=1 python3 scripts/sdpa_equivalence.py
Exit 0 = all cases pass; exit 1 = first failing case printed.
"""

import sys

import numpy as np

import mlx.core as mx

SIGMA = 0.5


def draw(key, shape):
    return (mx.random.normal(shape, key=key) * SIGMA).astype(mx.float16)


def causal_addend(q_len, k_len, dtype):
    """The causal additive mask as a host-loaded fixture array.

    Mirrors what each C++ path writes per element: 0 / -1e30 in f32,
    0 / -inf in f16 (f16 stores -1e30 as -inf). Built in numpy: the
    backend refuses GPU broadcast compares, and a fixture needs no GPU
    kernel. A multiplier form would NaN the kept positions (0 * -inf).
    """
    rows = np.arange(q_len)[:, None] + (k_len - q_len)
    cols = np.arange(k_len)[None, :]
    blocked = rows < cols
    if dtype == np.float16:
        values = np.where(blocked, -np.inf, 0.0).astype(np.float16)
    else:
        values = np.where(blocked, -1e30, 0.0).astype(np.float32)
    return mx.array(values)


def composed_reference(q, k, v, scale, mask=None, causal=False):
    """The validated f32 composition (pre-rework eval_gpu), in mx ops."""
    batch, heads, q_len, head_dim = q.shape
    kv_heads = k.shape[1]
    k_len = k.shape[2]
    v_dim = v.shape[3]
    q32 = q.astype(mx.float32)
    k32 = k.astype(mx.float32)
    v32 = v.astype(mx.float32)
    q32 = q32 * mx.array(scale, mx.float32)
    if kv_heads != heads:
        repeats = heads // kv_heads
        qs = q32.reshape(batch, kv_heads, repeats, q_len, head_dim)
        k32 = k32.reshape(batch, kv_heads, 1, k_len, head_dim)
        v32 = v32.reshape(batch, kv_heads, 1, k_len, v_dim)
    else:
        qs = q32
    scores = qs @ k32.swapaxes(-1, -2)
    if causal:
        scores = scores + causal_addend(q_len, k_len, np.float32)
    elif mask is not None:
        m = mask.astype(mx.float32)
        if kv_heads != heads:
            m = m.reshape(
                batch, kv_heads, heads // kv_heads, q_len, k_len)
        scores = scores + m
    probs = mx.softmax(scores.astype(mx.float32), axis=-1)
    out = (probs @ v32).reshape(batch, heads, q_len, v_dim)
    return out.astype(q.dtype), scores, probs


def f16_storage_emulation(q, k, v, scale, mask=None, causal=False):
    """What the reworked path computes: f16 storage, f32 accumulation.

    matmul.comp loads f16 operands as float, accumulates in float, scales
    by alpha in float, and stores f16; softmax.comp runs float math over
    f16 storage. This mirrors that round trip.
    """
    batch, heads, q_len, head_dim = q.shape
    kv_heads = k.shape[1]
    k_len = k.shape[2]
    v_dim = v.shape[3]
    q16 = q.astype(mx.float16)
    k16 = k.astype(mx.float16)
    v16 = v.astype(mx.float16)
    if kv_heads != heads:
        repeats = heads // kv_heads
        qs = q16.reshape(batch, kv_heads, repeats, q_len, head_dim)
        k32 = k16.reshape(
            batch, kv_heads, 1, k_len, head_dim).astype(mx.float32)
        v32 = v16.reshape(
            batch, kv_heads, 1, k_len, v_dim).astype(mx.float32)
    else:
        qs = q16
        k32 = k16.astype(mx.float32)
        v32 = v16.astype(mx.float32)
    scores = (qs.astype(mx.float32) @ k32.swapaxes(-1, -2)) * scale
    scores = scores.astype(mx.float16)
    if causal:
        scores = scores + causal_addend(q_len, k_len, np.float16)
    elif mask is not None:
        scores = scores + mask.astype(mx.float16)
    probs = mx.softmax(scores.astype(mx.float32), axis=-1).astype(mx.float16)
    out = (probs.astype(mx.float32) @ v32).reshape(
        batch, heads, q_len, v_dim)
    return out.astype(mx.float16), scores, probs


def check(name, q, k, v, scale, mask=None, causal=False):
    kwargs = {}
    if causal:
        kwargs["mask"] = "causal"
    elif mask is not None:
        kwargs["mask"] = mask
    got = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, **kwargs)
    mx.eval(got)
    ref, ref_scores, ref_probs = composed_reference(q, k, v, scale, mask, causal)
    sim, sim_scores, sim_probs = f16_storage_emulation(q, k, v, scale, mask, causal)

    failures = []

    # Masked positions compare over the finite pairing only: f16 stores
    # the causal addend as -inf while f32 keeps -1e30, so the raw
    # difference at masked positions is meaningless by design.
    def max_err(sim, ref):
        a = np.asarray(sim.astype(mx.float32))
        b = np.asarray(ref.astype(mx.float32))
        finite = np.isfinite(a) & np.isfinite(b)
        if not finite.any():
            return float("nan")
        return float(np.abs(a[finite] - b[finite]).max())

    sim_scores_err = max_err(sim_scores, ref_scores)
    sim_probs_err = max_err(sim_probs, ref_probs)
    sim_out_err = max_err(sim, ref)
    if sim_scores_err > 2e-3:
        failures.append(f"emulated scores err {sim_scores_err:.3e} > 2e-3")
    if sim_probs_err > 2e-3:
        failures.append(f"emulated probs err {sim_probs_err:.3e} > 2e-3")
    if sim_out_err > 1e-3:
        failures.append(f"emulated out err {sim_out_err:.3e} > 1e-3")

    got32 = got.astype(mx.float32)
    ref32 = ref.astype(mx.float32)
    if got32.shape != ref32.shape:
        failures.append(f"shape mismatch got {got32.shape} ref {ref32.shape}")
        out_err = float("inf")
    else:
        out_err = float(mx.abs(got32 - ref32).max())
        if out_err > 1e-3:
            failures.append(f"primitive out err {out_err:.3e} > 1e-3")
    # mx bool reductions carry a known llvmpipe defect (receipts/
    # boolall-2026-09-03); finiteness goes through numpy.
    if not np.isfinite(np.asarray(got32)).all():
        failures.append("primitive output has non-finite values")

    status = "PASS" if not failures else "FAIL"
    print(f"[{status}] {name}: out_err={out_err:.3e} "
          f"(scores_err={sim_scores_err:.3e}, probs_err={sim_probs_err:.3e})")
    for failure in failures:
        print(f"    {failure}")
    return not failures


def cache_slice_case(rng, offset, heads, kv_heads, head_dim, q_len=1):
    """k/v drawn from a longer cache array, sliced - never row-contiguous."""
    max_len = offset + 16
    k_len = offset + q_len
    key_q, key_k, key_v = mx.random.split(rng, 3)
    cache_k = draw(key_k, (1, kv_heads, max_len, head_dim))
    cache_v = draw(key_v, (1, kv_heads, max_len, head_dim))
    q = draw(key_q, (1, heads, q_len, head_dim))
    k = cache_k[:, :, :k_len, :]
    v = cache_v[:, :, :k_len, :]
    scale = 1.0 / head_dim ** 0.5
    repeats = heads // kv_heads
    name = (f"cache-slice off={offset} hd={head_dim} "
            f"heads={heads} kv={kv_heads} rep={repeats} q_len={q_len}")
    return check(name, q, k, v, scale)


def main():
    rng = mx.random.key(0)
    ok = True

    # (b) non-contiguous cache slices at offsets 1, 7, 41, 256.
    for offset in (1, 7, 41, 256):
        ok &= cache_slice_case(rng, offset, heads=4, kv_heads=4, head_dim=64)

    # (c) GQA repeat factors 1, 2, 7 over cache slices (the decode shape).
    for heads, kv_heads in ((4, 4), (4, 2), (7, 1)):
        ok &= cache_slice_case(rng, 41, heads, kv_heads, 64)

    # (d) head_dim 48 (not a power-of-two multiple), decode and GQA 7.
    ok &= cache_slice_case(rng, 41, heads=7, kv_heads=1, head_dim=48)
    ok &= cache_slice_case(rng, 7, heads=2, kv_heads=2, head_dim=48)

    # (e) q_len 1 and q_len > 1, dense inputs, batch > 1.
    key_q, key_k, key_v = mx.random.split(rng, 3)
    q = draw(key_q, (1, 4, 16, 48))
    k = draw(key_k, (1, 4, 16, 48))
    v = draw(key_v, (1, 4, 16, 48))
    ok &= check("q_len=16 dense hd=48 heads=4", q, k, v, 1.0 / 48 ** 0.5)
    key_q, key_k, key_v = mx.random.split(rng, 3)
    q = draw(key_q, (2, 7, 5, 64))
    k = draw(key_k, (2, 1, 9, 64))
    v = draw(key_v, (2, 1, 9, 64))
    ok &= check("q_len=5 k_len=9 GQA7 batch=2", q, k, v, 1.0 / 64 ** 0.5)

    # Prefill into an existing cache: q_len > 1 against a cache slice.
    ok &= cache_slice_case(rng, 41, heads=4, kv_heads=2, head_dim=64, q_len=5)

    # (a) decode shapes [1, heads, 1, 64], dense and causal.
    key_q, key_k, key_v = mx.random.split(rng, 3)
    q = draw(key_q, (1, 4, 1, 64))
    k = draw(key_k, (1, 4, 64, 64))
    v = draw(key_v, (1, 4, 64, 64))
    ok &= check("decode dense [1,4,1,64]", q, k, v, 0.125)
    key_q, key_k, key_v = mx.random.split(rng, 3)
    q = draw(key_q, (1, 4, 1, 64))
    k = draw(key_k, (1, 4, 64, 64))
    v = draw(key_v, (1, 4, 64, 64))
    ok &= check("decode causal [1,4,1,64]", q, k, v, 0.125, causal=True)
    key_q, key_k, key_v = mx.random.split(rng, 3)
    q = draw(key_q, (1, 4, 1, 64))
    k = draw(key_k, (1, 2, 64, 64))
    v = draw(key_v, (1, 2, 64, 64))
    ok &= check("decode causal GQA2", q, k, v, 0.125, causal=True)

    # Causal with q_len > 1 and an explicit additive f16 mask.
    key_q, key_k, key_v, key_m = mx.random.split(rng, 4)
    q = draw(key_q, (1, 4, 8, 64))
    k = draw(key_k, (1, 4, 32, 64))
    v = draw(key_v, (1, 4, 32, 64))
    mask = draw(key_m, (1, 4, 8, 32))
    ok &= check("causal q_len=8 k_len=32", q, k, v, 0.125, causal=True)
    ok &= check("additive f16 mask", q, k, v, 0.125, mask=mask)

    print("ALL PASS" if ok else "SUITE FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
