#!/usr/bin/env python3
"""f64-reference accuracy of the affected Q4 projections.

Loads the pinned Qwen2.5-0.5B-Instruct-4bit checkpoint, takes the four
real prefill projection weights (q 896x896, v 896x128, down 4864x896,
gate_up 896x9728 at layer 5), runs the loaded wheel's qmm prefill at
m=1053 on seeded f16 activations, and compares against three float64
oracles:

  ref        pure f64 dequant + f64 matmul (s64*q+b64)
  ref_w16    oracle with the weight pre-rounded to f16 after f32 s*q+b
             arithmetic - the upstream Metal data path (f32 arithmetic,
             one f16 rounding at the tile store, f16 staged operands,
             f32 accumulate)

Metrics per projection: max/mean abs error vs ref (f32-widened),
outputs whose f16 differs from RNE16(oracle), and exact-match fraction
against RNE16(ref_w16) - the Metal-arithmetic identity test.

usage: qmm_f16_accuracy.py OUT_JSON
"""
import glob
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

M = 1053
SEED = 20260911
SNAP = glob.glob(str(Path.home() / ".cache/huggingface/hub/"
                     "models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/"
                     "snapshots/*"))
# The checkpoint stores gate_proj and up_proj separately; the runtime
# SwiGLU fusion concatenates them into the fused 896x9728 qmm the same
# way, so the fused projection is synthesized from both.
LAYERS = {
    "q_896x896": ["model.layers.5.self_attn.q_proj"],
    "v_896x128": ["model.layers.5.self_attn.v_proj"],
    "down_4864x896": ["model.layers.5.mlp.down_proj"],
    "gate_up_896x9728": [
        "model.layers.5.mlp.gate_proj",
        "model.layers.5.mlp.up_proj"],
}


def rne16(a):
    return a.astype(np.float16)


# mx.load() has been observed to yield an empty mapping
# intermittently on this driver (2026-09-11 window logs: empty-key
# asserts that vanish with an added stderr print and reappear without).
# Materialize every needed tensor eagerly, exactly once, and retry the
# whole load when the mapping comes back empty instead of trusting it.
def load_all(prefixes, attempts=4):
    flat = [p_ for group in prefixes for p_ in group]
    want = tuple(p_ + "." for p_ in flat)
    for attempt in range(attempts):
        out = {}
        for path in sorted(glob.glob(SNAP[0] + "/*.safetensors")):
            for key, arr in mx.load(path).items():
                for w in want:
                    if key.startswith(w):
                        out[key] = np.array(arr)  # eager copy off the lazy map
        if all(any(k.startswith(w) for k in out) for w in want):
            return out
        print(f"load_all attempt {attempt}: empty/partial ({len(out)} keys)",
              file=sys.stderr)
        time.sleep(2)
    raise RuntimeError("mx.load empty after retries")


def dequant_f64(w, scales, biases, round16):
    n, kp = w.shape
    k = kp * 8
    out = np.empty((n, k), dtype=np.float64)
    q = np.stack(
        [(w >> np.uint32(4 * nib)) & np.uint32(0xF)
         for nib in range(8)], axis=-1).astype(np.float64)  # n, kp, 8
    flat = q.reshape(n, k)
    if round16:
        s32 = scales.astype(np.float32)
        b32 = biases.astype(np.float32)
        wv = (s32.reshape(n, k // 64, 1) *
              flat.reshape(n, k // 64, 64) +
              b32.reshape(n, k // 64, 1)).astype(np.float16)
        out = wv.astype(np.float64).reshape(n, k)
    else:
        s64 = scales.astype(np.float64)
        b64 = biases.astype(np.float64)
        out = (s64.reshape(n, k // 64, 1) *
               flat.reshape(n, k // 64, 64) +
               b64.reshape(n, k // 64, 1)).reshape(n, k)
    return out


def main():
    assert SNAP, "model snapshot not found"
    import sys
    rng = np.random.default_rng(SEED)
    store = load_all(list(LAYERS.values()))
    rows = []
    for name, prefixes in LAYERS.items():
        w = np.concatenate(
            [store[p_ + ".weight"] for p_ in prefixes], axis=0)
        s = np.concatenate(
            [store[p_ + ".scales"].astype(np.float16) for p_ in prefixes],
            axis=0)
        b = np.concatenate(
            [store[p_ + ".biases"].astype(np.float16) for p_ in prefixes],
            axis=0)
        n, kp = w.shape
        k = kp * 8
        x = rng.standard_normal((M, k)).astype(np.float16)
        out = mx.quantized_matmul(
            mx.array(x), mx.array(w), mx.array(s), mx.array(b),
            transpose=True, group_size=64, bits=4)
        mx.eval(out)
        got = np.asarray(out).astype(np.float32)

        ref = (x.astype(np.float64) @
               dequant_f64(w, s, b, round16=False).T)
        ref16 = (x.astype(np.float64) @
                 dequant_f64(w, s, b, round16=True).T)

        err = np.abs(got.astype(np.float64) - ref)
        err16 = np.abs(got.astype(np.float64) - ref16)
        flips_ref = int((rne16(ref).astype(np.float32) != got).sum())
        exact_metal = int(
            (rne16(ref16).astype(np.float32) == got).sum())
        rows.append({
            "layer": name, "m": M, "k": k, "n": n,
            "max_abs_err_vs_ref": float(err.max()),
            "mean_abs_err_vs_ref": float(err.mean()),
            "max_abs_err_vs_refw16": float(err16.max()),
            "mean_abs_err_vs_refw16": float(err16.mean()),
            "outputs": got.size,
            "f16_flips_vs_rne16_ref": flips_ref,
            "exact_vs_rne16_refw16": exact_metal,
            "exact_frac_vs_metal_model": exact_metal / got.size,
        })
        print(json.dumps(rows[-1]), flush=True)
        mx.clear_cache()
    Path(sys.argv[1]).write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
