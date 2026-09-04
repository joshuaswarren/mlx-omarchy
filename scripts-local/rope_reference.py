#!/usr/bin/env python3
"""Float64 host reference for the three Honeykrisp-failing RoPE configs.

Ground truth: the RoPE algebra evaluated in float64 from the exact stored
input values (the f32/f16 input tensor, the f32 freqs where the battery
passes one, or the f64 formula exp(-i*log(base)/half) where it passes a
base). Parses the rope_probe dump (f32-image hex bits) and scores the fused
and composed paths element by element against the reference.

Usage:
  rope_reference.py reference             # reference + expected tables only
  rope_reference.py score <dump-file>     # score probe dump against reference
"""

import struct
import sys

import numpy as np

F32 = np.float32
F16 = np.float16


def pattern(count: int, seed: int) -> np.ndarray:
    """Verbatim from test_fast_ops.cpp pattern()."""
    state = np.uint32(seed)
    out = np.empty(count, dtype=np.float32)
    for i in range(count):
        state = np.uint32((int(state) * 1664525 + 1013904223) & 0xFFFFFFFF)
        out[i] = F32(np.float64(np.uint64(state % np.uint32(20000)) / 10000.0) - 1.0)
    return out


def rope_input(shape, seed: int, dtype) -> np.ndarray:
    count = int(np.prod(shape))
    x = pattern(count, seed).reshape(shape)
    return x.astype(dtype)


def theta_grid(t_values, offset: int, half: int, base: float, freqs_f32=None):
    """Return (positions, inv_freq_true, theta_true) in float64.

    freqs_f32: the exact f32 freqs values the program passes (stored-input
    truth); None -> the base formula exp(-i*log(base)/half) in f64.
    """
    positions = np.array([float(t + offset) for t in t_values])  # exact ints
    if freqs_f32 is not None:
        inv_freq = 1.0 / freqs_f32.astype(np.float64)
    else:
        inv_freq = np.exp(-np.arange(half, dtype=np.float64)
                          * (np.log(base) / half))
    theta = positions[:, None] * inv_freq[None, :]
    return positions, inv_freq, theta


def rope_reference_f64(x, dims: int, traditional: bool, positions, theta):
    """Full RoPE in float64 from stored inputs. x: (B,N,T,D) float64 view."""
    half = dims // 2
    cos = np.cos(theta)  # (T, half)
    sin = np.sin(theta)
    B, N, T, D = x.shape
    out = np.empty((B, N, T, D), dtype=np.float64)
    if traditional:
        x1 = x[..., 0:dims:2]
        x2 = x[..., 1:dims:2]
        c = cos[None, None, :, :]
        s = sin[None, None, :, :]
        out[..., 0:dims:2] = x1 * c - x2 * s
        out[..., 1:dims:2] = x1 * s + x2 * c
    else:
        x1 = x[..., :half]
        x2 = x[..., half:dims]
        c = cos[None, None, :, :]
        s = sin[None, None, :, :]
        out[..., :half] = x1 * c - x2 * s
        out[..., half:dims] = x1 * s + x2 * c
    return out


CONFIGS = {
    # name: (shape, dims, traditional, with_freqs, base, offset, seed, dtype)
    "variant2_f32": ((2, 3, 7, 16), 16, False, True, 10000.0, 3, 101, F32),
    "decode_q_f16": ((1, 14, 1, 128), 128, False, False, 500000.0, 17, 103, F16),
    "decode_k_f16": ((1, 2, 1, 64), 64, False, False, 500000.0, 17, 107, F16),
    "partial_f32": ((2, 3, 5, 32), 16, True, False, 10000.0, 2, 109, F32),
    "partial_f16": ((2, 3, 5, 32), 16, True, False, 10000.0, 2, 109, F16),
}


def f32_bits_to_floats(text_lines):
    return np.array([struct.unpack(">f", struct.pack(">I", int(b, 16)))[0]
                     for b in text_lines], dtype=np.float32)


def read_dump(path):
    sections = {}
    with open(path) as fh:
        lines = [line.strip() for line in fh if line.strip()]
    i = 0
    while i < len(lines):
        header = lines[i].split()
        name, count = header[0], int(header[1])
        body = lines[i + 1:i + 1 + count]
        assert len(body) == count, f"short section {name}"
        sections[name] = f32_bits_to_floats([row.split()[0] for row in body])
        i += 1 + count
    return sections


def score_config(name, cfg, sections, verbose=True):
    shape, dims, traditional, with_freqs, base, offset, seed, dtype = cfg
    B, N, T, D = shape
    half = dims // 2
    x_stored = rope_input(shape, seed, dtype).astype(np.float64)
    freqs_f32 = sections["freqs"] if with_freqs and "freqs" in sections else None
    _, inv_freq_true, theta_true = theta_grid(
        range(T), offset, half, base, freqs_f32)
    truth = rope_reference_f64(x_stored, dims, traditional,
                               None, theta_true).ravel()

    fused = sections[f"fused_{name}"].astype(np.float64)
    composed = sections[f"composed_{name}"].astype(np.float64)

    # Passthrough tail: stored input bits must survive untouched.
    tail = (np.arange(D) >= dims)
    tail_mask = np.tile(tail, int(np.prod(shape)) // D)
    x_flat = sections[f"input_{name}"]
    tail_fused = np.abs(fused[tail_mask] - x_flat[tail_mask].astype(np.float64))
    tail_comp = np.abs(composed[tail_mask] - x_flat[tail_mask].astype(np.float64))
    tail_bits_fused = (sections[f"fused_{name}"][tail_mask]
                       != x_flat[tail_mask]).sum()
    tail_bits_comp = (sections[f"composed_{name}"][tail_mask]
                      != x_flat[tail_mask]).sum()
    rot = ~tail_mask
    err_f = np.abs(fused[rot] - truth[rot])
    err_c = np.abs(composed[rot] - truth[rot])

    # Trig decomposition: each path's stored cos/sin vs f64 trig of the
    # replicated f32 theta (pure codegen difference) and of theta_true.
    pairs_t, pairs_half = T, half
    pos_f32 = sections.get(f"positions_{name}")
    invf_f32 = sections.get(f"inv_freqs_{name}")
    trig_lines = {}
    if pos_f32 is not None and invf_f32 is not None:
        theta_f32 = (pos_f32[:, None].astype(np.float64)
                     * invf_f32[None, :].astype(np.float64))
        theta_rep = (pos_f32[:, None] * invf_f32[None, :]).astype(np.float64)
        ft = sections[f"fused_trig_{name}"].astype(np.float64)
        ct = sections[f"composed_trig_{name}"].astype(np.float64)
        cos_true = np.cos(theta_f32)  # f64 trig of the paths' own f32 theta
        sin_true = np.sin(theta_f32)
        cos_true_rep = np.cos(theta_rep.astype(np.float64))
        sin_true_rep = np.sin(theta_rep.astype(np.float64))
        # fused_trig layout: forward half-split emits cos then sin; for
        # traditional, even lanes cos, odd lanes sin. Rows are D wide
        # (D may exceed dims when passthrough lanes exist; the probe
        # feeds (1,0) pairs only inside the rotated range).
        def split_trig(v):
            v = v.reshape(-1, D)
            if traditional:
                return v[:, 0:dims:2].ravel(), v[:, 1:dims:2].ravel()
            return v[:, :half].ravel(), v[:, half:dims].ravel()

        fc, fs = split_trig(ft)
        cc, cs = split_trig(ct)
        # Rows count: rotate extent; B/N from the shape for tiling.
        Bv, Nv = shape[0], int(np.prod(shape[1:-2]))
        # Quantize the reference to the storage grid so the metric
        # isolates codegen error from f16 quantization, then tile over
        # (B, N) into the sections' (b, n, t, i) order.
        def store(v):
            return v.astype(np.float64) if dtype == F32 else v.astype(
                np.float16).astype(np.float64)

        cos_ref = np.tile(store(np.cos(theta_f32)), (Bv * Nv, 1, 1)).ravel()
        sin_ref = np.tile(store(np.sin(theta_f32)), (Bv * Nv, 1, 1)).ravel()
        trig_lines["fused_trig_vs_f64_of_f32_theta_max"] = float(
            max(np.max(np.abs(fc - cos_ref)), np.max(np.abs(fs - sin_ref))))
        trig_lines["composed_trig_vs_f64_of_f32_theta_max"] = float(
            max(np.max(np.abs(cc - cos_ref)), np.max(np.abs(cs - sin_ref))))
        trig_lines["fused_minus_composed_trig_max"] = float(
            max(np.max(np.abs(fc - cc)), np.max(np.abs(fs - cs))))

    result = {
        "max_theta": float(np.max(np.abs(theta_true))),
        "fused_max_err": float(np.max(err_f)),
        "composed_max_err": float(np.max(err_c)),
        "fused_median_err": float(np.median(err_f)),
        "composed_median_err": float(np.median(err_c)),
        "fused_rms_err": float(np.sqrt(np.mean(err_f ** 2))),
        "composed_rms_err": float(np.sqrt(np.mean(err_c ** 2))),
        "fused_wins": int(np.sum(err_f < err_c)),
        "composed_wins": int(np.sum(err_c < err_f)),
        "ties": int(np.sum(err_f == err_c)),
        "n_rotated": int(rot.sum()),
        "tail_fused_max": float(np.max(tail_fused)) if tail_fused.size else 0.0,
        "tail_composed_max": float(np.max(tail_comp)) if tail_comp.size else 0.0,
        "tail_fused_bit_diffs": int(tail_bits_fused),
        "tail_composed_bit_diffs": int(tail_bits_comp),
    }
    result.update(trig_lines)
    if verbose:
        print(f"== {name} ==")
        for k, v in result.items():
            print(f"  {k}: {v:.6g}" if isinstance(v, float) else f"  {k}: {v}")
    return result
def score_sweep(sections):
    """Position sweep: error vs the f64 reference as a function of
    position, per path. This is the mechanism discriminator:
      - both paths growing together = driver range-reduction band,
      - fused growing faster = fused-only defect,
      - flat = neither (sub-ulp codegen variance)."""
    import re

    sweep_re = re.compile(
        r"^(?:fused|composed|input)_sweep_(\w+)_pos(\d+)_d(\d+)$")
    found = {}
    for key in sections:
        m = sweep_re.match(key)
        if m:
            entry = (int(m.group(2)), int(m.group(3)))
            if entry not in found.setdefault(m.group(1), []):
                found[m.group(1)].append(entry)
    for tag, entries in sorted(found.items()):
        dims = entries[0][1]
        half = dims // 2
        freqs_variant = tag.startswith("freqs")
        base = 10000.0 if freqs_variant else 500000.0
        if freqs_variant:
            # The freqs variant reciprocates the f32 freqs values, so
            # inv_freq runs UP to base**((half-1)/half) ~ 3162.
            synth = np.exp(-np.arange(half) * (np.log(base) / half))
            invf = 1.0 / synth
        else:
            invf = np.exp(-np.arange(half) * (np.log(base) / half))
        dtype = F16 if tag.endswith("f16") else F32
        print(f"== sweep {tag} (dims {dims}, base {base:g}) ==")
        print("  position  theta_max   err_fused     err_composed"
              "    fused/comp")
        for pos, _ in sorted(entries):
            name = f"sweep_{tag}_pos{pos}_d{dims}"
            # section keys are fused_<name>/composed_<name>/input_<name>
            # with name = sweep_<tag>_pos<p>_d<d>; tag already carries
            # no "sweep_" prefix after the regex fix.
            x = sections[f"input_{name}"].reshape(1, 1, 1, dims)
            theta_row = float(pos) * invf
            theta = theta_row[None, :]
            truth = rope_reference_f64(
                x.astype(np.float64), dims, False, None, theta).ravel()
            fused = sections[f"fused_{name}"].astype(np.float64)
            comp = sections[f"composed_{name}"].astype(np.float64)
            ef = np.abs(fused - truth)
            ec = np.abs(comp - truth)
            ratio = float(np.max(ef) / max(np.max(ec), 1e-300))
            print(f"  {pos:>8d}  {np.abs(theta).max():>9.1f}   "
                  f"{np.max(ef):.3e}    {np.max(ec):.3e}    {ratio:8.3f}")


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "score":
        sections = read_dump(sys.argv[2])
        rows = {}
        for name, cfg in CONFIGS.items():
            rows[name] = score_config(name, cfg, sections)
        print("\n== verdict inputs ==")
        for name, r in rows.items():
            ratio = (r["fused_max_err"] / max(r["composed_max_err"], 1e-300)
                     if r["composed_max_err"] else float("inf"))
            print(
                f"{name}: fused max {r['fused_max_err']:.3g} "
                f"median {r['fused_median_err']:.3g} | composed max "
                f"{r['composed_max_err']:.3g} median "
                f"{r['composed_median_err']:.3g} | ratio {ratio:.3g} | "
                f"wins {r['fused_wins']}/{r['composed_wins']}/{r['ties']} "
                f"| tail bits fused {r['tail_fused_bit_diffs']} composed "
                f"{r['tail_composed_bit_diffs']}")
        score_sweep(sections)
        return
    # Reference-only mode: print theta envelopes per config. The freqs
    # variant's inv_freq is reciprocal(exp(-i*log(base)/half)), so its
    # envelope must model the reciprocal, not the bare exp.
    for name, cfg in CONFIGS.items():
        shape, dims, traditional, with_freqs, base, offset, seed, dtype = cfg
        half = dims // 2
        T = shape[-2]
        if with_freqs:
            synth = np.exp(
                -np.arange(half) * (np.log(base) / half)).astype(F32)
            _, inv_freq, theta = theta_grid(
                range(T), offset, half, base, synth)
        else:
            _, inv_freq, theta = theta_grid(range(T), offset, half, base)
        bound = np.abs(theta).max() * 2 ** -23 * 2
        print(f"{name}: max|theta|={np.abs(theta).max():.6g} "
              f"f32-theta-quantization bound~{bound:.3g}")


if __name__ == "__main__":
    main()
