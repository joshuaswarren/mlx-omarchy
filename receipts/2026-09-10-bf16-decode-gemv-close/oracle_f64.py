#!/usr/bin/env python3
"""Compare captured BF16 GEMV device bits against the RNE(f64) oracle.

Inputs are regenerated bit-exactly from the same salted pattern used by
fixed_bf16.py, so any captured outputs npz (baseline or candidate, fork
or stock) can be scored without shipping tensors around.

Oracle definition — identical to test_matmul_family's
host_matmul_bf16_reference + host_bf16_round, and to the kernel store
path for an exact f32 accumulator: f64 products of the bf16-widened
operands, plain f64 sum, then RNE f64->f32 and RNE f32->bf16.

Scoring:
  - exact_elements: device bit == oracle bit
  - max_ulp_from_truth: worst bf16 grid-step distance from the oracle
With --baseline, additionally asserts the per-element closeness bar:
  dist(candidate, oracle) <= dist(baseline, oracle) for every element.
"""

import argparse
import json
from pathlib import Path

import numpy as np

U = np.uint32


def patterned_bits(count, salt):
    out = np.empty(count, dtype=np.uint16)
    chunk = 1 << 20
    for start in range(0, count, chunk):
        stop = min(start + chunk, count)
        idx = np.arange(start, stop, dtype=U)
        mixed = idx + U((salt * 0x9E3779B9) & 0xFFFFFFFF)
        mixed ^= mixed >> U(16)
        mixed *= U(0x7FEB352D)
        mixed ^= mixed >> U(15)
        mixed *= U(0x846CA68B)
        mixed ^= mixed >> U(16)
        mantissa = (mixed >> U(9)) & U(0x7F)
        exponent = U(120) + ((mixed >> U(1)) & U(7))
        sign = (mixed & U(1)) << U(15)
        out[start:stop] = (sign | (exponent << U(7)) | mantissa).astype(np.uint16)
    return out


def widen(bits16):
    """bf16 bits -> f32 values (exact widening)."""
    return (bits16.astype(U) << U(16)).view(np.float32)


def rne_f64_to_bf16_bits(sums_f64):
    """f64 -> f32 (RNE) -> bf16 (RNE), byte-identical to host_bf16_round."""
    f32 = sums_f64.astype(np.float32)
    bits32 = f32.view(U)
    isnan = np.isnan(f32)
    bits16 = (
        (bits32 + U(0x7FFF) + ((bits32 >> U(16)) & U(1))) >> U(16)
    ).astype(np.uint16)
    bits16[isnan] = ((bits32[isnan] >> U(16)) | U(0x40)).astype(np.uint16)
    return bits16


def bf16_ulp_distance(a_f32, b_f32):
    """Distance in bf16 grid steps between two bf16-widened f32 values."""
    a = a_f32.view(U)
    b = b_f32.view(U)
    a = np.where((a & U(0x80000000)) != 0, U(0x80000000) - (a & U(0x7FFFFFFF)), a)
    b = np.where((b & U(0x80000000)) != 0, U(0x80000000) - (b & U(0x7FFFFFFF)), b)
    return np.abs(a.astype(np.int64) - b.astype(np.int64))


def oracle_bits_for_shape(k, n, trials):
    w_bits = patterned_bits(k * n, 23).reshape(n, k)
    x_bits = [patterned_bits(k, 11 + t * 101).reshape(1, k) for t in range(trials)]
    x_bits[0].fill(np.uint16(0x3F80))
    w_bits[:4].fill(np.uint16(0))
    w_bits[:4, 0] = np.uint16(0x3F80)
    w_bits[:4, 4] = np.uint16(0x3380)
    w_bits[:4, 8] = np.uint16(0xBF80)
    wf = widen(w_bits.reshape(-1)).reshape(n, k).astype(np.float64)
    oracle = np.empty((trials, 1, n), dtype=np.uint16)
    for t, xb in enumerate(x_bits):
        xf = widen(xb.reshape(-1)).astype(np.float64)
        oracle[t] = rne_f64_to_bf16_bits(xf @ wf.T).reshape(1, n)
    return oracle


def score(npz_path, k, n, trials):
    result = np.load(npz_path)["result"]
    assert result.shape == (trials, 1, n), (npz_path, result.shape)
    oracle = oracle_bits_for_shape(k, n, trials)
    flat = result.reshape(-1)
    flat_oracle = oracle.reshape(-1)
    dist = bf16_ulp_distance(widen(flat), widen(flat_oracle))
    return {
        "npz": str(npz_path),
        "shape": [k, n],
        "elements": int(flat.size),
        "exact_elements": int(np.count_nonzero(flat == flat_oracle)),
        "max_ulp_from_truth": int(dist.max()),
        "_dist": dist,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--baseline", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    shapes = [(896, 896), (896, 128), (896, 4864), (4864, 896), (896, 151936)]
    trials = 8
    report = {"schema": "bf16-decode-gemv-close-oracle/1", "shapes": []}
    regressions = 0
    for k, n in shapes:
        cand_path = (
            args.candidate / f"bf16_{k}x{n}.npz" if args.candidate.is_dir() else args.candidate
        )
        cand = score(cand_path, k, n, trials)
        entry = {key: value for key, value in cand.items() if not key.startswith("_")}
        if args.baseline is not None:
            base = score(args.baseline / f"bf16_{k}x{n}.npz", k, n, trials)
            entry["baseline_exact_elements"] = base["exact_elements"]
            entry["baseline_max_ulp_from_truth"] = base["max_ulp_from_truth"]
            worse = np.count_nonzero(cand["_dist"] > base["_dist"])
            entry["elements_closer_than_baseline"] = int(
                np.count_nonzero(cand["_dist"] < base["_dist"])
            )
            entry["elements_equal_distance_as_baseline"] = int(
                int(cand["elements"]) - int(worse)
                - int(np.count_nonzero(cand["_dist"] < base["_dist"]))
            )
            entry["elements_farther_than_baseline"] = int(worse)
            regressions += int(worse)
        report["shapes"].append(entry)

    report["per_element_closeness_bar"] = "PASS" if regressions == 0 else "FAIL"
    report["elements_farther_than_baseline_total"] = regressions
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
