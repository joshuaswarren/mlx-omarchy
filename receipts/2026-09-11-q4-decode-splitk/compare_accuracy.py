#!/usr/bin/env python3
"""Compare the base and split-K accuracy captures against float64 RNE
references (compare_accuracy.py; runs on the CPU, no GPU needed).

Two references per shape:
  refA (model level): round_f16(x_f64 @ dequant(w)_f16->f64)
  refB (kernel algebra): round_f16(sum_g s_g*(sum x_i q_i)
                                    + b_g*(sum x_i) in f64)
Report, per shape and map: elements differing from each reference,
max abs error, ulp16 distance, and splitk-vs-base differing elements.
"""
import json
from pathlib import Path

import numpy as np

SHAPES = [
    ("q_proj", 896, 896), ("k_proj", 896, 128), ("v_proj", 896, 128),
    ("o_proj", 896, 896), ("gate_proj", 896, 4864),
    ("up_proj", 896, 4864), ("down_proj", 4864, 896),
]


def unpack_nibbles(w_u32, k):
    """[N, K/8] uint32 -> [N, K] uint8 nibbles, low nibble first."""
    n_words = w_u32.shape[1]
    shifts = (np.arange(8, dtype=np.uint32) * np.uint32(4))
    nib = (w_u32[:, :, None] >> shifts[None, None, :]) & np.uint32(0xF)
    return nib.reshape(w_u32.shape[0], n_words * 8)[:, :k].astype(np.uint8)


def ulp16_distance(a, b):
    """Distance in f16 representation steps between two f16 arrays."""
    def key(v):
        sign = (v >> 15) & 1
        mag = v & 0x7FFF
        return np.where(sign == 1, -(mag.astype(np.int64)),
                        mag.astype(np.int64))
    return np.abs(key(a.view(np.uint16)) - key(b.view(np.uint16)))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    report = {}
    for name, k, n in SHAPES:
        base = np.load(args.dir / f"base-{name}.npz")
        pair = np.load(args.dir / f"splitk-{name}.npz")
        x64 = base["x"].astype(np.float64)
        entry = {"k": k, "n": n}
        refs = {}
        # refA: model level through the f16 dequantized weights.
        refs["refA"] = (base["deq"].astype(np.float64) @ x64).astype(
            np.float16)
        # refB: the kernel's own group algebra in f64.
        q = unpack_nibbles(base["w"], k)  # [N, K]
        scales = base["scales"].astype(np.float64)  # [N, K/64]
        biases = base["biases"].astype(np.float64)
        xg = x64[None, :].repeat(n, axis=0).reshape(n, k // 64, 64)
        qg = q.reshape(n, k // 64, 64).astype(np.float64)
        dot_g = (xg * qg).sum(axis=2)          # sum x_i q_i per group
        sum_g = xg.sum(axis=2)                 # sum x_i per group
        refs["refB"] = (scales * dot_g + biases * sum_g).sum(axis=1).astype(
            np.float16)
        for map_name, cap in (("base", base), ("pair", pair)):
            out = cap["out"].astype(np.float16)
            entry[map_name] = {}
            for ref_name, ref in refs.items():
                diff = out != ref
                err = np.abs(out.astype(np.float64) -
                             ref.astype(np.float64))
                entry[map_name][ref_name] = {
                    "differing": int(diff.sum()),
                    "of": int(out.size),
                    "max_abs_err": float(err.max()),
                    "max_ulp16": int(ulp16_distance(out, ref).max()),
                }
        pb = pair["out"].astype(np.float16) != base["out"].astype(np.float16)
        entry["splitk_vs_base_differing"] = int(pb.sum())
        report[name] = entry

    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
