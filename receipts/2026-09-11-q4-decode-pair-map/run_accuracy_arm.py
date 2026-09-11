#!/usr/bin/env python3
"""Run the seven decode projection shapes through the device Q4 GEMV
under ONE lane map (chosen by MLX_OMARCHY_QMM_VEC_Q4_PAIR in the
environment), saving x, packed w, scales, biases, dequantized w, and the
device output per shape for the f64 comparison in compare_accuracy.py.

Run twice on the M1 GPU: once with the env unset (base map), once with
MLX_OMARCHY_QMM_VEC_Q4_PAIR=1 (pair map). Inputs are seeded numpy, so
both processes see identical bytes.
"""
import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

# Qwen2.5-0.5B decode projections: (name, K, N) f16; group 64, 4 bit.
SHAPES = [
    ("q_proj", 896, 896),
    ("k_proj", 896, 128),
    ("v_proj", 896, 128),
    ("o_proj", 896, 896),
    ("gate_proj", 896, 4864),
    ("up_proj", 896, 4864),
    ("down_proj", 4864, 896),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import os
    map_name = "pair" if os.environ.get(
        "MLX_OMARCHY_QMM_VEC_Q4_PAIR") == "1" else "base"

    rng = np.random.default_rng(20260911)
    results = {}
    args.out.mkdir(parents=True, exist_ok=True)
    for name, k, n in SHAPES:
        x = (rng.standard_normal(k) * 0.35).astype(np.float16)
        w_f16 = (rng.standard_normal((n, k)) * 0.03).astype(np.float16)
        wt = mx.array(w_f16)
        w_q, scales, biases = mx.quantize(wt, group_size=64, bits=4)
        x_mx = mx.array(x)[None, :]  # [1, K] -> the decode GEMV shape
        out = mx.quantized_matmul(
            x_mx, w_q, scales, biases, transpose=True,
            group_size=64, bits=4)
        mx.eval(out)
        deq = mx.dequantize(w_q, scales, biases, group_size=64, bits=4)
        mx.eval(deq)
        np.savez(
            args.out / f"{map_name}-{name}.npz",
            x=x, w=np.array(w_q), scales=np.array(scales),
            biases=np.array(biases), deq=np.array(deq),
            out=np.array(out[0]))
        results[name] = {
            "k": k, "n": n,
            "out_f16_first8": [float(v) for v in np.array(out[0])[:8]],
        }
    (args.out / f"{map_name}-index.json").write_text(
        json.dumps({"map": map_name, "shapes": results}, indent=2) + "\n")
    print(f"{map_name} arm written to {args.out}")


if __name__ == "__main__":
    main()
