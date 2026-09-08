#!/usr/bin/env python3
"""Correctness for the prefill candidates at the Qwen QMM shapes.

Runs the same 4x4 shape matrix as the C++ 'qmm coopmat prefill' case
(m in {8, 32, 262, 1053} x (k, n) in {896x896, 896x4864, 4864x896,
896x128}, f16, transposed affine 4-bit / group 64) against the host
reference with the qmm tile anchor bound:

  (3k + 1) * 2 * max_abs * 2^-23 + 2 * max_abs * 2^-11

whatever kernels the current env selects (staged coopmat, inline
fragment, register-blocked tile). Prints one JSON line per case.
"""

import json
import sys

import mlx.core as mx


def bound_for(k, max_abs):
    return (3 * k + 1) * 2 * max_abs * 2.0 ** -23 + 2 * max_abs * 2.0 ** -11

def main():
    mx.random.seed(0)
    shapes = [
        (m, k, n)
        for m in (8, 32, 262, 1053)
        for (k, n) in ((896, 896), (896, 4864), (4864, 896), (896, 128))
    ]
    worst = 0.0
    for (m, k, n) in shapes:
        x = (mx.random.normal((m, k)) * 0.05).astype(mx.float16)
        w = (mx.random.normal((n, k)) * 0.05)
        wq, scales, biases = mx.quantize(w, group_size=64, bits=4)
        wq = wq.astype(mx.uint32)
        scales = scales.astype(mx.float16)
        biases = biases.astype(mx.float16)
        out = mx.quantized_matmul(
            x, wq, scales, biases, transpose=True,
            group_size=64, bits=4)
        mx.eval(out)
        # Host reference in float32.
        w_hat = mx.dequantize(
            wq, scales.astype(mx.float32), biases.astype(mx.float32),
            group_size=64, bits=4)
        ref = (x.astype(mx.float32) @ w_hat.T)
        mx.eval(ref)
        diff = mx.abs(out.astype(mx.float32) - ref)
        max_abs = mx.abs(ref).max().item()
        err = diff.max().item()
        bound = bound_for(k, max_abs)
        ok = err <= bound
        worst = max(worst, err / bound)
        print(json.dumps({
            "m": m, "k": k, "n": n, "err": err, "bound": bound,
            "max_abs": max_abs, "ok": bool(ok),
            "err_over_bound": err / bound,
        }), flush=True)
        if not ok:
            sys.exit(1)
    print(json.dumps({"all_ok": True, "worst_err_over_bound": worst}))


if __name__ == "__main__":
    main()
