#!/usr/bin/env python3
"""Q4 qmm wall time at the missing m=262 rung (receipt 2026-09-10-qmm-prefill-tile
measured m=1053 and m=30 only). Same method: isolated primitive wall, median of
30 after 4 warmups, f16 scales/biases, per-shape f16 output digest.

  MLX_DISABLE_COMPILE=1 python3 qmm_m262_probe.py --out qmm_m262.json
"""
import argparse
import hashlib
import json
import statistics
import time

import numpy as np
import mlx.core as mx

SHAPES = [(896, 128), (896, 896), (4864, 896), (896, 9728)]


def timed(fn, warmup=4, reps=30):
    for _ in range(warmup + 1):
        mx.eval(fn())
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(fn())
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples)


def f16_digest(a):
    b = np.ascontiguousarray(np.asarray(a, dtype=np.float16)).tobytes()
    return hashlib.sha256(b).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="qmm_m262.json")
    args = ap.parse_args()
    rows = []
    for k, n in SHAPES:
        w = mx.random.normal((n, k), key=mx.random.key(7)) * 0.5
        w, scales, biases = mx.quantize(w.astype(mx.float32), 64, 4)
        scales, biases = scales.astype(mx.float16), biases.astype(mx.float16)
        x = mx.random.normal((262, k), key=mx.random.key(11)).astype(mx.float16)
        mx.eval(w, scales, biases, x)
        call = lambda: mx.quantized_matmul(x, w, scales, biases, True, 64, 4)
        ms = timed(call)
        row = {"shape": f"262x{k}x{n}", "median_ms": round(ms, 4),
               "f16_digest": f16_digest(call()),
               "gflops": round(2.0 * 262 * n * k / (ms * 1e6), 1)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    total = round(sum(r["median_ms"] for r in rows) * 24.0, 1)
    print(json.dumps({"per_layer_sum_ms": round(sum(r["median_ms"] for r in rows), 4),
                      "per_model_x24_ms": total}))
    with open(args.out, "w") as f:
        json.dump({"schema": "mlx-omarchy/q4-longctx/qmm-m262/1",
                   "rows": rows, "per_model_x24_ms": total}, f, indent=2)


if __name__ == "__main__":
    main()
