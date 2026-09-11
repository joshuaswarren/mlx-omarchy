#!/usr/bin/env python3
"""Kernel-isolated Q4 coopmat data-path bench (no model in the loop).

Runs mx.quantized_matmul at the real Qwen2.5-0.5B prefill shapes for
each MLX_OMARCHY_QMM_COOP_BENCH arm and prints one JSON row per
(shape, arm) with the median wall time and the f16 output digest.

Arm-to-arm digest equality is the bit-identity gate: arms 1 and 2 must
match arm 0 exactly (same values, same per-output ascending 8-wide
coopmat accumulation order). Arms 3 and 4 are ceiling measurement
arms; their digests are deterministic but not comparable to arm 0.

Run one process per arm: MLX_OMARCHY_QMM_COOP_BENCH=<arm> python3 ...
"""
import hashlib
import json
import statistics
import sys
import time

import numpy as np
import mlx.core as mx

SHAPES = [(896, 128), (896, 896), (4864, 896), (896, 9728)]
MS = (262, 1053)


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


def quantize_f16(n, k, group=64, bits=4, seed=7):
    w = mx.random.normal((n, k), key=mx.random.key(seed)) * 0.5
    w, scales, biases = mx.quantize(w.astype(mx.float32), group, bits)
    scales = scales.astype(mx.float16)
    biases = biases.astype(mx.float16)
    mx.eval(w, scales, biases)
    return w, scales, biases


def f16_seed(shape, seed):
    x = mx.random.normal(shape, key=mx.random.key(seed)).astype(mx.float16)
    mx.eval(x)
    return x


def main():
    arm = int(__import__("os").environ.get("MLX_OMARCHY_QMM_COOP_BENCH", "0"))
    for k, n in SHAPES:
        w_words, scales, biases = quantize_f16(n, k, seed=7)
        for m in MS:
            x = f16_seed((m, k), 11)

            def call():
                return mx.quantized_matmul(
                    x, w_words, scales, biases, True, 64, 4)

            ms = timed(call)
            row = {
                "arm": arm,
                "shape": f"{m}x{k}x{n}",
                "median_ms": round(ms, 4),
                "f16_digest": f16_digest(call()),
                "tflops": round(2.0 * m * n * k / (ms * 1e6), 1),
            }
            print(json.dumps(row))
    print("PROBE-DONE")


if __name__ == "__main__":
    main()
