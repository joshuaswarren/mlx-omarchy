#!/usr/bin/env python3
"""Q4 coopmat prefill GFLOP/s on the four real Qwen2.5-0.5B shapes.

Wall-anchored timing only (honeykrisp device timestamps undercount ~2x):
time.perf_counter around mx.eval of a freshly built expression (the
mx.eval of an already-materialized array is a no-op), 4 warmups + 30
timed, median. Shapes (m, k, n): m in {262, 1053}; (k, n) in
{(896, 128), (896, 896), (4864, 896), (896, 9728)} - kv, q, down_proj,
gate_up at the 262/1053 prefill legs. f16 scales/biases, fixed seeds,
x drawn from N(0,1) in f16.

usage: qmm_f16_shapes.py OUT_JSON
"""
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

SHAPES = [(262, 896, 128), (262, 896, 896), (262, 4864, 896),
          (262, 896, 9728), (1053, 896, 128), (1053, 896, 896),
          (1053, 4864, 896), (1053, 896, 9728)]
WARMUP = 4
TIMED = 30


def make_case(m, k, n, seed=1234):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((m, k)).astype(np.float16)
    q = rng.integers(0, 16, size=(n, k)).astype(np.uint8)
    packed = np.zeros((n, k // 8), dtype=np.uint32)
    for nib in range(8):
        packed |= q[:, nib::8].astype(np.uint32) << (8 * nib)
    scales = rng.standard_normal((n, k // 64)).astype(np.float16) * 0.01
    biases = rng.standard_normal((n, k // 64)).astype(np.float16) * 0.1
    return (mx.array(x), mx.array(packed), mx.array(scales),
            mx.array(biases))


def main():
    out_path = Path(sys.argv[1])
    rows = []
    for m, k, n in SHAPES:
        x, w, s, b = make_case(m, k, n)

        def build():
            return mx.quantized_matmul(
                x, w, s, b, transpose=True, group_size=64, bits=4)

        out = build()
        mx.eval(out)
        for _ in range(WARMUP):
            mx.eval(build())
        samples = []
        for _ in range(TIMED):
            t0 = time.perf_counter()
            mx.eval(build())
            samples.append(time.perf_counter() - t0)
        med = statistics.median(samples)
        gflops = 2.0 * m * k * n / med / 1e9
        row = {"m": m, "k": k, "n": n, "median_ms": med * 1e3,
               "gflop_s": gflops,
               "digest": hashlib.sha256(np.asarray(out)).hexdigest()[:16]}
        rows.append(row)
        print(json.dumps(row), flush=True)
    out_path.write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
