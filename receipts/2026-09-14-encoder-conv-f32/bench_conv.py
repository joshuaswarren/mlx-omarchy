#!/usr/bin/env python3
"""Microbench the encoder expand 1x1 ConvF32 shape.

NHWC [1,1,375,1024] x [2048,1,1,1024] is the vulkan_encoder lift of MIL
[1,1024,375] x [2048,1024,1] valid, output volume 768000.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time

import numpy as np

import mlx.core as mx


def sha256_bytes(array: mx.array) -> str:
    mx.eval(array)
    mx.synchronize()
    return hashlib.sha256(np.array(array).tobytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    mx.set_default_device(mx.gpu)
    mx.random.seed(args.seed)
    x = mx.random.normal((1, 1, 375, 1024), dtype=mx.float32)
    w = mx.random.normal((2048, 1, 1, 1024), dtype=mx.float32)
    mx.eval(x, w)
    mx.synchronize()

    y = mx.conv2d(x, w, stride=1, padding=0)
    digest = sha256_bytes(y)

    for _ in range(args.warmup):
        mx.eval(mx.conv2d(x, w, stride=1, padding=0))
        mx.synchronize()

    times = []
    for _ in range(args.reps):
        mx.synchronize()
        start = time.perf_counter()
        out = mx.conv2d(x, w, stride=1, padding=0)
        mx.eval(out)
        mx.synchronize()
        times.append(time.perf_counter() - start)

    times.sort()
    payload = {
        "shape_in": [1, 1, 375, 1024],
        "shape_wt": [2048, 1, 1, 1024],
        "n": 768000,
        "digest": digest,
        "times_s": times,
        "min_s": times[0],
        "median_s": times[len(times) // 2],
        "mlx": getattr(mx, "__file__", None),
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
