#!/usr/bin/env python3
import argparse
import json
import statistics
import time

import mlx.core as mx

SHAPES = {
    "hidden-hidden": (896, 896),
    "hidden-kv": (896, 128),
    "hidden-intermediate": (896, 4864),
    "intermediate-hidden": (4864, 896),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("shape", choices=SHAPES)
    parser.add_argument("m", type=int, choices=(262, 1053))
    parser.add_argument("--warmups", type=int, default=8)
    parser.add_argument("--reps", type=int, default=16)
    args = parser.parse_args()
    k, n = SHAPES[args.shape]
    info = mx.device_info()

    mx.random.seed(0)
    x = mx.random.uniform(low=-1, high=1, shape=(args.m, k)).astype(mx.float16)
    w = mx.random.uniform(low=-1, high=1, shape=(n, k)).astype(mx.float16)
    packed, scales, biases = mx.quantize(w, group_size=64, bits=4)
    mx.eval(x, packed, scales, biases)

    def operation():
        return mx.quantized_matmul(
            x, packed, scales, biases, transpose=True, group_size=64, bits=4
        )

    for _ in range(args.warmups):
        mx.eval(mx.abs(operation()))
    wall = []
    for _ in range(args.reps):
        start = time.perf_counter_ns()
        mx.eval(mx.abs(operation()))
        wall.append((time.perf_counter_ns() - start) / 1e6)

    print(json.dumps({
        "mlx_version": mx.__version__,
        "cooperative_matrix_f32_8": info["cooperative_matrix_f32_8"],
        "device_info": info,
        "shape": args.shape,
        "m": args.m,
        "k": k,
        "n": n,
        "warmups": args.warmups,
        "reps": args.reps,
        "wall_median_ms": statistics.median(wall),
    }))


if __name__ == "__main__":
    main()
