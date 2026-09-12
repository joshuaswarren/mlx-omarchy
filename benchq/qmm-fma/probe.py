#!/usr/bin/env python3
"""Kernel-level A/B probe for the scalar-FMA prefill routes.

Runs the real dispatch stack at the canonical prefill shapes: Q4 g64
f16 quantized_matmul (FMA route vs cooperative-matrix route via
MLX_OMARCHY_NO_QMM_FMA) and dense bf16 matmul in the linear-layer
orientation (FMA route vs MatmulBF16Coopmat via
MLX_OMARCHY_NO_MATMUL_FMA). Wall-anchored medians plus the sha256 of
the output bytes per cell, so a side that changes bits is visible next
to the timing.

Run one side per process (the env is read once per process):
  python3 probe.py --side fma --dtype q4
  python3 probe.py --side coopmat --dtype q4
  python3 probe.py --side fma --dtype bf16
  python3 probe.py --side coopmat --dtype bf16
"""

import argparse
import hashlib
import json
import os
import statistics
import sys
import time

import numpy as np

import mlx.core as mx

NO_QMM_FMA = "MLX_OMARCHY_NO_QMM_FMA"
NO_MATMUL_FMA = "MLX_OMARCHY_NO_MATMUL_FMA"

# name -> list of (m, n, k), the four-screen bakeoff cells
SHAPES = {
    "kv": [(262, 896, 128), (1053, 896, 128)],
    "q": [(262, 896, 896), (1053, 896, 896)],
    "down": [(262, 896, 4864), (1053, 896, 4864)],
    "gate_up": [(262, 9728, 896), (1053, 9728, 896)],
}


def digest(a):
    # bytes(memoryview(...)) keeps bf16/f16 arrays bit-exact; numpy
    # cannot attach a dtype to bf16 and mis-parses the buffer format.
    return hashlib.sha256(bytes(memoryview(a))).hexdigest()


def timed_median(fn, warmup, timed):
    for _ in range(warmup):
        mx.eval(fn())
    samples = []
    sha = None
    for _ in range(timed):
        y = fn()
        t0 = time.perf_counter()
        mx.eval(y)
        samples.append(time.perf_counter() - t0)
        sha = digest(y)
    return statistics.median(samples), sha


def bench(m, n, k, dtype, warmup, timed, seed):
    x = mx.random.normal((m, k), key=mx.random.key(seed))
    w = mx.random.normal((n, k), key=mx.random.key(seed + 1))
    if dtype == "q4":
        x = x.astype(mx.float16)
        q = mx.quantize(w.astype(mx.float16), group_size=64, bits=4)
        if hasattr(q, "scales"):
            words, scales, biases = q.weight, q.scales, q.biases
        else:
            words, scales, biases = q
        scales = scales.astype(mx.float16)
        biases = biases.astype(mx.float16)

        def run():
            return mx.quantized_matmul(
                x, words, scales, biases, True, 64, 4, "affine"
            )
    else:
        x = x.astype(mx.bfloat16)
        wt = w.astype(mx.bfloat16).T

        def run():
            return x @ wt

    ms, sha = timed_median(run, warmup, timed)
    return {
        "gflops": 2.0 * m * n * k / ms / 1e9,
        "ms": ms * 1e3,
        "sha": sha,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", choices=["fma", "coopmat"], required=True)
    ap.add_argument("--dtype", choices=["q4", "bf16"], required=True)
    ap.add_argument("--cells", default="kv,q,down,gate_up")
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--timed", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cfg", default=None, help="tile-shape cfg env value")
    args = ap.parse_args()

    env_name = NO_QMM_FMA if args.dtype == "q4" else NO_MATMUL_FMA
    cfg_name = "MLX_OMARCHY_QMM_FMA_CFG" if args.dtype == "q4" else (
        "MLX_OMARCHY_MATMUL_FMA_CFG"
    )
    if args.side == "coopmat":
        os.environ[env_name] = "1"
    else:
        os.environ.pop(env_name, None)
    if args.cfg is not None:
        os.environ[cfg_name] = args.cfg
    else:
        os.environ.pop(cfg_name, None)
    os.environ.setdefault("MLX_DISABLE_COMPILE", "1")

    try:
        from mlx_omarchy.version import __version__ as wheel_version
    except Exception:
        wheel_version = "unknown"

    summary = {
        "side": args.side,
        "dtype": args.dtype,
        "wheel": wheel_version,
        "cells": {},
    }
    for name in args.cells.split(","):
        for m, n, k in SHAPES[name]:
            cell = bench(m, n, k, args.dtype, args.warmup, args.timed, args.seed)
            cell["shape"] = [m, n, k]
            summary["cells"][f"{m}x{n}x{k}"] = cell
            print(
                json.dumps(
                    {
                        "side": args.side,
                        "dtype": args.dtype,
                        "shape": f"{m}x{n}x{k}",
                        "gflops": round(cell["gflops"], 1),
                        "ms": round(cell["ms"], 3),
                        "sha": cell["sha"],
                    },
                    flush=True,
                )
            )
    print("JSON " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    sys.exit(main())
