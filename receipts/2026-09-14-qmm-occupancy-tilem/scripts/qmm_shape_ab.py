#!/usr/bin/env python3
"""Per-shape A/B of the Q4 coopmat prefill kernel, one process per
MLX_OMARCHY_QMM_COOPMAT_WG_PER_CORE setting.

The four shapes are the Qwen2.5-0.5B projection set at m=1053 recovered
from a real prefill in receipts/2026-09-14-jw16-max-gpu-attribution.md
(shape-census.txt), with that capture's dispatch counts so a per-shape
delta can be summed into a predicted prefill delta.

Every run prints a sha256 over the exact output bytes. The tile pick is
a performance choice only: all three row counts must produce the
identical f16 buffer, and the digests here are what proves it, per
shape, independent of the end-to-end generated-id digests.

Timing is wall time around one qmm plus its eval, median of --reps,
which carries the same per-dispatch host cost in every arm.
"""

import argparse
import hashlib
import json
import os
import statistics
import sys
import time

# (n, k, dispatches in the 1053-token prefill capture)
SHAPES = [
    (4864, 896, 46),
    (896, 4864, 23),
    (896, 896, 46),
    (128, 896, 48),
]
M = 1053
GROUP = 64
BITS = 4
CORES_M1_MAX = 32
# kCoopmatWorkgroupsPerCore in primitives.cpp.
DEFAULT_WG_PER_CORE = 6


def landed_rows(m, n, cores, per_core):
    """The host-side rule in primitives.cpp coopmat_tile_rows().

    One step, not a ladder: the 8-row twin was measured and lost, so 16
    is the floor and the only question is whether the 32-row grid fills
    the part.
    """
    if cores == 0 or per_core == 0:
        return 32
    n_groups = (n + 31) // 32
    return 16 if ((m + 31) // 32) * n_groups < cores * per_core else 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8,
                    help="qmm dispatches per timed eval; amortizes the "
                         "per-eval host cost, which measures ~245 us on "
                         "this host and would otherwise swamp the 221 us "
                         "n=128 kernel")
    ap.add_argument("--shapes", default=None,
                    help="override the shape table as n:k[:count],... "
                         "for mapping the row-count crossover on shapes "
                         "the model does not contain")
    ap.add_argument("--cores", type=int, default=CORES_M1_MAX,
                    help="GPU cores the selector will see; only labels "
                         "the predicted pick, never changes it")
    ap.add_argument("--m", type=int, default=M,
                    help="prefill rows; 1053 is the ctx1024 leg, "
                         "30 the short leg")
    ap.add_argument("--seed", type=int, default=20260914)
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    import mlx.core as mx
    import numpy as np
    per_core_env = os.environ.get("MLX_OMARCHY_QMM_COOPMAT_WG_PER_CORE")
    device = str(mx.metal.device_info().get("device_name", "unknown"))
    out = {
        "device": device,
        "wg_per_core_env": per_core_env,
        "reps": args.reps,
        "batch": args.batch,
        "m": args.m,
        "shapes": [],
    }
    # Effective floor: unset means the compiled-in default.
    per_core = (DEFAULT_WG_PER_CORE if per_core_env is None
                else int(per_core_env))
    shapes = SHAPES
    if args.shapes:
        shapes = []
        for spec in args.shapes.split(","):
            parts = spec.split(":")
            shapes.append((int(parts[0]), int(parts[1]),
                           int(parts[2]) if len(parts) > 2 else 0))

    m = args.m
    for n, k, count in shapes:
        mx.random.seed(args.seed)
        # Distinct x per dispatch: identical arguments would let one
        # eval collapse the batch to a single kernel launch.
        xs = [mx.random.normal((m, k)).astype(mx.float16)
              for _ in range(args.batch)]
        w = mx.random.normal((n, k)).astype(mx.float16)
        wq, scales, biases = mx.quantize(w, group_size=GROUP, bits=BITS)
        mx.eval(xs, wq, scales, biases)

        def batch():
            return [mx.quantized_matmul(
                x, wq, scales, biases,
                transpose=True, group_size=GROUP, bits=BITS) for x in xs]

        for _ in range(args.warmup):
            mx.eval(batch())
        samples = []
        for _ in range(args.reps):
            t0 = time.perf_counter_ns()
            mx.eval(batch())
            samples.append((time.perf_counter_ns() - t0) / args.batch)
        y = batch()[0]
        mx.eval(y)
        assert y.dtype == mx.float16, y.dtype
        digest = hashlib.sha256(
            np.array(y, copy=False).tobytes()).hexdigest()[:16]

        median_ns = statistics.median(samples)
        rows = landed_rows(m, n, args.cores, per_core)
        flop = 2.0 * m * n * k
        out["shapes"].append({
            "n": n,
            "k": k,
            "dispatches_in_prefill": count,
            "workgroups": ((m + rows - 1) // rows) * ((n + 31) // 32),
            "predicted_tile_rows": rows,
            "median_us": round(median_ns / 1000.0, 3),
            "min_us": round(min(samples) / 1000.0, 3),
            "max_us": round(max(samples) / 1000.0, 3),
            "tflops": round(flop / median_ns / 1000.0, 4),
            "out_sha256_16": digest,
        })
        del xs, w, wq, scales, biases, y

    print(json.dumps(out, sort_keys=True))


if __name__ == "__main__":
    main()
