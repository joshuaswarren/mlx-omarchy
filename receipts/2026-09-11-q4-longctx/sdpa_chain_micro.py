#!/usr/bin/env python3
"""Paired decode-SDPA chain microbench: packed batched vs scalar in-kernel.

24 serial mx.fast.scaled_dot_product_attention calls (one per Qwen2.5-0.5B
layer) at q [1,14,1,64] over a growing KV cache slice, timed as eval-sync
walls with the graph rebuilt per rep (perturbed V) to force real GPU work.
Runs both MLX_OMARCHY_SDPA_DECODE_SCALAR arms alternately per rep.

  MLX_DISABLE_COMPILE=1 python3 sdpa_chain_micro.py --out chain.json --reps 20
"""
import argparse
import json
import os
import statistics
import time

import mlx.core as mx

HEADS, KVH, HD, L = 14, 2, 64, 24
CAP = 4096


def build_chain(k_len, delta):
    rs = mx.random.state(7)
    q = mx.random.normal((1, HEADS, 1, HD), key=rs).astype(mx.float16)
    rs = mx.random.state(8)
    k_cache = mx.random.normal((1, KVH, CAP, HD), key=rs).astype(mx.float16)
    rs = mx.random.state(9)
    v_cache = mx.random.normal((1, KVH, CAP, HD), key=rs).astype(mx.float16)
    k = k_cache[:, :, :k_len, :]
    v = v_cache[:, :, :k_len, :]
    outs = []
    for _ in range(L):
        outs.append(mx.fast.scaled_dot_product_attention(
            q, k, v + delta, scale=1.0 / (HD ** 0.5)))
    return outs


def time_chain(k_len, reps):
    walls = []
    for r in range(reps):
        delta = mx.array([float(r)], dtype=mx.float16)
        outs = build_chain(k_len, delta)  # build outside the timed window
        t0 = time.perf_counter()
        mx.eval(*outs)
        walls.append((time.perf_counter() - t0) * 1e3)
    return walls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="chain.json")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--k-lens", default="30,262,1053")
    args = ap.parse_args()

    rows = []
    for k_len in [int(x) for x in args.k_lens.split(",")]:
        arms = {}
        for name, env in [("scalar", "1"), ("packed", "0")]:
            os.environ["MLX_OMARCHY_SDPA_DECODE_SCALAR"] = env
            walls = time_chain(k_len, args.reps)
            arms[name] = {
                "median_ms": statistics.median(walls),
                "min_ms": min(walls),
                "max_ms": max(walls),
            }
        gain = arms["scalar"]["median_ms"] - arms["packed"]["median_ms"]
        rows.append({"k_len": k_len, **arms,
                     "gain_ms": gain,
                     "gain_pct": 100.0 * gain / arms["scalar"]["median_ms"]})
        print(rows[-1], flush=True)

    with open(args.out, "w") as f:
        json.dump({"schema": "mlx-omarchy/q4-longctx/sdpa-chain/1",
                   "reps": args.reps, "rows": rows}, f, indent=2)


if __name__ == "__main__":
    main()
