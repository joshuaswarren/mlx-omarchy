#!/usr/bin/env python3
"""Paired decode-SDPA chain microbench: packed batched vs scalar in-kernel.

Mirrors receipts/2026-09-10-decode-attribution/chain_microbench.py: inputs
are built once per k_len and memoized after the first eval; each rep rebuilds
a 24-call serial chain from a perturbed q (fresh graph nodes force real GPU
work); the timed window is mx.eval of the outputs. Two arms alternate,
selected by MLX_OMARCHY_SDPA_DECODE_SCALAR.

  MLX_DISABLE_COMPILE=1 python3 sdpa_chain_micro.py --out chain.json
"""
import argparse
import json
import os
import statistics
import time

import mlx.core as mx

HEADS, KVH, HD, L = 14, 2, 64, 24
CAP = 4096


def make_chain(k_len):
    q0 = mx.random.normal((1, HEADS, 1, HD),
                          key=mx.random.key(7)).astype(mx.float16)
    kc = mx.random.normal((1, KVH, CAP, HD),
                          key=mx.random.key(8)).astype(mx.float16)
    vc = mx.random.normal((1, KVH, CAP, HD),
                          key=mx.random.key(9)).astype(mx.float16)
    k = kc[:, :, :k_len, :]
    v = vc[:, :, :k_len, :]

    def build(d):
        q = q0 + d
        outs = []
        for _ in range(L):
            q = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=1.0 / (HD ** 0.5))
            outs.append(q)
        return outs

    return build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="chain.json")
    ap.add_argument("--reps", type=int, default=25)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--k-lens", type=int, nargs="*",
                    default=[30, 262, 511, 1023, 1053, 1084])
    args = ap.parse_args()

    rows = []
    for k_len in args.k_lens:
        build = make_chain(k_len)
        arms = {}
        for name, env in [("scalar", "1"), ("packed", "0")]:
            os.environ["MLX_OMARCHY_SDPA_DECODE_SCALAR"] = env
            evals = []
            for rep in range(args.warmup + args.reps):
                d = mx.array(float(rep % 17) * 1e-3, mx.float16)
                outs = build(d)  # host graph build, outside the timed window
                t0 = time.perf_counter()
                mx.eval(outs)
                evals.append((time.perf_counter() - t0) * 1e3)
            sample = evals[args.warmup:]
            arms[name] = {"median_ms": round(statistics.median(sample), 4),
                          "min_ms": round(min(sample), 4),
                          "max_ms": round(max(sample), 4)}
        gain = arms["scalar"]["median_ms"] - arms["packed"]["median_ms"]
        rows.append({"k_len": k_len, **arms, "gain_ms": round(gain, 4),
                     "gain_pct": round(100.0 * gain /
                                       arms["scalar"]["median_ms"], 2)})
        print(rows[-1], flush=True)

    with open(args.out, "w") as f:
        json.dump({"schema": "mlx-omarchy/q4-longctx/sdpa-chain/2",
                   "reps": args.reps, "rows": rows}, f, indent=2)


if __name__ == "__main__":
    main()
