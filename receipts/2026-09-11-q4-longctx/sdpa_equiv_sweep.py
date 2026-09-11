#!/usr/bin/env python3
"""Bit-identity gate: packed batched decode SDPA vs the frozen scalar path.

Runs mx.fast.scaled_dot_product_attention at Qwen2.5-0.5B decode shapes over
a k_len sweep that covers every kernel regime (one-pass, 64-block two-pass,
128-block two-pass), once with MLX_OMARCHY_SDPA_DECODE_SCALAR=1 (legacy
scalar in-kernel loops) and once without (aux_size==1 packed batched path).
The pass requires uint16-bit-exact outputs per k_len.

Run under the GPU lock on jwm1 with the wheel venv:
  python3 sdpa_equiv_sweep.py --out equiv.json
"""
import argparse
import json
import os
import platform

import mlx.core as mx

K_LENS = [1, 2, 31, 32, 33, 61, 262, 320, 511, 1023, 1024, 1025, 1053, 1084,
          1500, 2048, 2049]
HEADS, KVH, HD = 14, 2, 64
CAP = 4096


def run_one(seed, k_len):
    """Return output uint16 words for the CURRENT process env."""
    rs = mx.random.key(seed)
    q = mx.random.normal((1, HEADS, 1, 64), key=rs).astype(mx.float16)
    k_cache = mx.random.normal(
        (1, KVH, CAP, HD), key=mx.random.key(seed + 1)).astype(mx.float16)
    v_cache = mx.random.normal(
        (1, KVH, CAP, HD), key=mx.random.key(seed + 2)).astype(mx.float16)
    out = mx.fast.scaled_dot_product_attention(
        q, k, v, scale=1.0 / (HD ** 0.5))
    out.eval()
    return out.view(mx.uint16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="equiv.json")
    args = ap.parse_args()

    results = []
    failures = []
    for seed, k_len in [(11, k) for k in K_LENS]:
        os.environ["MLX_OMARCHY_SDPA_DECODE_SCALAR"] = "1"
        scalar = run_one(seed, k_len)
        os.environ["MLX_OMARCHY_SDPA_DECODE_SCALAR"] = "0"
        packed = run_one(seed, k_len)
        same = bool(mx.array_equal(scalar, packed))
        blocks = 128 if k_len > 1024 else (64 if k_len == 1024 else 0)
        row = {"seed": seed, "k_len": k_len, "blocks": blocks,
               "bit_exact": same}
        results.append(row)
        if not same:
            failures.append(row)
            print(f"FAIL k={k_len} blocks={blocks}", flush=True)
        else:
            print(f"ok k={k_len} blocks={blocks}", flush=True)

    verdict = {
        "schema": "mlx-omarchy/q4-longctx/sdpa-equiv/1",
        "bit_exact_all": not failures,
        "cases": len(results),
        "failures": failures,
        "host": platform.machine(),
        "note": "scalar arm = frozen legacy kernel loops via "
                "MLX_OMARCHY_SDPA_DECODE_SCALAR=1; packed arm = aux_size==1 "
                "batched word loads. Pass requires uint16 equality.",
    }
    with open(args.out, "w") as f:
        json.dump(verdict, f, indent=2)
    print(json.dumps(verdict, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
