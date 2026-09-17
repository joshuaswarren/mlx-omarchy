#!/usr/bin/env python3
"""Per-shape GPU census of an MLX_OMARCHY_GPU_PROFILE stream.

Groups timed dispatches by (kernel, gx, gy, gz, weight-binding range), which
is enough to recover the exact quantized matmul shape: the dispatch path sets
gx = ceil(n/TILE_N), gy = ceil(m/TILE_M) and the InputW binding range is
exactly n*k/2 bytes for affine 4-bit weights, so k = 2*range/n.

Reports workgroups per dispatch, workgroups per GPU core for a given core
count, achieved GFLOP and TFLOP/s. Used to test whether a shape-derived grid
saturates a wider GPU.

Usage:
  shape_census.py PROFILE.jsonl --compute-h compute.h [--cores 32]
      [--kernel QmmPrefillCoopmatF16] [--m 1053]
"""

import argparse
import json
import re
import statistics
from collections import defaultdict


def kernel_names(header_path):
    names = []
    inside = False
    with open(header_path, encoding="utf-8") as f:
        for line in f:
            if "enum class ComputeKernel" in line:
                inside = True
                continue
            if inside:
                m = re.match(r"^\s{2}(\w+),\s*$", line)
                if m:
                    names.append(m.group(1))
                elif "};" in line:
                    break
    return {i: n for i, n in enumerate(names)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile")
    ap.add_argument("--compute-h", required=True)
    ap.add_argument("--cores", type=int, default=32)
    ap.add_argument("--kernel", default="QmmPrefillCoopmatF16")
    ap.add_argument("--tile-n", type=int, default=32)
    ap.add_argument("--m", type=int, default=1053)
    args = ap.parse_args()

    names = kernel_names(args.compute_h)
    want = {e for e, n in names.items() if n == args.kernel}
    if not want:
        raise SystemExit(f"kernel {args.kernel} not in {args.compute_h}")

    groups = defaultdict(list)
    with open(args.profile, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("k") != "d" or "t0" not in rec or rec["e"] not in want:
                continue
            w_range = rec["b"][1][2] if len(rec["b"]) > 1 else 0
            key = (rec["gx"], rec["gy"], rec["gz"], w_range)
            groups[key].append(rec["t1"] - rec["t0"])

    print(f"{args.kernel}: {sum(len(v) for v in groups.values())} timed "
          f"dispatches, {args.cores} GPU cores, m={args.m}")
    print(f"{'n':>7} {'k':>7} {'wgroups':>8} {'wg/core':>8} {'cnt':>4} "
          f"{'tot ms':>9} {'mean us':>9} {'GFLOP':>8} {'TFLOP/s':>8} {'rel':>6}")
    rows = []
    for (gx, gy, gz, w_range), durs in groups.items():
        n = gx * args.tile_n
        k = 2 * w_range // n if n else 0
        flops = 2 * args.m * n * k
        mean = statistics.mean(durs)
        rows.append((sum(durs), n, k, gx * gy * gz, len(durs), mean,
                     flops, flops / (mean * 1e-9) / 1e12))
    best = max(r[7] for r in rows) if rows else 1.0
    for tot, n, k, wg, cnt, mean, flops, tfs in sorted(rows, reverse=True):
        print(f"{n:>7} {k:>7} {wg:>8} {wg / args.cores:>8.1f} {cnt:>4} "
              f"{tot / 1e6:>9.3f} {mean / 1e3:>9.1f} {flops / 1e9:>8.3f} "
              f"{tfs:>8.2f} {tfs / best:>6.2f}")
    tot_flops = sum(r[6] * r[4] for r in rows)
    tot_time = sum(r[0] for r in rows)
    print(f"total {tot_flops / 1e9:.2f} GFLOP in {tot_time / 1e6:.3f} ms = "
          f"{tot_flops / (tot_time * 1e-9) / 1e12:.3f} TFLOP/s")
    print("ideal-occupancy floor (every shape at the best observed rate): "
          f"{tot_flops / (best * 1e12) * 1e3:.3f} ms")


if __name__ == "__main__":
    main()
