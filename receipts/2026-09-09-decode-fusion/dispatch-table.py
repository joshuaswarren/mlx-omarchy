#!/usr/bin/env python3
"""Every dispatch of one decode token from an MLX_OMARCHY_GPU_PROFILE
stream, in GPU order, with kernel, shape, GPU time, and the gap before
it, grouped by layer. A token is the run of submissions between two
consecutive `tok` markers (decode-window attribution, see
scripts/profile_analyze.py); layer boundaries are the FastRmsNorm
dispatches (two per layer, then the final norm).

usage: dispatch-table.py PROFILE.jsonl MARKERS.jsonl COMPUTE_H [--token N]
       [--json OUT.json]
"""
import argparse
import json
import re
import sys


def kernel_names(header):
    names, inside = [], False
    for line in open(header, encoding="utf-8"):
        if "enum class ComputeKernel" in line:
            inside = True
            continue
        if inside:
            m = re.match(r"^\s{2}(\w+),\s*$", line)
            if m:
                names.append(m.group(1))
            elif "};" in line:
                break
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile")
    ap.add_argument("markers")
    ap.add_argument("compute_h")
    ap.add_argument("--token", type=int, default=10,
                    help="inter-token interval index (1-based)")
    ap.add_argument("--json")
    args = ap.parse_args()
    names = kernel_names(args.compute_h)
    marks = [json.loads(l) for l in open(args.markers)]
    toks = [m["t"] for m in marks if m["p"] == "tok"]
    lo, hi = toks[args.token - 1], toks[args.token]
    submits, dispatches = {}, []
    for line in open(args.profile):
        e = json.loads(line)
        if e["k"] == "s":
            submits[e["s"]] = e["t"]
        elif e["k"] == "d":
            dispatches.append(e)
    chosen = [d for d in dispatches if lo <= submits.get(d["s"], -1) < hi]
    chosen.sort(key=lambda d: d["t0"])
    rows, layer, prev_end, norms = [], 0, None, 0
    for d in chosen:
        name = names[d["e"]] if d["e"] < len(names) else f"kernel{d['e']}"
        if name.startswith("FastRmsNorm"):
            norms += 1
            layer = (norms + 1) // 2 if norms <= 48 else 25
        gap = (d["t0"] - prev_end) if prev_end is not None else 0
        rows.append({
            "layer": layer, "kernel": name, "n": d["n"],
            "groups": [d["gx"], d["gy"], d["gz"]],
            "gpu_us": round((d["t1"] - d["t0"]) / 1e3, 1),
            "gap_us": round(gap / 1e3, 1), "submission": d["s"],
            "bindings_bytes": [b[2] for b in d["b"]],
        })
        prev_end = d["t1"]
    subs = sorted({r["submission"] for r in rows})
    total_gpu = sum(r["gpu_us"] for r in rows)
    total_gap = sum(r["gap_us"] for r in rows)
    span = (chosen[-1]["t1"] - chosen[0]["t0"]) / 1e3 if chosen else 0
    print(f"token interval {args.token}: {len(rows)} dispatches in "
          f"{len(subs)} submissions {subs}; GPU busy {total_gpu:.0f} us, "
          f"gaps {total_gap:.0f} us, span {span:.0f} us")
    by_kernel = {}
    for r in rows:
        k = by_kernel.setdefault(r["kernel"], [0, 0.0, 0.0])
        k[0] += 1
        k[1] += r["gpu_us"]
        k[2] += r["gap_us"]
    print(f"{'kernel':28s} {'n':>4s} {'gpu_us':>9s} {'gap_us':>9s} {'mean_us':>8s}")
    for k, (n, g, gap) in sorted(by_kernel.items(), key=lambda kv: -kv[1][1]):
        print(f"{k:28s} {n:4d} {g:9.0f} {gap:9.0f} {g / n:8.1f}")
    print()
    print(f"{'#':>4s} {'layer':>5s} {'sub':>4s} {'kernel':28s} {'n':>7s} "
          f"{'groups':>12s} {'gap_us':>7s} {'gpu_us':>7s}")
    for i, r in enumerate(rows):
        print(f"{i:4d} {r['layer']:5d} {r['submission']:4d} {r['kernel']:28s} "
              f"{r['n']:7d} {'x'.join(map(str, r['groups'])):>12s} "
              f"{r['gap_us']:7.1f} {r['gpu_us']:7.1f}")
    if args.json:
        json.dump({"token_interval": args.token, "dispatches": len(rows),
                   "submissions": subs, "gpu_busy_us": round(total_gpu),
                   "gap_us": round(total_gap), "span_us": round(span),
                   "by_kernel": {k: {"n": v[0], "gpu_us": round(v[1]),
                                     "gap_us": round(v[2])}
                                 for k, v in by_kernel.items()},
                   "rows": rows}, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
