#!/usr/bin/env python3
"""Rank prefill kernels by GPU slot time from an MLX_OMARCHY_GPU_PROFILE stream.

Honeykrisp writes the profiler's post-dispatch timestamp before the
dispatch has finished executing (t1 - t0 reports 37 us for a 1053 x 896 x
4864 Q4 matmul, an impossible 245 TFLOP/s), so the reported duration is
not the kernel time. The next dispatch's t0 is written after the full
post barrier plus pre barrier, so t0[next] - t0[this] within one
submission is the real wall the dispatch occupied on the GPU, barriers
included. This script aggregates that slot time per kernel for the
dispatches submitted inside the prefill marker window and prints a
ranking with the reported (broken) duration beside it.

Usage: slot_profile.py PROFILE.jsonl --markers MARKERS.jsonl \
           --compute-h overlay/mlx/backend/omarchy/compute.h [--json OUT]
"""

import argparse
import json
import re
import sys
from collections import defaultdict


def kernel_names(path):
    names = []
    pat = re.compile(r"^\s{2}(\w+),\s*$")
    inside = False
    for line in open(path, encoding="utf-8"):
        if "enum class ComputeKernel" in line:
            inside = True
            continue
        if inside:
            m = pat.match(line)
            if m:
                names.append(m.group(1))
            elif "};" in line:
                break
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile")
    ap.add_argument("--markers", required=True)
    ap.add_argument("--compute-h", required=True)
    ap.add_argument("--phase", default="prefill")
    ap.add_argument("--json")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    names = kernel_names(args.compute_h)
    meta = None
    disp = []
    submits = {}
    joins = []
    for line in open(args.profile, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        k = r.get("k")
        if k == "meta":
            meta = r
        elif k == "d":
            disp.append(r)
        elif k == "s":
            submits[r["s"]] = r
        elif k == "j":
            joins.append(r)
    period = meta["period_ns"]

    markers = [json.loads(l) for l in open(args.markers) if l.strip()]
    starts = {}
    for m in markers:
        starts.setdefault(m["p"], m["t"])
    p0 = starts[f"{args.phase}_start"]
    p1 = starts[f"{args.phase}_done"]

    def in_phase(d):
        s = submits.get(d["s"])
        return s is not None and p0 <= s["t"] < p1

    phase = [d for d in disp if in_phase(d) and "t0" in d]
    phase.sort(key=lambda d: (d["s"], d["t0"]))
    # slot time: next t0 in the same submission, else reported t1.
    for i, d in enumerate(phase):
        nxt = phase[i + 1] if i + 1 < len(phase) else None
        if nxt is not None and nxt["s"] == d["s"]:
            d["slot"] = (nxt["t0"] - d["t0"]) * period
        else:
            d["slot"] = (d["t1"] - d["t0"]) * period
        d["rep"] = (d["t1"] - d["t0"]) * period

    by_kernel = defaultdict(lambda: {"n": 0, "slot": 0.0, "rep": 0.0})
    by_sig = defaultdict(lambda: {"n": 0, "slot": 0.0, "rep": 0.0})
    for d in phase:
        name = names[d["e"]] if d["e"] < len(names) else f"kernel_{d['e']}"
        a = by_kernel[name]
        a["n"] += 1
        a["slot"] += d["slot"]
        a["rep"] += d["rep"]
        sig = (name, d["n"], d["gx"], d["gy"], d["gz"])
        b = by_sig[sig]
        b["n"] += 1
        b["slot"] += d["slot"]
        b["rep"] += d["rep"]

    total_slot = sum(v["slot"] for v in by_kernel.values())
    total_rep = sum(v["rep"] for v in by_kernel.values())
    gpu_span = ((phase[-1]["t1"] - phase[0]["t0"]) * period) if phase else 0
    wall = p1 - p0
    phase_submits = [s for s in submits.values() if p0 <= s["t"] < p1]
    phase_joins = [j for j in joins if p0 <= j["t"] < p1]
    join_wait = sum(j["wait"] for j in phase_joins)
    submit_cost = sum(s["dur"] for s in phase_submits)

    out = {
        "phase": args.phase,
        "wall_ms": wall / 1e6,
        "dispatches": len(phase),
        "submissions": len(phase_submits),
        "joins": len(phase_joins),
        "join_wait_ms": join_wait / 1e6,
        "submit_host_ms": submit_cost / 1e6,
        "slot_total_ms": total_slot / 1e6,
        "reported_total_ms": total_rep / 1e6,
        "gpu_busy_fraction_slot": total_slot / wall if wall else 0,
        "kernels": [],
    }
    print(f"== {args.phase}: wall {wall/1e6:.1f} ms, {len(phase)} dispatches, "
          f"{len(phase_submits)} submissions, {len(phase_joins)} joins")
    print(f"   slot total {total_slot/1e6:.1f} ms ({100*total_slot/wall:.1f}% of wall), "
          f"reported t1-t0 total {total_rep/1e6:.1f} ms, join wait {join_wait/1e6:.1f} ms, "
          f"submit host {submit_cost/1e6:.1f} ms")
    print(f"   {'kernel':28s} {'n':>5s} {'slot ms':>9s} {'share':>6s} {'mean us':>8s} {'rep ms':>8s}")
    for name, a in sorted(by_kernel.items(), key=lambda kv: -kv[1]["slot"])[: args.top]:
        row = {
            "kernel": name,
            "n": a["n"],
            "slot_ms": a["slot"] / 1e6,
            "share": a["slot"] / total_slot if total_slot else 0,
            "mean_us": a["slot"] / a["n"] / 1e3,
            "reported_ms": a["rep"] / 1e6,
        }
        out["kernels"].append(row)
        print(f"   {name:28s} {a['n']:5d} {a['slot']/1e6:9.2f} "
              f"{100*row['share']:5.1f}% {row['mean_us']:8.1f} {a['rep']/1e6:8.2f}")
    print(f"   top signatures (kernel, count, gx, gy, gz):")
    out["signatures"] = []
    for sig, b in sorted(by_sig.items(), key=lambda kv: -kv[1]["slot"])[: args.top]:
        out["signatures"].append({
            "kernel": sig[0], "count": sig[1], "gx": sig[2], "gy": sig[3],
            "gz": sig[4], "n": b["n"], "slot_ms": b["slot"] / 1e6,
            "mean_us": b["slot"] / b["n"] / 1e3})
        print(f"   {sig[0]:28s} count={sig[1]:<9d} g=({sig[2]},{sig[3]},{sig[4]}) "
              f"n={b['n']:4d} slot={b['slot']/1e6:8.2f} ms mean={b['slot']/b['n']/1e3:8.1f} us")
    if args.json:
        json.dump(out, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
