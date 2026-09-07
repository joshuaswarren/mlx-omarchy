#!/usr/bin/env python3
"""Per-kernel device time per decode token from an MLX_OMARCHY_GPU_PROFILE stream.

Usage: dk_decode_table.py <profile.jsonl> <markers.jsonl> <compute.h> [--md]
Selects dispatches whose submission ("s" record) host time lies between the
decode_start and decode_done markers, groups by kernel enum, and reports per
token (= per inter-token interval) counts and device-timestamp totals.
"""
import json, re, sys, statistics

def parse_kernel_names(path):
    names, inside = [], False
    pat = re.compile(r"^\s{2}(\w+),\s*$")
    for line in open(path, encoding="utf-8"):
        if "enum class ComputeKernel" in line:
            inside = True; continue
        if inside:
            m = pat.match(line)
            if m: names.append(m.group(1))
            elif "};" in line: break
    return names

def main():
    prof, markers, header = sys.argv[1:4]
    md = "--md" in sys.argv
    names = parse_kernel_names(header)
    marks = [json.loads(l) for l in open(markers) if l.strip()]
    t_start = next(m["t"] for m in marks if m["p"] == "decode_start")
    t_done = next(m["t"] for m in marks if m["p"] == "decode_done")
    toks = [m["t"] for m in marks if m["p"] == "tok"]
    intervals = len(toks) - 1
    period = 1.0
    sub_time = {}
    disp = []
    for line in open(prof):
        line = line.strip()
        if not line: continue
        r = json.loads(line)
        k = r.get("k")
        if k == "meta": period = r.get("period_ns", 1.0)
        elif k == "s": sub_time[r["s"]] = r["t"]
        elif k == "d": disp.append(r)
    per = {}
    grids = {}
    n_total = 0
    for d in disp:
        st = sub_time.get(d["s"])
        if st is None or st < t_start or st > t_done: continue
        n_total += 1
        dur = (d["t1"] - d["t0"]) * period
        per.setdefault(d["e"], []).append(dur)
        grids.setdefault(d["e"], set()).add((d.get("gx"), d.get("gy"), d.get("gz"), d.get("n")))
    rows = []
    for e, durs in per.items():
        name = names[e] if e < len(names) else f"enum{e}"
        tot = sum(durs)
        rows.append((tot / intervals, name, len(durs) / intervals, statistics.mean(durs), statistics.median(durs), tot, sorted(grids[e])[:4]))
    rows.sort(reverse=True)
    busy = sum(r[5] for r in rows) / intervals
    print(f"decode window: dispatches={n_total} intervals={intervals} dispatches/token={n_total/intervals:.1f} device-busy/token={busy/1e3:.1f} us")
    if md:
        print("| kernel | n/token | mean us | p50 us | total us/token | share |")
        print("|---|---|---|---|---|---|")
        for tpt, name, npt, mean, med, tot, g in rows:
            print(f"| {name} | {npt:.1f} | {mean/1e3:.1f} | {med/1e3:.1f} | {tpt/1e3:.1f} | {100*tpt/busy:.1f}% |")
    else:
        for tpt, name, npt, mean, med, tot, g in rows:
            print(f"{name:28s} n/tok={npt:6.1f} mean={mean/1e3:6.1f}us p50={med/1e3:6.1f}us total/tok={tpt/1e3:8.1f}us share={100*tpt/busy:5.1f}% grids={g}")

if __name__ == "__main__":
    main()
