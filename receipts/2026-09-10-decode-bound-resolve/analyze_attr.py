#!/usr/bin/env python3
"""Reduce attr_driver legs.ndjson into the verdict table.

Per workload x arm: medians of wall, thread CPU, process CPU, eval-call
wall/CPU, and the added-GPU wall (gup). Then the two discriminating
slopes at short-decode-32:

  host slope: d(wall/token)/d(spin)   over spin arms 0/1000/4000 us
  gpu slope:  d(wall/token)/d(added GPU wall) over gup arms 0/4/12

Reading: GPU-bound == host slope ~0 AND gpu slope ~1 AND thread CPU far
below wall. Host-paced == host slope ~1 AND gpu slope ~0 AND thread CPU
~ wall.
"""
import json
import statistics
import sys
from pathlib import Path


def med(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return statistics.median(vals) if vals else None


def slope(xs, ys):
    if len(xs) < 2:
        return None
    mx_ = statistics.mean(xs)
    my = statistics.mean(ys)
    den = sum((x - mx_) ** 2 for x in xs)
    return sum((x - mx_) * (y - my) for x, y in zip(xs, ys)) / den


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "results/attr/legs.ndjson")
    rows = [json.loads(l) for l in path.read_text().splitlines() if l]
    keys = ["decode_tok_s", "tok_ms", "thread_cpu_ms_per_token",
            "proc_cpu_ms_per_token", "eval_wall_ms_per_token",
            "eval_thread_cpu_ms_per_token", "gup_wall_ms_per_token"]
    for wl in dict.fromkeys(r["workload"] for r in rows):
        print(f"\n== {wl} ==")
        print(f"{'arm':8s} {'n':>2s} " + " ".join(
            f"{k.replace('_ms_per_token','').replace('_per_token',''):>12s}"
            for k in keys))
        arms = dict.fromkeys(r["arm"] for r in rows if r["workload"] == wl)
        agg = {}
        for arm in arms:
            sel = [r for r in rows if r["workload"] == wl and r["arm"] == arm]
            agg[arm] = sel
            print(f"{arm:8s} {len(sel):2d} " + " ".join(
                f"{(med(sel, k) or 0):12.3f}" for k in keys))
        if wl == "short-decode-32":
            b = med(agg["base"], "tok_ms")
            spins = [0, 1000, 4000]
            sw = [med(agg[a], "tok_ms") for a in ("base", "spin1", "spin4")]
            print(f"host slope: {slope(spins, sw)*1000:.3f} ms wall per ms "
                  f"spin (0 => none of the spin reaches the wall, i.e. the "
                  f"host has slack)")
            gups = [0] + [med(agg[a], "gup_wall_ms_per_token") - med(
                agg[a], "tok_ms") for a in ("gup4", "gup12")]
            gw = [med(agg[a], "tok_ms") for a in ("base", "gup4", "gup12")]
            print(f"gpu slope: {slope(gups, gw):.3f} ms wall per ms added "
                  f"GPU work (1 => added GPU work lands 1:1 on the wall)")
            print(f"baseline thread CPU share of wall: "
                  f"{med(agg['base'], 'thread_cpu_ms_per_token') / b:.2f} "
                  f"(thread), "
                  f"{med(agg['base'], 'proc_cpu_ms_per_token') / b:.2f} "
                  f"(process)")
    print()


if __name__ == "__main__":
    main()
