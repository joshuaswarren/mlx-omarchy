#!/usr/bin/env python3
"""Summarize dispatch-floor bench ndjson files into the verdict table."""
import json
import statistics
import sys
from pathlib import Path


def load(path):
    cases = {}
    for line in open(path):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("k") == "case":
            cases.setdefault(d["name"], []).append(d)
        elif d.get("k") == "hostcase":
            cases.setdefault(d["name"], []).append(d)
    return cases


def med(rows, field):
    return statistics.median(r[field] for r in rows)


def main():
    base = Path(sys.argv[1])
    files = sorted(base.glob("dfb-*.ndjson"))
    keys = [
        "empty_1wg", "empty_144wg", "empty_4096wg",
        "nop_1wg", "nop_4096wg",
        "trivial_1wg", "trivialpc_1wg",
        "local32_1wg", "local512_1wg",
        "barrier_trivial_1wg", "rebind_trivial_1wg",
        "pushconst_trivial_1wg", "pipeswitch_trivial_1wg",
        "tsper_trivial_1wg",
        "nullsubmit", "submitwait_trivial_1wg", "submitwait_trivial_144wg",
    ]
    print(f"{'case':24s}" + "".join(f"{f.stem.replace('dfb-',''):>18s}" for f in files))
    for k in keys:
        row = f"{k:24s}"
        for f in files:
            cases = load(f)
            rows = cases.get(k)
            if not rows:
                row += f"{'-':>18s}"
                continue
            if rows[0].get("k") == "hostcase":
                row += f"{med(rows, 'mean_us'):>18.2f}"
            else:
                w = med(rows, "per_dispatch_wall_us")
                rec = med(rows, "per_dispatch_record_us")
                row += f"{w:>10.2f}/{rec:<7.2f}"
        print(row)
    print("(compute cases: median per-dispatch wall_us/record_us over reps; "
          "hostcase: mean us per round trip)")


if __name__ == "__main__":
    main()
