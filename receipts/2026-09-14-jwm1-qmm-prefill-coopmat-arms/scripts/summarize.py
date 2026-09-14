"""Summarize the interleaved arm screen as Markdown.

Per (arm, shape) it takes the median of the per-round probe medians,
compares it with the base arm, and checks that every arm and round
produced one f16 output digest per shape.
"""
import collections
import json
import sys

SHAPES = [
    ("262x896x128", "k/v projection, short"),
    ("1053x896x128", "k/v projection, 1K"),
    ("262x896x896", "q/o projection, short"),
    ("1053x896x896", "q/o projection, 1K"),
    ("262x4864x896", "down projection, short"),
    ("1053x4864x896", "down projection, 1K"),
    ("262x896x9728", "gate+up projection, short"),
    ("1053x896x9728", "gate+up projection, 1K (dominant)"),
]


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def gflops(shape, ms):
    m, k, n = (int(v) for v in shape.split("x"))
    return 2.0 * m * k * n / (ms * 1e6)


def main() -> int:
    rows = [json.loads(l) for l in open(sys.argv[1]) if "median_ms" in l]
    out = open(sys.argv[2], "w")
    ms = collections.defaultdict(list)
    dig = collections.defaultdict(set)
    arms = []
    for r in rows:
        ms[(r["name"], r["shape"])].append(r["median_ms"])
        dig[(r["name"], r["shape"])].add(r["f16_digest"])
        if r["name"] not in arms:
            arms.append(r["name"])
    print("| cell | " + " | ".join(arms) + " |", file=out)
    print("| --- | " + " | ".join("---:" for _ in arms) + " |", file=out)
    for shape, _ in SHAPES:
        base = median(ms[("base", shape)])
        cells = []
        for a in arms:
            m = median(ms[(a, shape)])
            if a == "base":
                cells.append(f"{m:.3f} ms / {gflops(shape, m):.0f}")
            else:
                cells.append(f"{m:.3f} ms / {100.0 * (base / m - 1.0):+.1f}%")
        print(f"| {shape} | " + " | ".join(cells) + " |", file=out)
    print(file=out)
    print("Base column is median ms and GFLOP/s; arm columns are median ms"
          " and speed delta against base (positive is faster).", file=out)
    print(file=out)
    digests = {}
    bad = {}
    for shape, _ in SHAPES:
        seen = {d for a in arms for d in dig[(a, shape)]}
        digests[shape] = sorted(seen)
        if len(seen) != 1:
            bad[shape] = sorted(seen)
    print("| cell | f16 output digest, all arms and rounds |", file=out)
    print("| --- | --- |", file=out)
    for shape, _ in SHAPES:
        print(f"| {shape} | `{digests[shape][0]}`"
              f"{'' if len(digests[shape]) == 1 else ' MISMATCH'} |", file=out)
    print(file=out)
    print("digest equality:", "PASS" if not bad else f"FAIL {bad}", file=out)
    print("rounds per arm:",
          {a: len(ms[(a, SHAPES[-1][0])]) for a in arms}, file=out)
    out.close()
    print(open(sys.argv[2]).read())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
