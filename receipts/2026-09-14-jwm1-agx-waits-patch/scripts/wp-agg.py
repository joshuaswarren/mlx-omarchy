#!/usr/bin/env python3
"""Aggregate the interleaved A/B: per-cell paired medians and the digest gate.

The digest gate is a hazard test, not a checkbox. An under-inserted wait is
a race, so it can pass once and fail under different timing. Any digest that
differs between arms, or varies across rounds within an arm, is a failure.
"""
import collections
import json
import statistics
import sys

rows = []
for line in open(sys.argv[1]):
    line = line.strip()
    if not line.startswith("{"):
        continue
    rows.append(json.loads(line))

by = collections.defaultdict(lambda: collections.defaultdict(list))
digests = collections.defaultdict(lambda: collections.defaultdict(set))
for r in rows:
    by[r["shape"]][r["arm_name"]].append(r["median_ms"])
    digests[r["shape"]][r["arm_name"]].add(r["f16_digest"])

ORDER = ["262x896x128", "1053x896x128", "262x896x896", "1053x896x896",
         "262x4864x896", "1053x4864x896", "262x896x9728", "1053x896x9728"]
shapes = [s for s in ORDER if s in by] + [s for s in by if s not in ORDER]

print("| cell | base ms | patched ms | delta %% | base GF | patched GF | digest |")
print("|---|---:|---:|---:|---:|---:|---|")

gate_ok = True
for s in shapes:
    b = statistics.median(by[s]["base"])
    p = statistics.median(by[s]["patched"])
    m, k, n = (int(v) for v in s.split("x"))
    gfb = 2.0 * m * n * k / (b * 1e6)
    gfp = 2.0 * m * n * k / (p * 1e6)
    delta = (b - p) / b * 100.0
    ds = digests[s]["base"] | digests[s]["patched"]
    ok = len(ds) == 1
    gate_ok &= ok
    tag = sorted(ds)[0] if ok else "MISMATCH " + " vs ".join(sorted(ds))
    print("| %s | %.4f | %.4f | %+.2f | %.1f | %.1f | %s |"
          % (s, b, p, delta, gfb, gfp, tag))

print()
print("rounds per arm:", len(by[shapes[0]]["base"]))
print("digest gate:", "PASS - all cells bit-identical across arms and rounds"
      if gate_ok else "FAIL")

dom = "1053x896x9728"
if dom in by:
    b = by[dom]["base"]
    p = by[dom]["patched"]
    print()
    print("dominant cell %s per-round medians" % dom)
    print("  base   :", ", ".join("%.4f" % v for v in b))
    print("  patched:", ", ".join("%.4f" % v for v in p))
    print("  base    median %.4f  min %.4f" % (statistics.median(b), min(b)))
    print("  patched median %.4f  min %.4f" % (statistics.median(p), min(p)))

big = [s for s in shapes if s.startswith("1053")]
print()
print("1053-row cells only (the verdict set; x896x128 is work-starved):")
for s in big:
    b = statistics.median(by[s]["base"])
    p = statistics.median(by[s]["patched"])
    print("  %-18s %+.2f %%" % (s, (b - p) / b * 100.0))
