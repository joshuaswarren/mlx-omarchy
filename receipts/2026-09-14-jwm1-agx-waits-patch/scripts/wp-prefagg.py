#!/usr/bin/env python3
"""Aggregate the interleaved 1053-token end-to-end prefill A/B.

Three metrics, deliberately separated because they are not equally robust:

  QmmPrefillCoopmatF16 ms  cumulative GPU time of the kernel under test.
  prefill-window GPU busy  GPU busy time inside the prefill span. Immune to
                           host scheduling noise from other agents on the box.
  host wall-clock tok/s    includes host-side driver overhead, which differs
                           between a debugoptimized ICD build and a packaged
                           build, and is sensitive to unrelated CPU load.
"""
import glob
import json
import os
import re
import statistics
import sys

KERN = re.compile(r"QmmPrefillCoopmatF16\s+n=(\d+)\s+total=([0-9.]+)ms")
PREF = re.compile(r"prefill: dispatches=(\d+) gpu_busy=([0-9.]+)ms")

rows = {}
for root in sys.argv[1:]:
    for d in sorted(glob.glob(os.path.join(root, "*-r*"))):
        if not os.path.isdir(d):
            continue
        arm = os.path.basename(d).rsplit("-r", 1)[0]
        try:
            marks = [json.loads(l)
                     for l in open(os.path.join(d, "markers.jsonl"))]
            analysis = open(os.path.join(d, "analysis.txt")).read()
        except OSError:
            continue
        km, pm = KERN.search(analysis), PREF.search(analysis)
        if not km or not pm:
            continue
        m = {r["p"]: r["t"] for r in marks if r["p"] != "tok"}
        pre_ms = (m["prefill_done"] - m["prefill_start"]) / 1e6
        rows.setdefault(arm, []).append({
            "prefill_ms": pre_ms,
            "tok_s": 1053.0 / (pre_ms / 1000.0),
            "qmm_n": int(km.group(1)),
            "qmm_ms": float(km.group(2)),
            "pref_busy_ms": float(pm.group(2)),
            "pref_disp": int(pm.group(1)),
        })

print("| arm | run | QmmPrefillCoopmatF16 ms | n | prefill-window GPU busy ms "
      "| host prefill ms | tok/s |")
print("|---|---:|---:|---:|---:|---:|---:|")
for arm in ("base", "patched"):
    for i, r in enumerate(rows.get(arm, []), 1):
        print("| %s | %d | %.3f | %d | %.3f | %.1f | %.1f |"
              % (arm, i, r["qmm_ms"], r["qmm_n"], r["pref_busy_ms"],
                 r["prefill_ms"], r["tok_s"]))
print()


def med(arm, key):
    return statistics.median(r[key] for r in rows[arm])


LOWER_IS_BETTER = ("qmm_ms", "pref_busy_ms")

for key, label, unit in (
        ("qmm_ms", "QmmPrefillCoopmatF16 GPU time", "ms"),
        ("pref_busy_ms", "prefill-window GPU busy", "ms"),
        ("tok_s", "end-to-end prefill (host wall clock)", "tok/s")):
    b, p = med("base", key), med("patched", key)
    delta = (b - p) / b * 100.0 if key in LOWER_IS_BETTER else (p - b) / b * 100.0
    bs = [r[key] for r in rows["base"]]
    ps = [r[key] for r in rows["patched"]]
    spread = (max(bs) - min(bs)) / statistics.median(bs) * 100.0
    print("%s (%s):" % (label, unit))
    print("   base    median %9.3f  range %9.3f .. %9.3f" % (b, min(bs), max(bs)))
    print("   patched median %9.3f  range %9.3f .. %9.3f" % (p, min(ps), max(ps)))
    print("   patched vs base: %+.2f %%   (within-arm base spread %.1f %%)"
          % (delta, spread))
    if abs(delta) < spread:
        print("   -> delta is smaller than the within-arm spread: no resolvable"
              " difference")
    print()

print("runs per arm:", len(rows["base"]), "base /", len(rows["patched"]),
      "patched")
counts = {r["qmm_n"] for arm in rows for r in rows[arm]}
print("QmmPrefillCoopmatF16 dispatch count identical across every run:",
      counts == {163}, sorted(counts))
disp = {r["pref_disp"] for arm in rows for r in rows[arm]}
print("prefill dispatch count identical across every run:",
      len(disp) == 1, sorted(disp))
