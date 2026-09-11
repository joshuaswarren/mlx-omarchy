#!/usr/bin/env python3
"""Summarize window-2 gap legs: per-side widedep/iso4 medians vs base."""
import json, sys, glob

def load(paths):
    rows = {}
    for p in paths:
        for line in open(p):
            try:
                j = json.loads(line)
            except json.JSONDecodeError:
                continue
            if j.get("k") == "gap":
                rows[(j["arm"], j["pass"])] = j
    return rows

def med(arm, legs):
    vals = [legs[a]["layer_ns_med"] for a in legs]
    vals.sort()
    return vals[len(vals)//2], legs

for tag, pattern in (("A", "*window2-gap-a-*.ndjson"), ("B", "*window2-gap-b-*.ndjson")):
    paths = sorted(glob.glob(pattern))
    if not paths:
        continue
    legs = {}
    for p in paths:
        for line in open(p):
            try:
                j = json.loads(line)
            except json.JSONDecodeError:
                continue
            if j.get("k") == "gap":
                legs.setdefault(j["arm"], []).append(j["layer_ns_med"])
    print(f"== leg {tag} ({paths[0].split('/')[-1]}) ==")
    base = legs.get("widedep", [])
    for arm in sorted(legs):
        vals = legs[arm]
        m = sorted(vals)[len(vals)//2]
        gb = 8427008 / (m * 1e-9) / 1e9
        ratio = (f"  vs base_widedep {m / sorted(base)[len(base)//2]:.4f}"
                 if base and arm.startswith(("widedep_", "widedep")) else "")
        print(f"{arm:16s} passes={len(vals)} med_us/layer={m/1000:8.1f} gb_s={gb:5.2f}{ratio}")
