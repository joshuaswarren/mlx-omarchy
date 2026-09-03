#!/usr/bin/env python3
"""Median prefill/decode tok/s per leg tag from ~/benchq/logs/<tag>.runN.log.

usage: parse_legs.py <regex-for-prompt> <regex-for-generation> <tag> [<tag>...]
Prints per-run numbers then the median for each tag.
"""
import glob
import os
import re
import statistics
import sys

PROMPT_RE = re.compile(sys.argv[1])
GEN_RE = re.compile(sys.argv[2])

for tag in sys.argv[3:]:
    rows = []
    for f in sorted(glob.glob(os.path.expanduser(
            f"~/benchq/logs/{tag}.run*.log"))):
        text = open(f, errors="replace").read()
        p = PROMPT_RE.search(text)
        g = GEN_RE.search(text)
        rows.append((os.path.basename(f), p and p.group(1), g and g.group(1)))
    pre = [float(r[1]) for r in rows if r[1]]
    gen = [float(r[2]) for r in rows if r[2]]
    print(f"[{tag}]")
    for name, p, g in rows:
        print(f"  {name}: prompt={p} gen={g}")
    if pre:
        print(f"  MEDIAN prompt={statistics.median(pre)}")
    if gen:
        print(f"  MEDIAN gen={statistics.median(gen)}")
