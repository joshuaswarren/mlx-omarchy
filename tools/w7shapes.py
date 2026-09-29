#!/usr/bin/env python3
"""Compare gemv_shapes decode rows across arms. usage: w7shapes.py <outdir>"""
import json
import sys

d = sys.argv[1]
arms = ("prod", "off", "on")
data = {}
for a in arms:
    try:
        data[a] = json.load(open(f"{d}/shapes-{a}.json"))
    except OSError:
        data[a] = {}
dec = {}
for a in arms:
    body = data[a]
    if isinstance(body, dict):
        for key, row in body.items():
            if key.startswith("decode") and isinstance(row, dict):
                dec.setdefault(key, {})[a] = row.get("GB_s")
for key, vals in sorted(dec.items()):
    p, o, n = vals.get("prod"), vals.get("off"), vals.get("on")
    if p and n:
        print(f"{key:28s} prod {p:7.1f}  off {o or 0:7.1f}  on {n:7.1f}  on/prod {n / p:5.2f}")
