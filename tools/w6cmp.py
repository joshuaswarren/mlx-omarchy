#!/usr/bin/env python3
"""Compare window 6 arms: medians, digests, bitcheck diffs. usage: w6cmp.py <outdir>"""
import json
import statistics as st
import sys

d = sys.argv[1]

def cell(path):
    try:
        j = json.load(open(path))
    except OSError:
        return [], ""
    dr = j.get("decode_tok_rate") or {}
    med = dr.get("median")
    digs = j.get("ordered_records_sha256") or j.get("digest") or ""
    if med is None:
        return [], digs
    return ([dr.get("min", med), med, dr.get("max", med)], digs)

for c in (64, 128, 256, 512):
    line = f"d{c}:"
    for arm in ("prod", "cand"):
        toks, digs = cell(f"{d}/{arm}-d{c}.json")
        if toks:
            line += (f"  {arm} {st.median(toks):7.2f} "
                     f"({min(toks):.2f}-{max(toks):.2f}) dig={str(digs)[:16]}")
        else:
            line += f"  {arm} NO-TOKS dig={str(digs)[:16]}"
    toks, _ = cell(f"{d}/candoff-d64.json")
    if c == 64 and toks:
        line += f"  candoff {st.median(toks):7.2f}"
    print(line)

for pair in (("bits-prod", "bits-cand"), ("np-prod", "np-cand")):
    try:
        a = json.load(open(f"{d}/{pair[0]}.json"))
        b = json.load(open(f"{d}/{pair[1]}.json"))
    except OSError:
        continue
    keys = sorted(set(a) | set(b))
    diff = [k for k in keys if a.get(k) != b.get(k)]
    print(f"{pair[0]} vs {pair[1]}: {len(keys) - len(diff)}/{len(keys)} identical"
          + (f" DIFF={diff[:8]}" if diff else ""))
