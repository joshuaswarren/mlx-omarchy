#!/usr/bin/env python3
"""Jw16DecodeNorm: rate/digest table over a gate window's json files.
usage: norm_cells_table.py DIR [PIN64 PIN128 PIN256 PIN512]
Prints one line per qwen-gpu json: file, decode median, digest prefix, and
PIN/MISMATCH/1PASS flag against the production pins when the length matches.
"""
import json
import re
import sys
from pathlib import Path

d = Path(sys.argv[1])
pins = {}
for arg in sys.argv[2:5]:
    pass
PINS = {
    64: "c84b3e7af640",
    128: "07c515e0338b",
    256: "c6aabbf0a51d",
    512: "5c120987f0e5",
}
for f in sorted(d.glob("*.json")):
    try:
        j = json.load(open(f))
    except Exception as e:
        print(f.name, "PARSE-FAIL", repr(e))
        continue
    med = j.get("decode_tok_rate", {}).get("median")
    dg = (j.get("ordered_records_sha256") or "?")[:12]
    m = re.search(r"-d(\d+)", f.name)
    flag = ""
    if m:
        n = int(m.group(1))
        want = PINS.get(n)
        if want:
            flag = "PIN" if dg.startswith(want[: len(dg)]) else "DIGEST-MISMATCH"
    print(f"{f.name:28s} {str(med):>8s} {dg} {flag}")
