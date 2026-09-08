#!/usr/bin/env python3
"""usage: probe2.py <icd.json> <tag>; exact case-12/13 sinh/cosh block."""
import os, subprocess, sys, json
import numpy as np
icd, tag = sys.argv[1], sys.argv[2]
here = os.path.dirname(os.path.abspath(__file__))
rows = [("sinh(89,.25)", 89.0, 0.25, 12), ("cosh(89,.25)", 89.0, 0.25, 13),
        ("sinh(-89,-.25)", -89.0, -0.25, 12), ("cosh(-89,-.25)", -89.0, -0.25, 13),
        ("sinh(44,.25)", 44.0, 0.25, 12), ("sinh(88,.25)", 88.0, 0.25, 12),
        ("sinh(2,.25)", 2.0, 0.25, 12)]
n = 64
inp = np.zeros((n, 4), dtype=np.float32)
for i, (_, a, b, op) in enumerate(rows):
    inp[i] = (a, b, op, 0)
inp.tofile(os.path.join(here, "in2.bin"))
env = dict(os.environ, VK_ICD_FILENAMES=icd, AGX_SIMDMAT="1")
out_path = os.path.join(here, f"out2-{tag}.bin")
cmd = ["flock", "/tmp/m1-gpu.lock", os.path.join(here, "mathrepro"),
       os.path.join(here, "probe2.spv"), os.path.join(here, "in2.bin"),
       out_path, str(n * 16 * 4), "1"]
r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120)
print(r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "", file=sys.stderr)
if r.returncode != 0:
    print(r.stdout, r.stderr); sys.exit(r.returncode)
out = np.fromfile(out_path, dtype=np.float32).reshape(n, 16)
names = ["value.x", "value.y", "h", "inverse", "sh", "ch", "f.x", "f.y", "fh.x", "fh.y",
         "t", "exp|x|", "exp2form", "sx", "cx", "sign"]
res = {"tag": tag, "rows": {}}
for i, (label, a, b, op) in enumerate(rows):
    row = {nm: repr(float(out[i, j])) for j, nm in enumerate(names)}
    res["rows"][label] = row
    print(f"{label:16s}", " ".join(f"{k}={v}" for k, v in row.items()))
json.dump(res, open(os.path.join(here, f"probe2-{tag}.json"), "w"), indent=1)
