#!/usr/bin/env python3
"""usage: probe.py <icd.json|stock> <tag>; runs probe.spv via mathrepro under the GPU lock."""
import os, subprocess, sys, json
import numpy as np
icd, tag = sys.argv[1], sys.argv[2]
here = os.path.dirname(os.path.abspath(__file__))
FMAX = np.float32(3.4028235e38)
rows = [
    ("fmax/fmax", FMAX, FMAX, 0.9689124, 44.5),
    ("1/fmax", 1.0, FMAX, 0.9689124, 44.5),
    ("intmax-as-float", 2147483648.0, 3000000000.0, 0.0, 0.0),
    ("intmin-as-float", -2147483648.0, -3000000000.0, 0.0, 0.0),
    ("1e30/1e30", 1e30, 1e30, 0.5, 44.5),
    ("0/fmax", 0.0, FMAX, 0.0, 44.5),
    ("fmax/2.8e30", FMAX, 2.8e30, 0.5, 44.5),
    ("fmax/1e37", FMAX, 1e37, 0.5, 44.5),
    ("1/1e38", 1.0, 1e38, 0.5, 44.5),
    ("1/1e37", 1.0, 1e37, 0.5, 44.5),
    ("1/8e37", 1.0, 8e37, 0.5, 44.5),
    ("1/1.7e38", 1.0, 1.7e38, 0.5, 44.5),
    ("neg250", -250.0, -25.0, 0.0, 0.0),
    ("2.5e9", 2.5e9, -2.5e9, 0.0, 0.0),
]
n = 64
inp = np.zeros((n, 4), dtype=np.float32)
for i, (_, a, b, c, d) in enumerate(rows):
    inp[i] = (a, b, c, d)
inp.tofile(os.path.join(here, "in.bin"))
env = dict(os.environ)
if icd != "stock":
    env["VK_ICD_FILENAMES"] = icd
    env["AGX_SIMDMAT"] = "1"
out_path = os.path.join(here, f"out-{tag}.bin")
cmd = ["flock", "/tmp/m1-gpu.lock", os.path.join(here, "mathrepro"),
       os.path.join(here, "probe.spv"), os.path.join(here, "in.bin"),
       out_path, str(n * 16 * 4), "1"]
r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120)
print(r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "", file=sys.stderr)
if r.returncode != 0:
    print(r.stdout, r.stderr); sys.exit(r.returncode)
out = np.fromfile(out_path, dtype=np.float32).reshape(n, 16)
names = ["a/b", "1/b", "a*(1/b)", "exp(d)", "(f*h)*h", "int(a)", "ldexp/m",
         "4/b", "len((a,c)/b)", "uint(a)", "e", "m", "(c*h)*h", "c*(h*h)", "int(b)", "a*b"]
res = {"tag": tag, "rows": {}}
for i, (label, a, b, c, d) in enumerate(rows):
    row = {}
    for j, nm in enumerate(names):
        val = out[i, j]
        if nm in ("int(a)", "int(b)"):
            row[nm] = int(val.view(np.int32))
        elif nm == "uint(a)":
            row[nm] = int(val.view(np.uint32))
        else:
            row[nm] = repr(float(val))
    res["rows"][label] = row
    print(f"{label:16s}", " ".join(f"{k}={v}" for k, v in row.items()))
json.dump(res, open(os.path.join(here, f"probe-{tag}.json"), "w"), indent=1)
