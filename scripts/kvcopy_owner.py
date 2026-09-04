#!/usr/bin/env python3
"""Pin the owner of the two growing per-layer cache copies.

Stages, each its own mx.eval under the GPU profiler:
  s1: slice-assign new k/v into a cache array (setitem alone)
  s2: s1 + reading the state slice as a matmul operand (dense consumption)

Any CopyGeneral seen in s1 belongs to the setitem implementation; a copy in
s2/s3 belongs to the state-slice consumer.
"""
import json
import os
import subprocess
import sys

PROF = "/tmp/kvcopy-owner-prof.jsonl"
MARK = "/tmp/kvcopy-owner-markers.jsonl"

script = """
import json, time, mlx.core as mx

mf = open("/tmp/kvcopy-owner-markers.jsonl", "w")
def mark(p):
    mf.write(json.dumps({"t": time.monotonic_ns(), "p": p}) + "\\n"); mf.flush()


cache_k = mx.zeros((1, 2, 256, 64), mx.float16)
cache_v = mx.zeros((1, 2, 256, 64), mx.float16)
newk = mx.ones((1, 2, 1, 64), mx.float16)
newv = mx.ones((1, 2, 1, 64), mx.float16)
off = 41

mark("s1_start")
cache_k[..., off:off+1, :] = newk
cache_v[..., off:off+1, :] = newv
mx.eval(cache_k, cache_v)
mark("s1_done")

mark("s2_start")
state_k = cache_k[..., :off+1, :]
state_v = cache_v[..., :off+1, :]
q = mx.ones((1, 2, 1, 64), mx.float16)
scores = (q @ state_k.transpose(0, 1, 3, 2))
probs = mx.softmax(scores, axis=-1)
out = probs @ state_v
mx.eval(out)
mark("s2_done")

mark("s3_start")
state_k = cache_k[..., :off+1, :]
state_v = cache_v[..., :off+1, :]
q = mx.ones((1, 14, 1, 64), mx.float16) * 0.5
o = mx.fast.scaled_dot_product_attention(q, state_k, state_v, scale=0.125)
mx.eval(o)
mark("s3_done")
print("probe done")
"""

env = dict(os.environ)
env.update({
    "MLX_OMARCHY_ALLOW_NON_APPLE": "1",
    "MLX_DISABLE_COMPILE": "1",
    "MLX_OMARCHY_GPU_PROFILE": PROF,
})
r = subprocess.run([sys.executable, "-c", script], env=env,
                   capture_output=True, text=True)
print(r.stdout.strip())
if r.returncode != 0:
    print(r.stderr[-2000:])
    sys.exit(1)

names = []
in_enum = False
with open(".work/mlx/mlx/backend/omarchy/compute.h") as fh:
    for line in fh:
        if "enum class ComputeKernel" in line:
            in_enum = True
            continue
        if in_enum:
            if "}" in line:
                break
            name = line.strip().rstrip(",")
            if name and name != "Count" and " " not in name:
                names.append(name)

sub_t = {}
events = []
with open(PROF) as fh:
    for line in fh:
        ev = json.loads(line)
        if ev.get("k") == "s":
            sub_t[ev["s"]] = ev["t"]
        elif ev.get("k") == "d":
            events.append((sub_t.get(ev["s"], 0), ev))
events.sort(key=lambda x: x[0])

marks = []
with open(MARK) as fh:
    for line in fh:
        marks.append(json.loads(line))

def stage_of(t):
    stage = None
    for m in marks:
        if m["t"] <= t:
            stage = m["p"]
    return stage

from collections import Counter, defaultdict
per = defaultdict(Counter)
for t, ev in events:
    k = names[ev["e"]] if ev["e"] < len(names) else "?"
    per[stage_of(t)][f"{k}/n={ev['n']}"] += 1

for stage in ["s1_start", "s1_done", "s2_start", "s2_done",
              "s3_start", "s3_done"]:
    if stage in per:
        print(f"== window {stage} ==")
        for k, n in sorted(per[stage].items()):
            print(f"  {k:28s} {n}")
