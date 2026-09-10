#!/usr/bin/env python3
"""Profile dense BF16 GEMV and same-traffic cast ceilings."""

import hashlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

SHAPES = [(896, 896), (896, 128), (896, 4864), (4864, 896), (896, 151936)]

WORKER = r'''
import json, os, sys, time
import numpy as np
import mlx.core as mx

out = sys.argv[1]
reps = int(sys.argv[2])
shapes = [tuple(v) for v in json.loads(sys.argv[3])]
LAYERS = 24

def bits(count, salt):
    a = np.empty(count, dtype=np.uint16)
    chunk = 1 << 20
    for start in range(0, count, chunk):
        stop = min(start + chunk, count)
        i = np.arange(start, stop, dtype=np.uint32)
        mixed = i + np.uint32((salt * 0x9e3779b9) & 0xffffffff)
        mixed ^= mixed >> np.uint32(16)
        mixed *= np.uint32(0x7feb352d)
        mixed ^= mixed >> np.uint32(15)
        mixed *= np.uint32(0x846ca68b)
        mixed ^= mixed >> np.uint32(16)
        mant = (mixed >> np.uint32(9)) & np.uint32(0x7f)
        exp = np.uint32(120) + ((mixed >> np.uint32(1)) & np.uint32(7))
        sign = (mixed & np.uint32(1)) << np.uint32(15)
        a[start:stop] = (sign | (exp << np.uint32(7)) | mant).astype(np.uint16)
    return a

def make(k, n, copies):
    x = mx.array(bits(k, 11).reshape(1, k)).view(mx.bfloat16)
    ws = [mx.array(bits(k * n, 23 + j).reshape(n, k)).view(mx.bfloat16)
          for j in range(copies)]
    mx.eval(x, *ws)
    return x, ws

def gemv(x, w):
    return x @ w.T

inputs = {}
saved = {}
for k, n in shapes:
    copies = 1 if n * k >= 64 << 20 else LAYERS
    inputs[(k, n)] = make(k, n, copies)
    y = gemv(inputs[(k, n)][0], inputs[(k, n)][1][0])
    mx.eval(y)
    saved[f"bf16_{k}x{n}"] = np.array(y.view(mx.uint16))

casts = {}
for k, n in shapes:
    traffic = (k * n + k) * 2
    elems = max(1, (traffic + 5) // 6)
    casts[(k, n)] = mx.array(np.arange(elems, dtype=np.float32))
mx.eval(*casts.values())

big = inputs[max(shapes, key=lambda s: s[0] * s[1])]
mx.eval(*[gemv(big[0], big[1][0]) for _ in range(20)])
outs = []
t0 = time.perf_counter()
for _ in range(reps):
    for k, n in shapes:
        x, ws = inputs[(k, n)]
        outs.extend(gemv(x, w) for w in ws)
        outs.append(casts[(k, n)].astype(mx.bfloat16))
mx.eval(*outs)
np.savez(out, **saved)
json.dump({"version": mx.__version__, "stream_s": time.perf_counter() - t0,
           "dispatches": len(outs)}, open(out + ".meta.json", "w"))
'''


def kernel_names(repo):
    sys.path.insert(0, str(Path(repo) / "scripts"))
    from profile_analyze import parse_kernel_names
    return parse_kernel_names(Path(repo) / "overlay/mlx/backend/omarchy/compute.h")


def bf16_bytes(k, n):
    return (k * n + k) * 2


def analyze(profile, names):
    rows = [json.loads(line) for line in open(profile)]
    period = next(row for row in rows if row.get("k") == "meta")["period_ns"]
    dispatches = [row for row in rows if row.get("k") == "d"]
    result = {}
    for k, n in SHAPES:
        weight_bytes = k * n * 2
        hits = [row for row in dispatches
                if names[row["e"]] in {"MatmulBF16", "MatmulVecBF16"}
                and row["n"] == n and len(row["b"]) >= 2
                and 0 <= row["b"][1][2] - weight_bytes < 65536]
        durations = sorted((row["t1"] - row["t0"]) * period for row in hits)
        if durations:
            traffic = bf16_bytes(k, n)
            result[f"bf16_{k}x{n}"] = {
                "kernel": sorted({names[row["e"]] for row in hits}),
                "workgroups": sorted({row["gx"] for row in hits}),
                "dispatches": len(durations),
                "bytes": traffic,
                "p50_us": statistics.median(durations) / 1e3,
                "min_us": durations[0] / 1e3,
                "gbps_p50": traffic / statistics.median(durations),
                "gbps_min_time": traffic / durations[0],
            }
        elems = max(1, (bf16_bytes(k, n) + 5) // 6)
        cast_hits = [row for row in dispatches
                     if names[row["e"]] == "CastF32BF16" and row["n"] == elems]
        cast_durations = sorted((row["t1"] - row["t0"]) * period for row in cast_hits)
        if cast_durations:
            traffic = elems * 6
            result[f"ceiling_{k}x{n}"] = {
                "kernel": "CastF32BF16",
                "elements": elems,
                "dispatches": len(cast_durations),
                "bytes": traffic,
                "p50_us": statistics.median(cast_durations) / 1e3,
                "min_us": cast_durations[0] / 1e3,
                "gbps_p50": traffic / statistics.median(cast_durations),
                "gbps_min_time": traffic / cast_durations[0],
            }
    return result


def run(argv):
    py, out_dir, repo = Path(argv[0]), Path(argv[1]), Path(argv[2])
    reps = int(argv[3]) if len(argv) > 3 else 4
    out_dir.mkdir(parents=True, exist_ok=True)
    wheel = next(Path(py).parents[1].glob("wheel/*.whl"), None)
    wheel_sha = hashlib.sha256(wheel.read_bytes()).hexdigest() if wheel else os.environ.get("WHEEL_SHA256")
    profile = out_dir / "profile.jsonl"
    npz = out_dir / "outputs.npz"
    env = {k: v for k, v in os.environ.items() if not k.startswith("MLX_")}
    env["MLX_OMARCHY_GPU_PROFILE"] = str(profile)
    env["MLX_DISABLE_COMPILE"] = "1"
    env["MESA_SHADER_CACHE_DISABLE"] = "true"
    subprocess.run([str(py), "-c", WORKER, str(npz), str(reps), json.dumps(SHAPES)],
                   env=env, check=True, timeout=1800)
    meta = json.loads(Path(str(npz) + ".meta.json").read_text())
    report = {"python": str(py), "version": meta["version"], "wheel_sha256": wheel_sha,
              "reps": reps, "wall": meta, "shapes": analyze(profile, kernel_names(repo))}
    (out_dir / "bandwidth.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def compare(argv):
    import numpy as np
    ref = np.load(argv[0])
    ok = True
    for other in argv[1:]:
        candidate = np.load(other)
        for key in ref.files:
            mismatches = int(np.count_nonzero(ref[key] != candidate[key]))
            ok &= mismatches == 0
            print(f"{other} {key} mismatches={mismatches}")
    print("ALL_IDENTICAL" if ok else "MISMATCH")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    {"run": run, "compare": compare}[sys.argv[1]](sys.argv[2:])
