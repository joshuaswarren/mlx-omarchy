#!/usr/bin/env python3
"""Q4 decode GEMV kernel micro-benchmark on the diagnostics wheel.

  kernel_bandwidth.py run PYTHON OUT_DIR REPO [--reps N] [--variant NAME ...]
  kernel_bandwidth.py compare REF.npz OTHER.npz [...]

`run` executes, for every variant (the env value of
MLX_OMARCHY_Q4_GEMV_SCREEN, or "default" for the wheel's own kernel), one
child process under MLX_OMARCHY_GPU_PROFILE that evaluates
mx.quantized_matmul for the model's decode shapes plus two bandwidth
ceilings, then reads the profile back and reports achieved GB/s per
dispatch (bytes = packed words + f16 scales + f16 biases + f16 x). Each
child also saves its outputs so `compare` can check bit identity across
variants and across wheels. Every table row carries the wheel sha256 and
version stamp of the python that produced it.
"""
import hashlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

SHAPES = [(896, 896), (896, 128), (896, 4864), (4864, 896), (896, 151936)]
EXACT_SHAPES = [(896, 7), (896, 37), (64, 9), (4864, 12)]

WORKER = r'''
import json, os, sys, time
import numpy as np
import mlx.core as mx
out = sys.argv[1]; reps = int(sys.argv[2])
shapes = json.loads(sys.argv[3]); exact = json.loads(sys.argv[4])
rng = np.random.default_rng(7)
def make(k, n):
    x = mx.array(rng.standard_normal((1, k)).astype(np.float32)).astype(mx.float16)
    w = mx.array(rng.integers(0, 2**32, size=(n, k // 8), dtype=np.uint64).astype(np.uint32))
    s = mx.array(rng.uniform(0.01, 0.1, size=(n, k // 64)).astype(np.float32)).astype(mx.float16)
    b = mx.array(rng.uniform(-0.5, 0.5, size=(n, k // 64)).astype(np.float32)).astype(mx.float16)
    mx.eval(x, w, s, b)
    return x, w, s, b
saved = {}
wall = {}
for k, n in shapes + exact:
    x, w, s, b = make(k, n)
    y = mx.quantized_matmul(x, w, s, b, transpose=True, group_size=64, bits=4)
    mx.eval(y)
    saved[f"q4_{k}x{n}"] = np.array(y.astype(mx.float32))
    if (k, n) in [tuple(t) for t in shapes]:
        t0 = time.perf_counter()
        for _ in range(reps):
            y = mx.quantized_matmul(x, w, s, b, transpose=True, group_size=64, bits=4)
            mx.eval(y)
        wall[f"q4_{k}x{n}"] = (time.perf_counter() - t0) / reps
# Bandwidth ceilings on the largest per-layer weight (2,179,072 bytes):
# an f16 elementwise identity (reads N, writes N) and an f32 sum (reads N).
nbytes = 4864 * 896 // 2
a16 = mx.array(rng.standard_normal(nbytes // 2).astype(np.float16))
a32 = mx.array(rng.standard_normal(nbytes // 4).astype(np.float32))
mx.eval(a16, a32)
for _ in range(reps):
    mx.eval(a16 * mx.array(1.0, dtype=mx.float16))
for _ in range(reps):
    mx.eval(mx.sum(a32))
np.savez(out, **saved)
json.dump({"version": mx.__version__, "wall_s": wall}, open(out + ".meta.json", "w"))
'''


def kernel_names(repo):
    sys.path.insert(0, str(Path(repo) / "scripts"))
    from profile_analyze import parse_kernel_names
    return parse_kernel_names(Path(repo) / "overlay/mlx/backend/omarchy/compute.h")


def q4_bytes(k, n):
    return n * k // 2 + 2 * (n * (k // 64) * 2) + k * 2


def analyze(profile, names):
    rows = [json.loads(l) for l in open(profile)]
    period = next(r for r in rows if r.get("k") == "meta")["period_ns"]
    dispatch = [r for r in rows if r.get("k") == "d"]
    result = {}
    for k, n in SHAPES:
        # Binding 1 is w; the allocator may round its buffer up, so match
        # the packed size within one page.
        hits = [r for r in dispatch if names[r["e"]].startswith("QmmVecQ4")
                and r["n"] == n and 0 <= r["b"][1][2] - n * k // 2 < 65536]
        durs = sorted((r["t1"] - r["t0"]) * period for r in hits)
        if not durs:
            continue
        nbytes = q4_bytes(k, n)
        gx = {r["gx"] for r in hits}
        kern = {names[r["e"]] for r in hits}
        result[f"q4_{k}x{n}"] = {
            "kernel": sorted(kern), "workgroups": sorted(gx), "dispatches": len(durs),
            "bytes": nbytes,
            "p50_us": statistics.median(durs) / 1e3, "min_us": durs[0] / 1e3,
            "gbps_p50": nbytes / statistics.median(durs),
            "gbps_min_time": nbytes / durs[0]}
    nbytes = 4864 * 896 // 2
    for label, prefix, traffic in (("ceiling_elementwise_f16", "Elementwise", 2 * nbytes),
                                   ("ceiling_reduce_f32", "Reduce", nbytes)):
        durs = sorted((r["t1"] - r["t0"]) * period for r in dispatch
                      if names[r["e"]].startswith(prefix) and any(b[2] >= nbytes for b in r["b"]))
        if durs:
            result[label] = {
                "kernel": sorted({names[r["e"]] for r in dispatch if names[r["e"]].startswith(prefix)}),
                "dispatches": len(durs), "bytes_traffic": traffic,
                "p50_us": statistics.median(durs) / 1e3, "min_us": durs[0] / 1e3,
                "gbps_p50": traffic / statistics.median(durs),
                "gbps_min_time": traffic / durs[0]}
    return result


def run(argv):
    py, out_dir, repo = Path(argv[0]), Path(argv[1]), Path(argv[2])
    reps, variants = 20, []
    it = iter(argv[3:])
    for a in it:
        if a == "--reps":
            reps = int(next(it))
        elif a == "--variant":
            variants.append(next(it))
    variants = variants or ["default"]
    out_dir.mkdir(parents=True, exist_ok=True)
    names = kernel_names(repo)
    dist = json.loads(subprocess.run(
        [str(py), "-c", "import json,mlx.core as mx,importlib.metadata as m;"
         "d=m.distribution('mlx-omarchy');print(json.dumps({'version':mx.__version__,"
         "'files':[str(f) for f in d.files if str(f).endswith('.so')][:1]}))"],
        check=True, capture_output=True, text=True).stdout)
    report = {"python": str(py), "version": dist["version"], "reps": reps, "variants": {}}
    for v in variants:
        env = {k: val for k, val in os.environ.items() if not k.startswith("MLX_")}
        env["MLX_OMARCHY_GPU_PROFILE"] = str(out_dir / f"{v}.profile.jsonl")
        env["MESA_SHADER_CACHE_DISABLE"] = "true"
        if v != "default":
            env["MLX_OMARCHY_Q4_GEMV_SCREEN"] = v
        npz = out_dir / f"{v}.npz"
        subprocess.run([str(py), "-c", WORKER, str(npz), str(reps),
                        json.dumps(SHAPES), json.dumps(EXACT_SHAPES)],
                       env=env, check=True, timeout=1800)
        meta = json.loads((out_dir / f"{v}.npz.meta.json").read_text())
        report["variants"][v] = {"wall_s": meta["wall_s"],
                                 **analyze(out_dir / f"{v}.profile.jsonl", names)}
        print(v, json.dumps({k: round(r.get("gbps_p50", 0), 2) for k, r in report["variants"][v].items()
                             if isinstance(r, dict) and "gbps_p50" in r}), flush=True)
    (out_dir / "bandwidth.json").write_text(json.dumps(report, indent=2) + "\n")
    print("BANDWIDTH_DONE", flush=True)


def compare(argv):
    import numpy as np
    ref = np.load(argv[0])
    ok = True
    for other in argv[1:]:
        o = np.load(other)
        for key in ref.files:
            same = np.array_equal(ref[key].view(np.uint32), o[key].view(np.uint32))
            ok &= same
            print(f"{other} {key} {'identical' if same else 'DIFFERENT'}")
    print("ALL_IDENTICAL" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    {"run": run, "compare": compare}[sys.argv[1]](sys.argv[2:])
