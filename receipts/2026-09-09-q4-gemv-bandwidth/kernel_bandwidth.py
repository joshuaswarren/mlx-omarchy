#!/usr/bin/env python3
"""Q4 decode GEMV kernel micro-benchmark on the diagnostics wheel.

  kernel_bandwidth.py run PYTHON OUT_DIR REPO [--reps N] [--variant NAME ...]
  kernel_bandwidth.py compare REF.npz OTHER.npz [...]

`run` executes, for every variant ("default" for the wheel's own kernel;
other names select the rows x subgroups screening builds that existed
only in the screening commits 61f98e4..0dc2074 of wave/GemvBandwidth
through MLX_OMARCHY_Q4_GEMV_SCREEN), one
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
if os.environ.get("Q4_SHAPES"):
    SHAPES = [tuple(t) for t in json.loads(os.environ["Q4_SHAPES"])]
EXACT_SHAPES = [(896, 7), (896, 37), (64, 9), (4864, 12)]

WORKER = r'''
import json, os, sys, time
import numpy as np
import mlx.core as mx
out = sys.argv[1]; reps = int(sys.argv[2])
shapes = [tuple(t) for t in json.loads(sys.argv[3])]; exact = [tuple(t) for t in json.loads(sys.argv[4])]
LAYERS = 24  # distinct weight copies per shape so each dispatch streams cold weights, as decode does
rng = np.random.default_rng(7)
def make(k, n, copies):
    x = mx.array(rng.standard_normal((1, k)).astype(np.float32)).astype(mx.float16)
    ws = []
    for _ in range(copies):
        w = mx.array(rng.integers(0, 2**32, size=(n, k // 8), dtype=np.uint64).astype(np.uint32))
        s = mx.array(rng.uniform(0.01, 0.1, size=(n, k // 64)).astype(np.float32)).astype(mx.float16)
        b = mx.array(rng.uniform(-0.5, 0.5, size=(n, k // 64)).astype(np.float32)).astype(mx.float16)
        mx.eval(w, s, b)
        ws.append((w, s, b))
    mx.eval(x)
    return x, ws
def qmm(x, w, s, b):
    return mx.quantized_matmul(x, w, s, b, transpose=True, group_size=64, bits=4)
saved = {}
for k, n in exact:
    x, ws = make(k, n, 1)
    y = qmm(x, *ws[0]); mx.eval(y)
    saved[f"q4_{k}x{n}"] = np.array(y.astype(mx.float32))
inputs = {}
for k, n in shapes:
    inputs[(k, n)] = make(k, n, 1 if n * k >= 64 << 20 else LAYERS)
    y = qmm(inputs[(k, n)][0], *inputs[(k, n)][1][0]); mx.eval(y)
    saved[f"q4_{k}x{n}"] = np.array(y.astype(mx.float32))
nbytes = 4864 * 896 // 2
a32_small = mx.array(rng.standard_normal(nbytes // 4).astype(np.float32))
a32_big = mx.array(rng.standard_normal(151936 * 896 // 4).astype(np.float32))
a32_tiny = mx.array(rng.standard_normal(4096).astype(np.float32))
mx.eval(a32_small, a32_big, a32_tiny)
# Warm the GPU clock with sustained work, then one submission stream of
# every shape interleaved (the decode pattern) plus the cast ceilings.
big = inputs[max(shapes, key=lambda t: t[0] * t[1])]
mx.eval(*[qmm(big[0], *big[1][0]) for _ in range(30)])
t0 = time.perf_counter()
outs = []
for r in range(reps):
    for (k, n), (x, ws) in inputs.items():
        for w, s, b in ws:
            outs.append(qmm(x, w, s, b))
    outs.append(a32_small.astype(mx.float16))
    outs.append(a32_big.astype(mx.float16))
    outs.append(a32_tiny.astype(mx.float16))
mx.eval(*outs)
wall = {"stream_s": time.perf_counter() - t0, "dispatches": len(outs)}
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
    # Cast f32->f16 as the streaming ceiling (reads 4 B, writes 2 B per
    # element) at the per-layer weight size and the lm_head size, and on
    # 4096 elements as the per-dispatch floor.
    for label, elems in (("ceiling_cast_2mb", 4864 * 896 // 4),
                         ("ceiling_cast_76mb", 151936 * 896 // 4),
                         ("floor_cast_4k", 4096)):
        hits = [r for r in dispatch if names[r["e"]] == "CastF32F16" and r["n"] == elems]
        durs = sorted((r["t1"] - r["t0"]) * period for r in hits)
        if durs:
            result[label] = {
                "kernel": "CastF32F16", "elements": elems, "dispatches": len(durs),
                "bytes_traffic": elems * 6,
                "p50_us": statistics.median(durs) / 1e3, "min_us": durs[0] / 1e3,
                "gbps_p50": elems * 6 / statistics.median(durs),
                "gbps_min_time": elems * 6 / durs[0]}
    return result


def run(argv):
    py, out_dir, repo = Path(argv[0]), Path(argv[1]), Path(argv[2])
    reps, variants = 4, []
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
        print(v, json.dumps({k: [round(r["p50_us"], 1), round(r["gbps_p50"], 2)] for k, r in report["variants"][v].items()
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
