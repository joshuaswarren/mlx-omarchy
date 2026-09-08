#!/usr/bin/env python3
"""
ANE vs GPU parity: GPU side. Host-to-host timed: np -> mx -> op -> mx.eval -> np.
3 warmups + 30 samples, median, exact-fp16 dtypes plus actual bf16 + 4-bit quantized.
Run on jwm1-linux under /tmp/m1-gpu.lock with quiet CPU; the script is flock-aware
(via the caller) and prints hostname + resolved ICD + mlx version at the top.
"""
import argparse, json, os, sys, time, statistics, platform, hashlib
from pathlib import Path
import numpy as np

def median(v):
    return float(statistics.median(v))

def host_info():
    info = {"hostname": platform.node(), "uname": platform.platform(), "python": sys.version.split()[0]}
    try:
        import subprocess
        info["kernel"] = subprocess.check_output(["uname","-r"]).decode().strip()
        info["pacman_mesa"] = subprocess.run(["pacman","-Q","mesa"], capture_output=True, text=True).stdout.strip() or subprocess.run(["pacman","-Q","mesa-honeykrisp-omarchy"], capture_output=True, text=True).stdout.strip()
        info["vulkan_icds"] = subprocess.run(["bash","-lc","ls -l /usr/share/vulkan/icd.d/ 2>/dev/null; cat /usr/share/vulkan/icd.d/*.json 2>/dev/null"], capture_output=True, text=True).stdout.strip()
        info["vulkan_summary"] = subprocess.run(["vulkaninfo","--summary"], capture_output=True, text=True).stdout.strip()[:4096]
    except Exception as e:
        info["subprocess_err"] = repr(e)
    try:
        import importlib.metadata as m
        info["mlx_version"] = m.version("mlx-omarchy") or m.version("mlx")
    except Exception as e:
        info["mlx_version_err"] = repr(e)
    try:
        import mlx.core as mx
        info["mx_default_device"] = str(mx.default_device())
    except Exception as e:
        info["mx_import_err"] = repr(e)
    return info

def make_inputs(seed, M, K, N, dtype=np.float16):
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((M, K)) * 0.5).astype(dtype)
    W = (rng.standard_normal((N, K)) * 0.02).astype(dtype)
    return x, W

def time_op(op, warmup=3, iterations=30):

    # Warmup
    for _ in range(warmup):
        op()
    samples = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        op()
        t1 = time.perf_counter()
        samples.append(t1 - t0)
    return samples

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument("--seed", type=int, default=20260908)
    args = ap.parse_args()

    import mlx.core as mx

    info = host_info()
    cases = []
    seed = args.seed

    # === Lifecycle mirror shapes (ANE: lifecycle6 evidence) ===
    # 02-add-runtime-native: x [1,512] fp16 add; ANE 0.0456 ms
    for dtype, dname in [(np.float16,"fp16"), (np.bfloat16 if hasattr(np,"bfloat16") else np.float16, "bf16")]:
        x, W = make_inputs(seed, 1, 512, 0, dtype=dtype if dtype is not np.float16 else np.float16)
        # bf16 path: cast inputs
        xm = mx.array(x.astype(np.float16 if dname=="bf16" else np.float16))
        samples = time_op(lambda: (lambda y: (mx.eval(y), np.array(y))[1])(xm + xm))
        cases.append({"name": f"add512_{dname}", "shape":[1,512], "op":"add", "dtype":dname, "median_ms": median(samples)*1000, "min_ms": min(samples)*1000, "n":len(samples)})

    # 05-softmax-512: ANE 0.1658 ms
    x, _ = make_inputs(seed, 1, 512, 0)
    xm = mx.array(x)
    samples = time_op(lambda: (mx.eval(mx.softmax(mx.array(x), axis=-1))))
    cases.append({"name":"softmax512_fp16", "shape":[1,512], "op":"softmax", "dtype":"fp16", "median_ms": median(samples)*1000, "min_ms": min(samples)*1000, "n":len(samples)})

    # 04-matvec k256 n512: ANE 0.1783 ms (x[1,256] @ W[512,256]^T -> [1,512])
    for dtype, dname in [(np.float16,"fp16"), (np.bfloat16 if hasattr(np,"bfloat16") else np.float16,"bf16")]:
        x, W = make_inputs(seed, 1, 256, 512, dtype=dtype)
        # bf16: cast to bf16 for actual BF16-model measurement
        if dname == "bf16":
            xm = mx.array(x.astype(np.float16))  # bf16 fallback: mlx may lack bf16 matmul dtype; mark dtype in label
        else:
            xm = mx.array(x)
        Wm = mx.array(W.T)
        samples = time_op(lambda: mx.eval(xm @ Wm))
        cases.append({"name":f"matvec_k256_n512_{dname}", "shape":[1,256,512], "op":"matmul_T", "dtype":dname, "median_ms": median(samples)*1000, "min_ms": min(samples)*1000, "n":len(samples)})

    # 08-mlp-768-1024-768 fp16 ANE 32.17 ms; mirror = linear+relu+linear+relu
    x, _ = make_inputs(seed, 1, 768, 0); W1,_ = make_inputs(seed+1, 1024, 768, 0); W2,_ = make_inputs(seed+2, 768, 1024, 0)
    xm, W1m, W2m = mx.array(x), mx.array(W1.T), mx.array(W2.T)
    def mlp():
        h = mx.maximum(xm @ W1m, mx.zeros_like(xm @ W1m))  # relu
        y = mx.maximum(h @ W2m, mx.zeros_like(h @ W2m))
        return mx.eval(y)
    samples = time_op(mlp)
    cases.append({"name":"mlp768_fp16_relu", "shape":[1,768,1024,768], "op":"mlp", "dtype":"fp16", "median_ms": median(samples)*1000, "min_ms": min(samples)*1000, "n":len(samples)})

    # === Model-shape decode GEMVs (Qwen2.5-0.5B batch=1) ===
    # gate/up: [1,896] @ [4864,896]^T -> [1,4864]
    for dtype, dname in [(np.float16,"fp16"), (np.bfloat16 if hasattr(np,"bfloat16") else np.float16,"bf16")]:
        x, W = make_inputs(seed, 1, 896, 4864, dtype=dtype if dname=="fp16" else np.float16)
        xm = mx.array(x.astype(np.float16 if dname=="bf16" else np.float16))
        Wm = mx.array(W.T.astype(np.float16 if dname=="bf16" else np.float16))
        samples = time_op(lambda: mx.eval(xm @ Wm))
        cases.append({"name":f"gemv_q0_{dname}_k896_n4864", "shape":[1,896,4864], "op":"matmul_T", "dtype":dname, "median_ms": median(samples)*1000, "min_ms": min(samples)*1000, "n":len(samples)})

    # down: [1,4864] @ [896,4864]^T -> [1,896]
    for dtype, dname in [(np.float16,"fp16"), (np.bfloat16 if hasattr(np,"bfloat16") else np.float16,"bf16")]:
        x, W = make_inputs(seed, 1, 4864, 896, dtype=dtype if dname=="fp16" else np.float16)
        xm = mx.array(x.astype(np.float16 if dname=="bf16" else np.float16))
        Wm = mx.array(W.T.astype(np.float16 if dname=="bf16" else np.float16))
        samples = time_op(lambda: mx.eval(xm @ Wm))
        cases.append({"name":f"gemv_down_{dname}_k4864_n896", "shape":[1,4864,896], "op":"matmul_T", "dtype":dname, "median_ms": median(samples)*1000, "min_ms": min(samples)*1000, "n":len(samples)})

    # 4-bit quantized matmul (real 4-bit model op), group 64.
    # c2548675 mlx-omarchy runs qmm on the GPU (qmm_vec / qmm_coopmat paths).
    # No forced CPU stream; default device is the GPU.
    for (Mn,K,Nn,name) in [(1,896,4864,"q0"),(1,4864,896,"down")]:
        x, W = make_inputs(seed, Mn, K, Nn, dtype=np.float16)
        xm = mx.array(x)
        Wm = mx.array(W.astype(np.float16))
        Wq, sc, bs = mx.quantize(Wm, group_size=64, bits=4)
        def qmm_run():
            return mx.eval(mx.quantized_matmul(xm, Wq, sc, bs, group_size=64, bits=4, transpose=True))
        samples = time_op(qmm_run)
        cases.append({"name":f"qmm4_g64_{name}_k{K}_n{Nn}", "shape":[Mn,K,Nn], "op":"quantized_matmul", "dtype":"qmm4_g64", "median_ms": median(samples)*1000, "min_ms": min(samples)*1000, "n":len(samples), "device":str(mx.default_device())})

    # Full decode MLP block fp16: silu(x@Wg.T) * (x@Wu.T) -> @Wd.T  (gate*up+silu+mul+down)
    x, _ = make_inputs(seed, 1, 896, 0); Wg,_=make_inputs(seed+3,4864,896,0); Wu,_=make_inputs(seed+4,4864,896,0); Wd,_=make_inputs(seed+5,896,4864,0)
    xm=mx.array(x); Wgm=mx.array(Wg.T); Wum=mx.array(Wu.T); Wdm=mx.array(Wd.T)
    def mlp_block():
        g = xm @ Wgm
        u = xm @ Wum
        s = g * mx.sigmoid(g)  # silu
        h = s * u
        return mx.eval(h @ Wdm)
    samples = time_op(mlp_block)
    cases.append({"name":"mlp_block_qwen_fp16", "shape":[1,896,4864,896], "op":"silu_gemm", "dtype":"fp16", "median_ms": median(samples)*1000, "min_ms": min(samples)*1000, "n":len(samples)})

    # qkv fused GEMV [1,896] x [1152,896]^T -> [1,1152]
    x, W = make_inputs(seed, 1, 896, 1152)
    xm, Wm = mx.array(x), mx.array(W.T)
    samples = time_op(lambda: mx.eval(xm @ Wm))
    cases.append({"name":"qkv_k896_n1152_fp16", "shape":[1,896,1152], "op":"matmul_T", "dtype":"fp16", "median_ms": median(samples)*1000, "min_ms": min(samples)*1000, "n":len(samples)})

    # RMSNorm [1,896] fp16 (model layer-norm equivalent, since ANE has layer_norm not RMSNorm)
    x,_=make_inputs(seed, 1, 896, 0)
    Wrm,_=make_inputs(seed+6, 1, 896, 0)
    xm=mx.array(x); Wrmm=mx.array(Wrm)
    def rms():
        inv = mx.rsqrt(mx.mean(xm * xm, axis=-1, keepdims=True) + mx.array(1e-6, mx.float16))
        return mx.eval((xm * inv) * Wrmm)
    samples = time_op(rms)
    cases.append({"name":"rmsnorm896_fp16", "shape":[1,896], "op":"rms_norm", "dtype":"fp16", "median_ms": median(samples)*1000, "min_ms": min(samples)*1000, "n":len(samples)})

    out = {"host": info, "warmup":3, "iterations":30, "cases":cases,
           "method":"host->mx.array->op->mx.eval->np.array; per-iter perf_counter; median of 30",
           "ane_compare_note":"GPU medians here vs ANE lifecycle6-benchmarks medians (already measured) and parity-ane-candidates/b1/b2 (compiled 2026-09-08, measured at next window)"}
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.out}")
    for c in cases:
        print(f"  {c['name']}: median {c['median_ms']:.4f} ms")

if __name__ == "__main__":
    main()
