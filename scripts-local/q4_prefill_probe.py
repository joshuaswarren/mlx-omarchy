#!/usr/bin/env python3
"""Q4 1K prefill attribution + coopmat schedule probe (no GPU instrumentation).

Method matches receipts/2026-09-10-qmm-splitk-parity/probe-f16-scales.json:
isolated primitive wall time, median of 30 evals after 3 warmups, on a quiet
machine under /tmp/m1-gpu.lock. Every arm prints the f16 output sha256 prefix
so digest preservation is checked against the committed baseline values.

Sections:
  qmm     - kernel-isolated quantized_matmul at the four real Qwen2.5-0.5B
            1053-token shapes for each MLX_OMARCHY_QMM_COOP_TILE arm. The env
            var is read once by this script per process, so run one process
            per arm.
  attn    - isolated mx.fast.scaled_dot_product_attention at the real
            prefill shapes (B=1, 14 q heads, 2 kv heads, head_dim 64,
            L in {30, 262, 1053}), f16, causal - the composed graph the
            backend actually dispatches.
  norms   - isolated rms_norm at [1053, 896] f16.
  lmhead  - quantized lm_head at m=1053 (full logits) to establish whether
            the model pays it during prefill.
"""
import hashlib
import json
import statistics
import sys
import time

import mlx.core as mx
from mlx_lm.models import qwen2  # noqa: F401  (import guard for wheel sanity)


def timed(fn, warmup=3, reps=30):
    mx.eval(fn())
    samples = []
    for _ in range(warmup):
        mx.eval(fn())
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(fn())
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples)


def f16_digest(a):
    import numpy as np
    b = np.ascontiguousarray(np.asarray(a, dtype=np.float16)).tobytes()
    return hashlib.sha256(b).hexdigest()[:16]


def quantize_f16(n, k, group=64, bits=4, seed=7):
    w = mx.random.normal((n, k), key=mx.random.key(seed)) * 0.5
    w = mx.eval(w)
    w, scales, biases = mx.quantize(w.astype(mx.float32), group, bits)
    scales = scales.astype(mx.float16)
    biases = biases.astype(mx.float16)
    mx.eval(scales, biases)
    return w, scales, biases


def qmm_section(out):
    shapes = [(896, 128), (896, 896), (4864, 896), (896, 9728)]
    results = []
    for k, n in shapes:
        w_words, scales, biases = quantize_f16(n, k, seed=7)
        for m in (1053, 30):
            if m == 30 and n != 9728:
                continue
            x = mx.random.normal((m, k), key=mx.random.key(11)).astype(
                mx.float16)
            x = mx.eval(x)

            def call():
                return mx.quantized_matmul(
                    x, w_words, scales, biases, True, 64, 4)
            ms = timed(call)
            dig = f16_digest(call())
            results.append({
                "shape": f"{m}x{k}x{n}", "median_ms": round(ms, 4),
                "f16_digest": dig,
                "gflops": round(2.0 * m * n * k / (ms * 1e6), 1),
            })
            print(json.dumps(results[-1]))
    out.setdefault("qmm", results)


def attn_section(out):
    results = []
    scale = 64 ** -0.5
    for L in (30, 262, 1053):
        q = mx.random.normal((1, 14, L, 64), key=mx.random.key(21)).astype(
            mx.float16)
        k_kv = mx.random.normal((1, 2, L, 64), key=mx.random.key(22)).astype(
            mx.float16)
        v = mx.random.normal((1, 2, L, 64), key=mx.random.key(23)).astype(
            mx.float16)
        q, k_kv, v = map(mx.eval, (q, k_kv, v))

        def call():
            return mx.fast.scaled_dot_product_attention(
                q, k_kv, v, scale=scale, mask="causal")
        ms = timed(call)
        results.append({
            "shape": f"L={L}", "median_ms": round(ms, 4),
            "gflops": round(2.0 * 2 * 14 * L * L * 64 / (ms * 1e6), 1),
            "f16_digest": f16_digest(call()),
        })
        print(json.dumps(results[-1]))
    out.setdefault("attn", results)


def norms_section(out):
    x = mx.random.normal((1053, 896), key=mx.random.key(31)).astype(
        mx.float16)
    x = mx.eval(x)

    def call():
        return mx.fast.rms_norm(x, 1e-5)
    ms = timed(call)
    out.setdefault("norms", []).append({
        "shape": "1053x896", "median_ms": round(ms, 4)})
    print(json.dumps(out["norms"][-1]))


def lmhead_section(out):
    k, n = 896, 151936
    w_words, scales, biases = quantize_f16(n, k, seed=41)
    for m in (1053, 1):
        x = mx.random.normal((m, k), key=mx.random.key(43)).astype(
            mx.float16)
        x = mx.eval(x)

        def call():
            return mx.quantized_matmul(
                x, w_words, scales, biases, True, 64, 4)
        ms = timed(call, warmup=3, reps=8)
        out.setdefault("lmhead", []).append({
            "shape": f"{m}x{k}x{n}", "median_ms": round(ms, 4),
            "gflops": round(2.0 * m * n * k / (ms * 1e6), 1)})
        print(json.dumps(out["lmhead"][-1]))


def main():
    import os
    sections = sys.argv[1:] or ["qmm", "attn", "norms", "lmhead"]
    out = {"arm": os.environ.get("MLX_OMARCHY_QMM_COOP_TILE", "0")}
    print("arm:", out["arm"])
    for section in sections:
        {"qmm": qmm_section, "attn": attn_section, "norms": norms_section,
         "lmhead": lmhead_section}[section](out)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
