#!/usr/bin/env python3
"""Q4 1K prefill attribution + coopmat schedule probe (no GPU instrumentation).

Method matches receipts/2026-09-10-qmm-splitk-parity/probe-f16-scales.json:
isolated primitive wall time, median of 30 evals after 4 warmups, on a quiet
machine under /tmp/m1-gpu.lock. Every arm prints the f16 output sha256 prefix
so digest preservation is checked against the committed baseline values.
"""
import hashlib
import json
import os
import statistics
import sys
import time

import numpy as np
import mlx.core as mx


def timed(fn, warmup=4, reps=30):
    for _ in range(warmup + 1):
        mx.eval(fn())
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(fn())
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples)


def f16_digest(a):
    b = np.ascontiguousarray(np.asarray(a, dtype=np.float16)).tobytes()
    return hashlib.sha256(b).hexdigest()[:16]


def quantize_f16(n, k, group=64, bits=4, seed=7):
    w = mx.random.normal((n, k), key=mx.random.key(seed)) * 0.5
    w, scales, biases = mx.quantize(w.astype(mx.float32), group, bits)
    scales = scales.astype(mx.float16)
    biases = biases.astype(mx.float16)
    mx.eval(w, scales, biases)
    return w, scales, biases


def f16_seed(shape, seed):
    x = mx.random.normal(shape, key=mx.random.key(seed)).astype(mx.float16)
    mx.eval(x)
    return x


def qmm_section(out):
    # (k, n) per layer: k_proj/v_proj, q_proj, down_proj, fused gate+up.
    shapes = [(896, 128), (896, 896), (4864, 896), (896, 9728)]
    results = []
    for k, n in shapes:
        w_words, scales, biases = quantize_f16(n, k, seed=7)
        for m in (1053, 30):
            if m == 30 and n != 9728:
                continue
            x = f16_seed((m, k), 11)

            def call():
                return mx.quantized_matmul(
                    x, w_words, scales, biases, True, 64, 4)
            ms = timed(call)
            row = {
                "shape": f"{m}x{k}x{n}",
                "median_ms": round(ms, 4),
                "f16_digest": f16_digest(call()),
                "gflops": round(2.0 * m * n * k / (ms * 1e6), 1),
            }
            results.append(row)
            print(json.dumps(row))
    out["qmm"] = results


def attn_section(out):
    scale = 64 ** -0.5
    results = []
    for L in (30, 262, 1053):
        q = f16_seed((1, 14, L, 64), 21)
        k_kv = f16_seed((1, 2, L, 64), 22)
        v = f16_seed((1, 2, L, 64), 23)

        def call():
            return mx.fast.scaled_dot_product_attention(
                q, k_kv, v, scale=scale, mask="causal")
        ms = timed(call)
        row = {
            "shape": f"L={L}",
            "median_ms": round(ms, 4),
            "gflops": round(2.0 * 2 * 14 * L * L * 64 / (ms * 1e6), 1),
            "f16_digest": f16_digest(call()),
        }
        results.append(row)
        print(json.dumps(row))
    out["attn"] = results


def norms_section(out):
    x = f16_seed((1053, 896), 31)

    def call():
        return mx.fast.rms_norm(x, None, 1e-5)
    ms = timed(call)
    row = {"shape": "1053x896", "median_ms": round(ms, 4)}
    out["norms"] = [row]
    print(json.dumps(row))


def lmhead_section(out):
    k, n = 896, 151936
    w_words, scales, biases = quantize_f16(n, k, seed=41)
    results = []
    for m in (1053, 1):
        x = f16_seed((m, k), 43)

        def call():
            return mx.quantized_matmul(
                x, w_words, scales, biases, True, 64, 4)
        ms = timed(call, warmup=3, reps=8)
        row = {
            "shape": f"{m}x{k}x{n}",
            "median_ms": round(ms, 4),
            "gflops": round(2.0 * m * n * k / (ms * 1e6), 1),
        }
        results.append(row)
        print(json.dumps(row))
    out["lmhead"] = results


def main():
    sections = sys.argv[1:] or ["qmm", "attn", "norms", "lmhead"]
    out = {"arm": os.environ.get("MLX_OMARCHY_QMM_COOP_TILE", "0")}
    print("arm:", out["arm"])
    table = {"qmm": qmm_section, "attn": attn_section,
             "norms": norms_section, "lmhead": lmhead_section}
    for section in sections:
        table[section](out)
    print("SUMMARY " + json.dumps(out))


if __name__ == "__main__":
    main()
