#!/usr/bin/env python3
"""Kernel-isolated Q4 qmm prefill probe for the scheduling screens.

One process per screen arm (the caller exports MLX_OMARCHY_QMM_SMALLN_TILE
or MLX_OMARCHY_QMM_SPLITK before launch). Per shape it reports the median
eval wall time, the Omarchy dispatch counter delta (1 = single kernel,
2 = splitk + reduce), the f64 host-oracle error against the kernel bound,
and the sha256 of the f16 output bytes for cross-arm comparison.
"""
import argparse
import ctypes
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np

import mlx.core as mx


def dispatch_count(libmlx):
    class Snap(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in (
            "gpu_primitive_dispatches", "vk_submissions",
            "vk_buffer_copies", "vk_buffer_fills",
            "vk_compute_dispatches", "omarchy_finalize_calls",
            "commit_calls_with_work", "commit_calls_noop")]

    trace = ctypes.CDLL(str(libmlx))
    snap = Snap()
    trace.mlx_omarchy_trace_snapshot(ctypes.byref(snap))
    return snap.vk_compute_dispatches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--shapes", default="1053:896:128,1053:896:896,1053:896:9728,1053:4864:896,262:896:128")
    args = ap.parse_args()

    info = mx.device_info()
    prov = {
        "mlx_version": mx.__version__,
        "device_name": info.get("device_name"),
        "cooperative_matrix_f32_8": info.get("cooperative_matrix_f32_8"),
        "MLX_OMARCHY_QMM_SMALLN_TILE": os.environ.get("MLX_OMARCHY_QMM_SMALLN_TILE"),
        "MLX_OMARCHY_QMM_SPLITK": os.environ.get("MLX_OMARCHY_QMM_SPLITK"),
        "VK_DRIVER_FILES": os.environ.get("VK_DRIVER_FILES"),
        "iters": args.iters,
    }
    libmlx = Path(mx.__file__).parent / "lib" / "libmlx.so"

    rng = np.random.default_rng(20260910)
    results = []
    for spec in args.shapes.split(","):
        m, k, n = (int(v) for v in spec.split(":"))
        w_f32 = rng.standard_normal((n, k), dtype=np.float32) * 0.05
        x_f16 = (rng.standard_normal((m, k), dtype=np.float32) * 0.5).astype(np.float16)
        w_q, scales, biases = mx.quantize(mx.array(w_f32), group_size=64, bits=4)
        x = mx.array(x_f16)

        # Exact f64 host oracle from the packed words: nibble j of word i
        # is weight 8*i+j (low nibble first, qmm.comp lane split). Each
        # uint32 word is four little-endian bytes covering eight weights.
        word_bytes = np.asarray(w_q).reshape(n, k // 8).view(np.uint8)
        nibbles = np.empty((n, k), dtype=np.float64)
        nibbles[:, 0::2] = word_bytes & 0x0F
        nibbles[:, 1::2] = word_bytes >> 4
        groups = nibbles.reshape(n, k // 64, 64)
        np_scales = np.asarray(scales.astype(mx.float32)).astype(np.float64)
        np_biases = np.asarray(biases.astype(mx.float32)).astype(np.float64)
        deq = (groups * np_scales[:, :, None] + np_biases[:, :, None]).reshape(n, k)
        expected = x_f16.astype(np.float64) @ deq.T

        def run_once():
            out = mx.quantized_matmul(x, w_q, scales, biases, transpose=True)
            mx.eval(out)
            return out

        before = dispatch_count(libmlx)
        out = run_once()
        dispatch_delta = dispatch_count(libmlx) - before
        for _ in range(3):
            run_once()
        times = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            run_once()
            times.append((time.perf_counter() - t0) * 1000.0)
        out_np = np.asarray(out.astype(mx.float32))
        max_abs = max(1.0, float(np.abs(out_np).max()), float(np.abs(expected).max()))
        bound = (3.0 * k + 1.0) * (2.0 * max_abs) * float(np.ldexp(1.0, -23)) \
            + (2.0 * max_abs) * float(np.ldexp(1.0, -11))
        max_diff = float(np.abs(out_np - expected).max())
        row = {
            "m": m, "k": k, "n": n,
            "tiles": ((m + 31) // 32) * ((n + 31) // 32),
            "dispatch_delta": int(dispatch_delta),
            "median_ms": round(statistics.median(times), 4),
            "max_abs_err": max_diff,
            "bound": bound,
            "within_bound": bool(max_diff <= bound),
            "f16_digest": hashlib.sha256(np.asarray(out).tobytes()).hexdigest()[:16],
        }
        results.append(row)
        print(json.dumps(row), flush=True)

    args.out.write_text(json.dumps({"provenance": prov, "results": results},
                                   indent=2) + "\n")
    print("PROBE_OK", flush=True)


if __name__ == "__main__":
    main()
