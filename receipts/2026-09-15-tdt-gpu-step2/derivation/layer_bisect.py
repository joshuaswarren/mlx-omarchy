#!/usr/bin/env python3
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Layer-kernel cost bisection: which part of the 6.5 ms is the chain?

Variants patch the production layer source:
  full      - production kernel (baseline)
  no_shread - chain reads a register constant instead of sh_a[k]
  no_wload  - chain reads a register constant instead of weights[...]
  chains_only - fold/cell stages removed (bsum written straight to dbg)
  fastmath  - production source compiled with math_mode fast
"""
import sys
import time

import numpy as np

PKG = "/var/tmp/TdtGpuStep2/pkg"
MODEL = (
    "/home/joshuawarren/.cache/mlx-omarchy/parakeet-reference/mweinbach1/"
    "parakeet-tdt-0.6b-v3-coreml/b650695c2322ee5281dff48d7345b2f3a58ff018"
)


def main() -> int:
    import mlx.core as mx

    sys.path.insert(0, PKG)
    sys.path.insert(0, PKG + "/coreml")
    import coreml.vulkan_decoder_step as vds
    from coreml.vulkan_decoder import load_decoder

    src = vds._LAYER_SOURCE
    hdr = vds._EXACT_FMA16
    decoder = load_decoder(MODEL + "/decoder.mlpackage")
    packed = vds.pack_step_weights(decoder, MODEL + "/joint.mlpackage")
    frames = np.random.default_rng(3).standard_normal((1, 24, 640)).astype(np.float32)
    encoder_dev = mx.array(frames)
    mx.eval(encoder_dev)
    hidden = mx.zeros((1280,), dtype=mx.float32)
    mx.eval(hidden)

    G, T, L = vds._LAYER_TG, vds._LAYER_THREADS, vds._LAYER_LANES
    flags0 = mx.array(np.array([100, 0, 0], np.int32))

    def build(source, math_mode="safe", name="bisect"):
        return mx.fast.metal_kernel(
            name=f"parakeet_bisect_{name}",
            input_names=["embedding", "weights", "biases", "luts",
                         "hidden_in", "cell_in", "flags", "x_in"],
            output_names=["h1_out", "c1_out", "dbg"],
            header=hdr,
            source=source,
            compile_options={"math_mode": math_mode},
        )

    def timeit(k, reps=40):
        def call():
            r = k(
                inputs=[packed.embedding, packed.weights, packed.biases,
                        packed.luts, hidden, hidden, flags0, hidden],
                output_shapes=[(640,), (640,), (1,)],
                output_dtypes=[mx.float32, mx.float32, mx.float32],
                grid=(G * T, 1, 1), threadgroup=(T, 1, 1), stream=mx.gpu)
            mx.eval(r[0])
        call()
        best = 1e9
        for _ in range(reps):
            t0 = time.monotonic_ns()
            call()
            best = min(best, time.monotonic_ns() - t0)
        return best / 1e6

    no_shread = src.replace(
        "bc = exact_fma16(bc, sh_a[k], weights[wbase + k * 2560u + n]);",
        "bc = exact_fma16(bc, float16_t(0.001f), weights[wbase + k * 2560u + n]);")
    no_wload = src.replace(
        "bc = exact_fma16(bc, sh_a[k], weights[wbase + k * 2560u + n]);",
        "bc = exact_fma16(bc, sh_a[k], float16_t(0.001f));")
    fold_marker = "    threadgroup_barrier(mem_flags::mem_threadgroup);\n    if (t <"
    chains_only = src[: src.index(fold_marker)] + """
    if (idx < 640u) {
        dbg[idx] = float(bsum[idx]);
    }
"""

    variants = [
        ("full", src, "safe"),
        ("no_shread", no_shread, "safe"),
        ("no_wload", no_wload, "safe"),
        ("chains_only", chains_only, "safe"),
        ("fastmath", src, "fast"),
    ]
    for name, s2, mm in variants:
        try:
            k = build(s2, mm, name)
            print(f"{name}: {timeit(k):.3f} ms", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"{name}: ERROR {str(exc)[:140]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
