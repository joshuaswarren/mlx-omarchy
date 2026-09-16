#!/usr/bin/env python3
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""What drives the ~6.5 ms synced layer dispatch? Null kernels with the
layer's signature/grid, and stripped layer sources, all synced one call."""
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

    decoder = load_decoder(MODEL + "/decoder.mlpackage")
    packed = vds.pack_step_weights(decoder, MODEL + "/joint.mlpackage")
    hidden = mx.zeros((1280,), dtype=mx.float32)
    mx.eval(hidden)
    G, T = vds._LAYER_TG, vds._LAYER_THREADS
    flags0 = mx.array(np.array([100, 0, 0], np.int32))

    def timed_sync(k, inputs, outs_shapes, outs_dtypes, grid, tg, reps=30):
        def call():
            r = k(inputs=inputs, output_shapes=outs_shapes,
                  output_dtypes=outs_dtypes, grid=grid, threadgroup=tg,
                  stream=mx.gpu)
            mx.eval(r[0])
        call()
        best = 1e9
        for _ in range(reps):
            t0 = time.monotonic_ns()
            call()
            best = min(best, time.monotonic_ns() - t0)
        return best / 1e6

    # A: null with the layer's 8-input/3-output signature, same grid
    null_src = """
    uint g = threadgroup_position_in_grid.x;
    uint t = thread_index_in_threadgroup.x;
    h1_out[t] = float(t);
"""
    kA = mx.fast.metal_kernel(
        name="bisect_null8",
        input_names=["embedding", "weights", "biases", "luts",
                     "hidden_in", "cell_in", "flags", "x_in"],
        output_names=["h1_out", "c1_out", "dbg"],
        header="", source=null_src, compile_options={"math_mode": "safe"})
    tA = timed_sync(kA, [packed.embedding, packed.weights, packed.biases,
                         packed.luts, hidden, hidden, flags0, hidden],
                    [(640,), (640,), (1,)],
                    [mx.float32] * 3, (G * T, 1, 1), (T, 1, 1))

    # B: production layer kernel
    tB = timed_sync(vds._layer_kernel(),
                    [packed.embedding, packed.weights, packed.biases,
                     packed.luts, hidden, hidden, flags0, hidden],
                    [(640,), (640,), (1,)],
                    [mx.float32] * 3, (G * T, 1, 1), (T, 1, 1))

    # C: layer source with the fold+cell stages cut (chains write bsum->dbg)
    src = vds._LAYER_SOURCE
    cut = src.index("    threadgroup_barrier(mem_flags::mem_threadgroup);\n    if (t <")
    chains_only = (src[:cut] + """
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < 1u) {
        dbg[0] = float(bsum[0]);
    }
""")
    kC = mx.fast.metal_kernel(
        name="bisect_chainsonly",
        input_names=["embedding", "weights", "biases", "luts",
                     "hidden_in", "cell_in", "flags", "x_in"],
        output_names=["h1_out", "c1_out", "dbg"],
        header=vds._EXACT_FMA16, source=chains_only,
        compile_options={"math_mode": "safe"})
    tC = timed_sync(kC, [packed.embedding, packed.weights, packed.biases,
                         packed.luts, hidden, hidden, flags0, hidden],
                    [(640,), (640,), (1,)],
                    [mx.float32] * 3, (G * T, 1, 1), (T, 1, 1))

    # D: chains with exact_fma16 replaced by hw fma (source complexity down)
    chains_hw = chains_only.replace("bc = exact_fma16(bc, sh_a[k], weights[wbase + k * 2560u + n]);",
                                    "bc = fma(bc, sh_a[k], weights[wbase + k * 2560u + n]);")
    kD = mx.fast.metal_kernel(
        name="bisect_chainshw",
        input_names=["embedding", "weights", "biases", "luts",
                     "hidden_in", "cell_in", "flags", "x_in"],
        output_names=["h1_out", "c1_out", "dbg"],
        header="", source=chains_hw, compile_options={"math_mode": "safe"})
    tD = timed_sync(kD, [packed.embedding, packed.weights, packed.biases,
                         packed.luts, hidden, hidden, flags0, hidden],
                    [(640,), (640,), (1,)],
                    [mx.float32] * 3, (G * T, 1, 1), (T, 1, 1))

    print(f"A null-8in same-grid : {tA:.3f} ms")
    print(f"B production layer   : {tB:.3f} ms")
    print(f"C chains only        : {tC:.3f} ms")
    print(f"D chains hw-fma      : {tD:.3f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
