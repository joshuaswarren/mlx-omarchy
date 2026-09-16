#!/usr/bin/env python3
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Per-dispatch wall share of run_step (uses the module's sync hook)."""
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

    rng = np.random.default_rng(20260915)
    decoder = load_decoder(MODEL + "/decoder.mlpackage")
    packed = vds.pack_step_weights(decoder, MODEL + "/joint.mlpackage")
    frames = rng.standard_normal((1, 24, 640)).astype(np.float32)
    with mx.stream(mx.gpu):
        encoder_dev = mx.array(frames)
    mx.eval(encoder_dev)
    hidden = mx.zeros((2, 1, 640), dtype=mx.float32)
    cell = mx.zeros((2, 1, 640), dtype=mx.float32)
    mx.eval(hidden, cell)

    layer = vds._layer_kernel()
    proj = vds._proj_kernel()
    joint = vds._joint_kernel()
    G, T = vds._LAYER_TG, vds._LAYER_THREADS
    flat_h = hidden.reshape(1280,)

    def timeit(fn, reps=50):
        fn()
        best = 1e9
        for _ in range(reps):
            t0 = time.monotonic_ns()
            fn()
            best = min(best, time.monotonic_ns() - t0)
        return best / 1e6

    f0 = mx.array(np.array([100, 0, 0], np.int32))
    h1_0, c1_0, _ = layer(
        inputs=[packed.embedding, packed.weights, packed.biases, packed.luts,
                flat_h, flat_h, f0, flat_h],
        output_shapes=[(640,), (640,), (1,)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
        grid=(G * T, 1, 1), threadgroup=(T, 1, 1), stream=mx.gpu)
    f1 = mx.array(np.array([100, 1, 0], np.int32))
    t_layer0 = timeit(lambda: mx.eval(layer(
        inputs=[packed.embedding, packed.weights, packed.biases, packed.luts,
                flat_h, flat_h, f0, flat_h],
        output_shapes=[(640,), (640,), (1,)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
        grid=(G * T, 1, 1), threadgroup=(T, 1, 1), stream=mx.gpu)))
    t_layer1 = timeit(lambda: mx.eval(layer(
        inputs=[packed.embedding, packed.weights, packed.biases, packed.luts,
                flat_h, flat_h, f1, h1_0],
        output_shapes=[(640,), (640,), (1,)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
        grid=(G * T, 1, 1), threadgroup=(T, 1, 1), stream=mx.gpu)))
    fp = mx.array(np.array([3, 0, 0], np.int32))
    t_proj = timeit(lambda: mx.eval(proj(
        inputs=[h1_0, flat_h, packed.projector, encoder_dev.reshape(-1), fp],
        output_shapes=[(640,), (640,), (1,)],
        output_dtypes=[mx.float32, mx.float16, mx.float32],
        grid=(640, 1, 1), threadgroup=(640, 1, 1), stream=mx.gpu)[0]))
    relu = mx.array(np.zeros(640, dtype=np.float16))
    mx.eval(relu)
    t_joint = timeit(lambda: mx.eval(joint(
        inputs=[relu, packed.joint],
        output_shapes=[(8198,)],
        output_dtypes=[mx.float32],
        grid=(33 * 256, 1, 1), threadgroup=(256, 1, 1), stream=mx.gpu)[0]))
    print(f"layer0 {t_layer0:.3f} ms  layer1 {t_layer1:.3f} ms  "
          f"proj {t_proj:.3f} ms  joint {t_joint:.3f} ms")
    print(f"sum {t_layer0 + t_layer1 + t_proj + t_joint:.3f} ms "
          f"(bench full step was 17.0)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
