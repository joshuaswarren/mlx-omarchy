#!/usr/bin/env python3
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Host submit cost vs GPU cost per dispatch: enqueue N layer/joint calls
with no intermediate sync, then one final eval."""
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
    frames = np.random.default_rng(3).standard_normal((1, 24, 640)).astype(np.float32)
    encoder_dev = mx.array(frames)
    mx.eval(encoder_dev)
    hidden = mx.zeros((1280,), dtype=mx.float32)
    mx.eval(hidden)
    G, T = vds._LAYER_TG, vds._LAYER_THREADS
    layer = vds._layer_kernel()
    joint = vds._joint_kernel()
    N = 50

    def enqueue_layers():
        outs = []
        for i in range(N):
            flags0 = mx.array(np.array([100 + i, 0, 0], np.int32))
            r = layer(
                inputs=[packed.embedding, packed.weights, packed.biases,
                        packed.luts, hidden, hidden, flags0, hidden],
                output_shapes=[(640,), (640,), (1,)],
                output_dtypes=[mx.float32, mx.float32, mx.float32],
                grid=(G * T, 1, 1), threadgroup=(T, 1, 1), stream=mx.gpu)
            outs.append(r[0])
        return outs

    # warm
    mx.eval(enqueue_layers()[-1])
    t0 = time.monotonic_ns()
    outs = enqueue_layers()
    t_submit = (time.monotonic_ns() - t0) / 1e6
    t0 = time.monotonic_ns()
    mx.eval(outs[-1])
    t_gpu_tail = (time.monotonic_ns() - t0) / 1e6
    print(f"layer: submit {t_submit:.3f} ms total for {N} "
          f"({t_submit / N:.3f} ms/call); final-sync tail {t_gpu_tail:.3f} ms")

    flagsj = mx.array(np.array([3, 0], np.int32))

    def enqueue_joints():
        outs = []
        for i in range(N):
            r = joint(
                inputs=[hidden, hidden, packed.projector,
                        encoder_dev.reshape(-1), flagsj, packed.joint],
                output_shapes=[(8198,), (640,)],
                output_dtypes=[mx.float32, mx.float32],
                grid=(33 * 256, 1, 1), threadgroup=(256, 1, 1), stream=mx.gpu)
            outs.append(r[0])
        return outs

    mx.eval(enqueue_joints()[-1])
    t0 = time.monotonic_ns()
    outs = enqueue_joints()
    t_submit = (time.monotonic_ns() - t0) / 1e6
    t0 = time.monotonic_ns()
    mx.eval(outs[-1])
    t_gpu_tail = (time.monotonic_ns() - t0) / 1e6
    print(f"joint: submit {t_submit:.3f} ms total for {N} "
          f"({t_submit / N:.3f} ms/call); final-sync tail {t_gpu_tail:.3f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
