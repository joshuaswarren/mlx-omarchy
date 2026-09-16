#!/usr/bin/env python3
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Isolate the single-NaN projector lane with fully synced inputs."""
import sys

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
    from coreml.vulkan_decoder import load_decoder
    from coreml.vulkan_decoder_step import (
        _joint_kernel,
        _proj_kernel,
        pack_step_weights,
    )

    rng = np.random.default_rng(20260915)
    decoder = load_decoder(MODEL + "/decoder.mlpackage")
    packed = pack_step_weights(decoder, MODEL + "/joint.mlpackage")
    frames = rng.standard_normal((1, 24, 640)).astype(np.float32)
    with mx.stream(mx.gpu):
        encoder_dev = mx.array(frames)
    mx.eval(encoder_dev)

    # feed a KNOWN finite input: reuse h1-like signal
    h1 = mx.array(rng.standard_normal((640,)).astype(np.float32) * 0.3)
    mx.eval(h1)
    flagsp = mx.array(np.array([3, 0, 1], np.int32))
    pj, relu_out, dbg2 = _proj_kernel()(
        inputs=[h1, h1, packed.projector, encoder_dev.reshape(-1), flagsp],
        output_shapes=[(640,), (640,), (1280,)],
        output_dtypes=[mx.float32, mx.float16, mx.float32],
        grid=(1, 1, 1),
        threadgroup=(640, 1, 1),
        stream=mx.gpu,
    )
    mx.eval(pj, relu_out, dbg2)
    pjh = np.asarray(pj)
    acc = np.asarray(dbg2)[0:640]
    print("synthetic input: acc finite:", np.isfinite(acc).all(),
          "pj finite:", np.isfinite(pjh).all())
    if not np.isfinite(acc).all():
        bad = np.flatnonzero(~np.isfinite(acc))
        print("  NaN lanes:", bad[:10])

    # numpy expectation on the same input
    proj = np.asarray(packed.projector, dtype=np.float16)
    W = proj[: 640 * 640].reshape(640, 640)  # K-major rows
    bias = proj[640 * 640:]
    h1n = np.asarray(h1)
    acc_ref = np.zeros(640, dtype=np.float32)
    for k in range(640):
        acc_ref += np.float32(h1n[k]) * W[k].astype(np.float32)
    same = np.array_equal(acc, acc_ref)
    print("acc bitwise == numpy ascending:", same)
    if not same:
        d = np.flatnonzero(acc != acc_ref)
        print("  first differing lanes:", d[:10],
              "kernel:", acc[d[:3]], "numpy:", acc_ref[d[:3]])
    return 0


if __name__ == "__main__":
    sys.exit(main())
