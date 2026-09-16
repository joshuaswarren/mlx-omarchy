#!/usr/bin/env python3
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Wall time of the new step vs the numpy contract path, decode-loop shaped."""
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
    from coreml.vulkan_decoder import load_decoder
    from coreml.vulkan_decoder_step import pack_step_weights, run_step

    rng = np.random.default_rng(20260915)
    decoder = load_decoder(MODEL + "/decoder.mlpackage")
    packed = pack_step_weights(decoder, MODEL + "/joint.mlpackage")
    frames = rng.standard_normal((1, 24, 640)).astype(np.float32)
    with mx.stream(mx.gpu):
        encoder_dev = mx.array(frames)
    mx.eval(encoder_dev)
    hidden = mx.zeros((2, 1, 640), dtype=mx.float32)
    cell = mx.zeros((2, 1, 640), dtype=mx.float32)
    mx.eval(hidden, cell)

    # decode-loop shape: 145 sequential steps, recurrent state carried,
    # frame sweeping 0..23
    for _ in range(5):
        run_step(packed, hidden, cell, 100, encoder_dev, 0)
    reps = 145
    tok_sum = 0
    t0 = time.monotonic_ns()
    for i in range(reps):
        state_out, tok, dur, logits, _ = run_step(
            packed, hidden, cell, tok_sum % 8000, encoder_dev, i % 24
        )
        hidden = state_out[640:1920].reshape(2, 1, 640)
        cell = state_out[1920:3200].reshape(2, 1, 640)
        tok_sum += tok
    wall = (time.monotonic_ns() - t0) / 1e6 / reps
    print(f"step mean ms: {wall:.3f}  (reps={reps})")
    print(f"projected tdt_decode ms: {wall * 146:.1f}  (numpy baseline 2880)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
