#!/usr/bin/env python3
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Localize the NaN: run one step and report finiteness per stage."""
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

    state_out, tok, dur, logits, dbg = run_step(
        packed, hidden, cell, 1234, encoder_dev, 3, mode=2
    )
    so = np.asarray(state_out)
    dbga = np.asarray(dbg)
    print("state_out finite:", np.isfinite(so).all(),
          "nan count:", int(np.isnan(so).sum()))
    for name, sl in (
        ("pj(dec_hidden)", slice(0, 640)),
        ("h1_l0", slice(640, 1280)),
        ("h1_l1", slice(1280, 1920)),
        ("c1_l0", slice(1920, 2560)),
        ("c1_l1", slice(2560, 3200)),
    ):
        seg = so[sl]
        print(f"  {name}: finite={np.isfinite(seg).all()} nan={int(np.isnan(seg).sum())} "
              f"max_abs={np.nanmax(np.abs(seg)) if np.isfinite(seg).any() else 'na'}")
    print("dbg shape:", dbga.shape)
    print("  dbg h1_l0 finite:", np.isfinite(dbga[0:640]).all())
    print("  dbg h1_l1 finite:", np.isfinite(dbga[640:1280]).all())
    print("  dbg acc finite:", np.isfinite(dbga[1280:1920]).all())
    print("  dbg pj finite:", np.isfinite(dbga[1920:2560]).all())
    lg = np.asarray(logits)
    print("logits finite:", np.isfinite(lg).all(), "tok:", tok, "dur:", dur)
    return 0


if __name__ == "__main__":
    sys.exit(main())
