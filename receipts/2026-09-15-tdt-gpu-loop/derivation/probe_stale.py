#!/usr/bin/env python3
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Test which stale input explains the joint@14 blank after decode(439)."""
import sys
from pathlib import Path

import numpy as np

PKG = Path("/var/tmp/TdtGpuLoop/pkg")
MODEL = (
    Path.home()
    / ".cache/mlx-omarchy/parakeet-reference/mweinbach1"
    / "parakeet-tdt-0.6b-v3-coreml/b650695c2322ee5281dff48d7345b2f3a58ff018"
)
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(PKG / "coreml"))

import mlx.core as mx  # noqa: E402

from coreml.reference import ReferenceLock  # noqa: E402
from coreml.vulkan_decoder import load_decoder  # noqa: E402
from coreml.vulkan_decoder_step import pack_step_weights, run_step  # noqa: E402


def main() -> int:
    mx.set_default_device(mx.gpu)
    lock = ReferenceLock.load(PKG / "coreml" / "parakeet-reference.lock")
    decoder = load_decoder(MODEL / "decoder.mlpackage")
    packed = pack_step_weights(decoder, MODEL / "joint.mlpackage")

    rng = np.random.default_rng(20260918)
    FRAMES = 24
    enc_np = (rng.standard_normal((1, FRAMES, 640)) * 4.0).astype(np.float32)
    encoder = mx.array(enc_np)
    mx.eval(encoder)

    out0, _, _, _, _ = run_step(
        packed, mx.zeros((2, 1, 640), dtype=mx.float32),
        mx.zeros((2, 1, 640), dtype=mx.float32),
        int(lock.tdt.blank_token_id), encoder, 0,
    )
    mx.eval(out0)
    pj0 = out0[0:640].reshape(1, 640)
    out1, _, _, _, _ = run_step(
        packed, out0[640:1920].reshape(2, 1, 640),
        out0[1920:3200].reshape(2, 1, 640), 439, encoder, 14,
    )
    mx.eval(out1)
    pj1 = out1[0:640].reshape(1, 640)

    for label, pj, frame in (
        ("pj0 + enc[12]  (iter2: the 439 emission)", pj0, 12),
        ("pj0 + enc[14]  (stale pj at iter3)", pj0, 14),
        ("pj1 + enc[12]  (stale frame at iter3)", pj1, 12),
        ("pj1 + enc[14]  (correct iter3)", pj1, 14),
        ("pj1 + enc[15]  ", pj1, 15),
    ):
        _, tok, dur, _, _ = run_step(
            packed, None, None, 0, encoder, frame,
            skip_lstm=True, dec_in=pj,
        )
        mx.eval()
        print(f"{label}: token {tok} duration_index {dur}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
