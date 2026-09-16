#!/usr/bin/env python3
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Bench the GPU loop on the real ANE-arm encoder output; check tokens."""
import json
import sys
import time
from pathlib import Path

import numpy as np

PKG = Path("/var/tmp/TdtGpuLoop/pkg")
MODEL = (
    Path.home()
    / ".cache/mlx-omarchy/parakeet-reference/mweinbach1"
    / "parakeet-tdt-0.6b-v3-coreml/b650695c2322ee5281dff48d7345b2f3a58ff018"
)
ENC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/var/tmp/TdtGpuStep2/e2e-after/encoder_hidden.npy"
)
GOLD = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
    "/var/tmp/TdtGpuStep2/e2e-after/token_ids.json"
)

sys.path.insert(0, str(PKG))
sys.path.insert(0, str(PKG / "coreml"))

import mlx.core as mx  # noqa: E402

from coreml.reference import ReferenceLock  # noqa: E402
from coreml.vulkan_decoder import load_decoder  # noqa: E402
from coreml.vulkan_decoder_step import pack_step_weights  # noqa: E402
from coreml.vulkan_tdt_loop import run_tdt_loop  # noqa: E402


def main() -> int:
    mx.set_default_device(mx.gpu)
    lock = ReferenceLock.load(PKG / "coreml" / "parakeet-reference.lock")
    decoder = load_decoder(MODEL / "decoder.mlpackage")
    packed = pack_step_weights(decoder, MODEL / "joint.mlpackage")

    enc_np = np.load(ENC)
    print("encoder:", enc_np.shape, enc_np.dtype)
    encoder = mx.array(enc_np)
    mx.eval(encoder)
    frames = int(enc_np.shape[1])

    with mx.stream(mx.gpu):
        hidden = mx.zeros((2, 1, 640), dtype=mx.float32)
        cell = mx.zeros((2, 1, 640), dtype=mx.float32)
    mx.eval(hidden, cell)

    # warmup/JIT on the first call, then timed reps
    got = run_tdt_loop(
        packed, encoder, valid_frames=frames, config=lock.tdt,
        initial_hidden=hidden, initial_cell=cell,
    )
    best = None
    for _ in range(3):
        t0 = time.monotonic_ns()
        got = run_tdt_loop(
            packed, encoder, valid_frames=frames, config=lock.tdt,
            initial_hidden=hidden, initial_cell=cell,
        )
        ms = (time.monotonic_ns() - t0) / 1e6
        best = ms if best is None else min(best, ms)
        print(f"loop run: {ms:.1f} ms")

    if GOLD.exists():
        gold = json.loads(GOLD.read_text())
        ok = (
            got.token_ids == gold["token_ids"]
            and got.frame_indices == gold["frame_indices"]
            and got.durations == gold["durations"]
        )
        print("tokens match golden:", ok)
        if not ok:
            for i, (a, b) in enumerate(
                zip(got.token_ids, gold["token_ids"])
            ):
                if a != b:
                    print(f"first divergence at emission {i}: "
                          f"{a} vs {b}")
                    break
    print(
        f"emissions {len(got.token_ids)} frames {frames} "
        f"best {best:.1f} ms (gate < 2880, target < 1000)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
