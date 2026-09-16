#!/usr/bin/env python3
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Instrument the exact harness callbacks: log every decoder/joint call."""
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

from coreml.parakeet_tdt import (  # noqa: E402
    DecoderStep,
    JointDecision,
    greedy_tdt_decode,
)
from coreml.reference import ReferenceLock  # noqa: E402
from coreml.vulkan_decoder import load_decoder  # noqa: E402
from coreml.vulkan_decoder_step import pack_step_weights, run_step  # noqa: E402


def main() -> int:
    mx.set_default_device(mx.gpu)
    lock = ReferenceLock.load(PKG / "coreml" / "parakeet-reference.lock")
    config = lock.tdt
    decoder = load_decoder(MODEL / "decoder.mlpackage")
    packed = pack_step_weights(decoder, MODEL / "joint.mlpackage")

    rng = np.random.default_rng(20260918)
    FRAMES = 24
    enc_np = (rng.standard_normal((1, FRAMES, 640)) * 4.0).astype(np.float32)
    encoder = mx.array(enc_np)
    mx.eval(encoder)

    with mx.stream(mx.gpu):
        hidden = mx.zeros((2, 1, 640), dtype=mx.float32)
        cell = mx.zeros((2, 1, 640), dtype=mx.float32)
    mx.eval(hidden, cell)

    state = {"frame": 0, "out": None}
    log = []

    def decoder_callback(token_id, current_hidden, current_cell):
        out, tok, dur, _, _ = run_step(
            packed, current_hidden, current_cell, token_id,
            encoder, state["frame"],
        )
        mx.eval(out)
        state["out"] = (tok, dur)
        ds = out[0:640].reshape(1, 640)
        log.append(
            f"decode   fh={state['frame']} token={token_id} -> "
            f"tok={tok} dur={dur} ds_id={id(ds)}"
        )
        return DecoderStep(
            ds,
            out[640:1920].reshape(2, 1, 640),
            out[1920:3200].reshape(2, 1, 640),
        )

    def joint_callback(frame_index, dec_state):
        state["frame"] = frame_index
        if state["out"] is not None and False:
            pass
        if state["out"] is not None:
            tok, dur = state["out"]
            state["out"] = None
            log.append(
                f"joint    f={frame_index} REUSE dec_state_id={id(dec_state)}"
                f" -> tok={tok} dur={dur}"
            )
            return JointDecision(tok, dur)
        out, tok, dur, _, _ = run_step(
            packed, None, None, 0, encoder, frame_index,
            skip_lstm=True, dec_in=dec_state,
        )
        mx.eval(out)
        log.append(
            f"joint    f={frame_index} mode1 dec_state_id={id(dec_state)}"
            f" -> tok={tok} dur={dur}"
        )
        return JointDecision(tok, dur)

    ref = greedy_tdt_decode(
        valid_frames=FRAMES, config=config,
        initial_hidden=hidden, initial_cell=cell,
        run_decoder=decoder_callback, run_joint=joint_callback,
    )
    print("ref tokens :", list(ref.token_ids))
    print("ref frames :", list(ref.frame_indices))
    print("ref durs   :", list(ref.durations))
    print("--- callback log ---")
    for line in log:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
