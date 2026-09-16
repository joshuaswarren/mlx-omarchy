#!/usr/bin/env python3
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Validate the GPU-resident TDT loop against the landed step2 host path.

Reference: ``greedy_tdt_decode`` driven by the step2 ``run_step`` callbacks
(24/24 bit-exact vs the numpy BNNS contract, receipts/2026-09-15-tdt-gpu-step2).
Candidate: ``run_tdt_loop`` (one dispatch, control on GPU).  The gate is an
identical emission stream: token ids, frame indices and durations.

Also checks the windowed mode (``frame_stop``) concatenates to the same
stream, since the same kernel must support frame-window readbacks.
"""
import sys
import time
from pathlib import Path

import numpy as np

PKG = Path(sys.argv[1]) if len(sys.argv) > 1 else None
MODEL = Path(
    sys.argv[2]
    if len(sys.argv) > 2
    else (Path.home() / ".cache/mlx-omarchy/parakeet-reference/mweinbach1"
          / "parakeet-tdt-0.6b-v3-coreml")
)
FRAMES = int(sys.argv[3]) if len(sys.argv) > 3 else 24
SEEDS = int(sys.argv[4]) if len(sys.argv) > 4 else 5

sys.path.insert(0, str(PKG))
sys.path.insert(0, str(PKG / "coreml"))

import mlx.core as mx  # noqa: E402

from coreml.parakeet_tdt import (  # noqa: E402
    DecoderStep,
    JointDecision,
    greedy_tdt_decode,
)
from coreml.reference import ReferenceLock, TdtConfig  # noqa: E402
from coreml.vulkan_decoder import load_decoder  # noqa: E402
from coreml.vulkan_decoder_step import pack_step_weights, run_step  # noqa: E402
from coreml.vulkan_tdt_loop import run_tdt_loop  # noqa: E402


def main() -> int:
    mx.set_default_device(mx.gpu)
    lock_path = PKG / "coreml" / "parakeet-reference.lock"
    lock = ReferenceLock.load(lock_path)
    config = lock.tdt
    decoder = load_decoder(MODEL / "decoder.mlpackage")
    packed = pack_step_weights(decoder, MODEL / "joint.mlpackage")

    failures = 0
    for seed in range(20260915, 20260915 + SEEDS):
        rng = np.random.default_rng(seed)
        scale = [0.5, 1.0, 2.0, 4.0, 8.0][seed % 5]
        encoder = mx.array(
            (rng.standard_normal((1, FRAMES, 640)) * scale).astype(np.float32)
        )
        mx.eval(encoder)

        with mx.stream(mx.gpu):
            hidden = mx.zeros((2, 1, 640), dtype=mx.float32)
            cell = mx.zeros((2, 1, 640), dtype=mx.float32)
        mx.eval(hidden, cell)
        fused = {"frame": None, "state": None, "tok": None, "dur": None}
        frame_holder = [0]

        def decoder_callback(token_id, current_hidden, current_cell):
            out, tok, dur, _, _ = run_step(
                packed, current_hidden, current_cell, token_id,
                encoder, frame_holder[0],
            )
            mx.eval(out)
            dec_state = out[0:640].reshape(1, 640)
            fused["frame"] = frame_holder[0]
            fused["state"] = dec_state
            fused["tok"] = tok
            fused["dur"] = dur
            return DecoderStep(
                dec_state,
                out[640:1920].reshape(2, 1, 640),
                out[1920:3200].reshape(2, 1, 640),
            )

        def joint_callback(frame_index, dec_state):
            frame_holder[0] = frame_index
            if fused["frame"] == frame_index and fused["state"] is dec_state:
                return JointDecision(fused["tok"], fused["dur"])
            out, tok, dur, _, _ = run_step(
                packed, None, None, 0, encoder, frame_index,
                skip_lstm=True, dec_in=dec_state,
            )
            mx.eval(out)
            return JointDecision(tok, dur)

        t0 = time.monotonic_ns()
        ref = greedy_tdt_decode(
            valid_frames=FRAMES,
            config=config,
            initial_hidden=hidden,
            initial_cell=cell,
            run_decoder=decoder_callback,
            run_joint=joint_callback,
        )
        ref_ms = (time.monotonic_ns() - t0) / 1e6

        t0 = time.monotonic_ns()
        got = run_tdt_loop(
            packed, encoder, valid_frames=FRAMES, config=config,
            initial_hidden=hidden, initial_cell=cell,
        )
        loop_ms = (time.monotonic_ns() - t0) / 1e6

        same = (
            got.token_ids == list(ref.token_ids)
            and got.frame_indices == list(ref.frame_indices)
            and got.durations == list(ref.durations)
        )
        # windowed split: two kernels, stop at FRAMES//2 then run to end
        mid = FRAMES // 2
        w1 = run_tdt_loop(
            packed, encoder, valid_frames=FRAMES, config=config,
            frame_stop=mid,
        )
        w2 = run_tdt_loop(
            packed, encoder, valid_frames=FRAMES, config=config,
            initial_hidden=w1.hidden, initial_cell=w1.cell,
            initial_decoder_state=w1.decoder_state,
            initial_input_token=w1.input_token,
            initial_valid=w1.decoder_state_valid,
            frame_start=w1.final_frame,
        )
        joined = list(w1.token_ids) + list(w2.token_ids)
        jframes = list(w1.frame_indices) + list(w2.frame_indices)
        jdurs = list(w1.durations) + list(w2.durations)
        win_ok = (
            joined == list(ref.token_ids)
            and jframes == list(ref.frame_indices)
            and jdurs == list(ref.durations)
        )
        status = "PASS" if (same and win_ok) else "FAIL"
        if not (same and win_ok):
            failures += 1
        print(
            f"seed {seed} scale {scale}: {status} "
            f"emissions {len(ref.token_ids)} vs {len(got.token_ids)} "
            f"ref {ref_ms:.1f} ms loop {loop_ms:.1f} ms window_ok {win_ok}"
        )
        if not same:
            print("  ref :", list(ref.token_ids)[:16])
            print("  loop:", got.token_ids[:16])
        if not win_ok:
            print("  window joined:", joined[:16])
    print("FAILURES:", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
