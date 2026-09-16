#!/usr/bin/env python3
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Localize the GPU loop divergence: state probes + stream dump on seed 4."""
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
from coreml.vulkan_tdt_loop import run_tdt_loop  # noqa: E402


def main() -> int:
    mx.set_default_device(mx.gpu)
    lock = ReferenceLock.load(PKG / "coreml" / "parakeet-reference.lock")
    config = lock.tdt
    decoder = load_decoder(MODEL / "decoder.mlpackage")
    packed = pack_step_weights(decoder, MODEL / "joint.mlpackage")

    rng = np.random.default_rng(20260918)
    scale = 4.0
    FRAMES = 24
    enc_np = (rng.standard_normal((1, FRAMES, 640)) * scale).astype(np.float32)
    encoder = mx.array(enc_np)
    mx.eval(encoder)

    # ---- probe 1: single decode from non-zero random state ----
    h = mx.array((rng.standard_normal((2, 1, 640)) * 0.5).astype(np.float32))
    c = mx.array((rng.standard_normal((2, 1, 640)) * 0.5).astype(np.float32))
    mx.eval(h, c)
    frame0 = mx.array(enc_np[:, 0:1, :])
    mx.eval(frame0)
    rs_state, rs_tok, rs_dur, _, _ = run_step(packed, h, c, 8192, frame0, 0)
    mx.eval(rs_state)
    got = run_tdt_loop(
        packed, frame0, valid_frames=1, config=config,
        initial_hidden=h, initial_cell=c,
    )
    rs = np.asarray(rs_state).ravel()
    pj_d = float(np.abs(rs[0:640] - got.decoder_state.ravel()).max())
    h_d = float(np.abs(rs[640:1920] - got.hidden.ravel()).max())
    c_d = float(np.abs(rs[1920:3200] - got.cell.ravel()).max())
    print(
        f"probe1 single decode(blank) from random state: "
        f"pj_max {pj_d} h_max {h_d} c_max {c_d} "
        f"({'PASS' if pj_d == 0 and h_d == 0 and c_d == 0 else 'FAIL'})"
    )

    # ---- probe 2: full stream dump for seed 4 ----
    hidden = mx.zeros((2, 1, 640), dtype=mx.float32)
    cell = mx.zeros((2, 1, 640), dtype=mx.float32)
    mx.eval(hidden, cell)
    state = {"frame": 0, "out": None}
    calls = []

    def decoder_callback(token_id, current_hidden, current_cell):
        out, tok, dur, _, _ = run_step(
            packed, current_hidden, current_cell, token_id,
            encoder, state["frame"],
        )
        mx.eval(out)
        state["out"] = (tok, dur)
        calls.append(
            ("decode", state["frame"], token_id,
             np.asarray(out).ravel().copy())
        )
        return DecoderStep(
            out[0:640].reshape(1, 640),
            out[640:1920].reshape(2, 1, 640),
            out[1920:3200].reshape(2, 1, 640),
        )

    def joint_callback(frame_index, dec_state):
        state["frame"] = frame_index
        if state["out"] is not None:
            tok, dur = state["out"]
            state["out"] = None
            return JointDecision(tok, dur)
        out, tok, dur, _, _ = run_step(
            packed, None, None, 0, encoder, frame_index,
            skip_lstm=True, dec_in=dec_state,
        )
        mx.eval(out)
        return JointDecision(tok, dur)

    ref = greedy_tdt_decode(
        valid_frames=FRAMES, config=config,
        initial_hidden=hidden, initial_cell=cell,
        run_decoder=decoder_callback, run_joint=joint_callback,
    )
    print("ref  tokens :", list(ref.token_ids))
    print("ref  frames :", list(ref.frame_indices))
    print("ref  durs   :", list(ref.durations))
    got = run_tdt_loop(
        packed, encoder, valid_frames=FRAMES, config=config,
        initial_hidden=hidden, initial_cell=cell,
    )
    print("loop tokens :", got.token_ids)
    print("loop frames :", got.frame_indices)
    print("loop durs   :", got.durations)
    ref_h = np.asarray(ref.hidden).ravel()
    got_h = got.hidden.ravel()
    ref_c = np.asarray(ref.cell).ravel()
    got_c = got.cell.ravel()
    print(
        "final state delta: h_max",
        float(np.abs(ref_h - got_h).max()),
        "c_max",
        float(np.abs(ref_c - got_c).max()),
    )
    for entry in calls:
        print(
            f"ref decode frame {entry[1]} token {entry[2]} "
            f"h0[:4] {np.array2string(entry[3][:4], precision=6)}"
        )
    # probe 3: in-kernel state right after the first decode (frame_stop=1)
    stop1 = run_tdt_loop(
        packed, encoder, valid_frames=FRAMES, config=config,
        initial_hidden=hidden, initial_cell=cell, frame_stop=1,
    )
    d0 = calls[0][3]
    print(
        "probe3 post-decode@0 state: pj_max",
        float(np.abs(d0[0:640] - stop1.decoder_state.ravel()).max()),
        "h_max",
        float(np.abs(d0[640:1920] - stop1.hidden.ravel()).max()),
        "c_max",
        float(np.abs(d0[1920:3200] - stop1.cell.ravel()).max()),
    )
    print(
        "probe3 h0[:4] ref  ",
        np.array2string(d0[640:644], precision=6),
        "\nprobe3 h0[:4] loop ",
        np.array2string(stop1.hidden.ravel()[0:4], precision=6),
    )
    from coreml.reference import TdtConfig

    cfg439 = TdtConfig(
        blank_token_id=439,
        durations=list(config.durations),
        max_symbols_per_step=config.max_symbols_per_step,
        vocab_size=config.vocab_size,
    )
    for label, h0, c0 in (
        ("zeros", hidden, cell),
        ("post-decode@0", stop1.hidden, stop1.cell),
    ):
        rs_state, _, _, _, _ = run_step(packed, h0, c0, 439, frame0, 0)
        mx.eval(rs_state)
        p = run_tdt_loop(
            packed, frame0, valid_frames=1, config=cfg439,
            initial_hidden=h0, initial_cell=c0,
        )
        rs = np.asarray(rs_state).ravel()
        print(
            f"probe token439/{label}: pj_max",
            float(np.abs(rs[0:640] - p.decoder_state.ravel()).max()),
            "h_max",
            float(np.abs(rs[640:1920] - p.hidden.ravel()).max()),
            "c_max",
            float(np.abs(rs[1920:3200] - p.cell.ravel()).max()),
        )
    # probe 6: state after the in-loop decode(439)@14 (frame_stop=15)
    S14 = calls[1][3]
    frame14 = mx.array(enc_np[:, 14:15, :])
    mx.eval(frame14)
    rs6, _, _, _, _ = run_step(
        packed,
        mx.array(S14[640:1920].reshape(2, 1, 640)),
        mx.array(S14[1920:3200].reshape(2, 1, 640)),
        439, frame14, 14,
    )
    mx.eval(rs6)
    stop15 = run_tdt_loop(
        packed, encoder, valid_frames=FRAMES, config=config,
        initial_hidden=hidden, initial_cell=cell, frame_stop=15,
    )
    rs6 = np.asarray(rs6).ravel()
    print(
        "probe6 decode(439)@14 in-loop: pj_max",
        float(np.abs(rs6[0:640] - stop15.decoder_state.ravel()).max()),
        "h_max",
        float(np.abs(rs6[640:1920] - stop15.hidden.ravel()).max()),
        "c_max",
        float(np.abs(rs6[1920:3200] - stop15.cell.ravel()).max()),
        "emissions", stop15.token_ids,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
