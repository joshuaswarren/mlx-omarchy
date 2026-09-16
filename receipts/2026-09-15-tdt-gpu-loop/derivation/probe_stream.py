#!/usr/bin/env python3
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Dump the kernel's per-iteration decision stream and compare with ref."""
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
from coreml.vulkan_decoder_step import (  # noqa: E402
    _EXACT_FMA16,
    pack_step_weights,
    run_step,
)
from coreml import vulkan_tdt_loop as vtl  # noqa: E402

STREAM_N = 512


def stream_source() -> str:
    src = vtl._loop_glsl()
    init_anchor = "        s_ctl[" + str(vtl._CTL_SYM) + "] = 0;"
    src = src.replace(
        init_anchor,
        init_anchor + "\n        s_ctl[11] = 0;",
        1,
    )
    ctl_anchor = "                s_ctl[" + str(vtl._CTL_OP) + "] = op;"
    stream = (
        ctl_anchor + "\n"
        "                uint sidx = uint(s_ctl[11]);\n"
        "                if (sidx < " + str(STREAM_N) + "u) {\n"
        "                    stream[sidx * 3u + 0u] = float("
        "s_ctl[" + str(vtl._CTL_FRAME) + "]);\n"
        "                    stream[sidx * 3u + 1u] = float("
        "s_ctl[" + str(vtl._CTL_TOK) + "]);\n"
        "                    stream[sidx * 3u + 2u] = float("
        "s_ctl[" + str(vtl._CTL_DURI) + "]);\n"
        "                }\n"
        "                s_ctl[11] = s_ctl[11] + 1;"
    )
    if ctl_anchor not in src:
        raise SystemExit("ctl anchor not found")
    src = src.replace(ctl_anchor, stream, 1)
    return src


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

    kernel = mx.fast.metal_kernel(
        name="parakeet_tdt_loop_stream",
        input_names=[
            "embedding", "weights", "biases", "luts", "projector",
            "joint", "encoder", "hidden_in", "cell_in", "cfg",
        ],
        output_names=["emissions", "state", "ctl", "stream"],
        header=_EXACT_FMA16,
        source=stream_source(),
        compile_options={"math_mode": "safe"},
    )
    cfg = np.array(
        [FRAMES, FRAMES, int(config.blank_token_id),
         int(config.max_symbols_per_step), len(config.durations),
         *[int(d) for d in config.durations], FRAMES * 10],
        dtype=np.int32,
    )
    with mx.stream(mx.gpu):
        (emissions, state, ctl, stream) = kernel(
            inputs=[
                packed.embedding, packed.weights, packed.biases,
                packed.luts, packed.projector, packed.joint,
                encoder.reshape(-1),
                mx.zeros((1280,), mx.float32),
                mx.zeros((1280,), mx.float32),
                mx.array(cfg),
            ],
            output_shapes=[
                (FRAMES * 30,), (3200,), (8,), (STREAM_N * 3,),
            ],
            output_dtypes=[mx.int32, mx.float32, mx.int32, mx.float32],
            grid=(vtl._LOOP_THREADS, 1, 1),
            threadgroup=(vtl._LOOP_THREADS, 1, 1),
            stream=mx.gpu,
        )
        mx.eval(emissions, state, ctl, stream)
    st = np.asarray(stream).reshape(STREAM_N, 3)

    # host reference stream: replicate greedy_tdt_decode with run_step
    hidden = mx.zeros((2, 1, 640), dtype=mx.float32)
    cell = mx.zeros((2, 1, 640), dtype=mx.float32)
    mx.eval(hidden, cell)
    state_box = {"frame": 0, "out": None}

    def decoder_callback(token_id, cur_h, cur_c):
        out, tok, dur, _, _ = run_step(
            packed, cur_h, cur_c, token_id, encoder, state_box["frame"]
        )
        mx.eval(out)
        state_box["out"] = (tok, dur)
        return DecoderStep(
            out[0:640].reshape(1, 640),
            out[640:1920].reshape(2, 1, 640),
            out[1920:3200].reshape(2, 1, 640),
        )

    def joint_callback(frame_index, dec_state):
        state_box["frame"] = frame_index
        if state_box["out"] is not None:
            tok, dur = state_box["out"]
            state_box["out"] = None
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

    # rebuild the host's full decision stream (frame, tok, dur_index)
    decisions = []
    input_token = int(config.blank_token_id)
    valid = False
    frame = 0
    dec_i = 0
    ref_decs = list(
        zip(ref.token_ids, ref.frame_indices, ref.durations)
    )
    # replay via the recorded native emissions is insufficient for blanks;
    # instead rerun the loop capturing each decision
    cur_h, cur_c = hidden, cell
    decoder_state = None
    while frame < FRAMES:
        symbols = 0
        advanced = False
        while symbols < config.max_symbols_per_step:
            if not valid or input_token != int(config.blank_token_id):
                out, tok, dur, _, _ = run_step(
                    packed, cur_h, cur_c, input_token, encoder, frame
                )
                mx.eval(out)
                decoder_state = out[0:640].reshape(1, 640)
                cur_h = out[640:1920].reshape(2, 1, 640)
                cur_c = out[1920:3200].reshape(2, 1, 640)
                valid = True
                pending = (tok, dur)
            else:
                pending = None
            if pending is not None:
                tok, dur = pending
            else:
                out, tok, dur, _, _ = run_step(
                    packed, None, None, 0, encoder, frame,
                    skip_lstm=True, dec_in=decoder_state,
                )
                mx.eval(out)
            decisions.append((frame, int(tok), int(dur)))
            if int(tok) == int(config.blank_token_id):
                frame += max(int(config.durations[int(dur)]), 1)
                advanced = True
                break
            input_token = int(tok)
            symbols += 1
            if int(config.durations[int(dur)]) > 0:
                frame += int(config.durations[int(dur)])
                advanced = True
                break
        if not advanced:
            frame += 1

    print("kernel decisions (frame, tok, dur_idx):")
    kcount = 0
    for i in range(STREAM_N):
        f, t, d = st[i]
        if f == 0 and t == 0 and d == 0 and i > 0:
            break
        kcount = i + 1
        print(f"  it{i}: frame {int(f)} tok {int(t)} duri {int(d)}")
    print("host decisions:")
    for i, (f, t, d) in enumerate(decisions):
        print(f"  it{i}: frame {f} tok {t} duri {d}")
    same = all(
        int(st[i][0]) == decisions[i][0]
        and int(st[i][1]) == decisions[i][1]
        and int(st[i][2]) == decisions[i][2]
        for i in range(min(kcount, len(decisions)))
    ) and kcount == len(decisions)
    print("streams identical:", same, f"(kernel {kcount} vs host {len(decisions)})")
    print("ref emissions:", len(ref.token_ids))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
