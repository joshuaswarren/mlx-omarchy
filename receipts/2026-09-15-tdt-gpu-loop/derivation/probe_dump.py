#!/usr/bin/env python3
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Per-decode pre/post state dump of the GPU loop vs run_step.

Debug variant of the loop kernel dumps, for every decode k (indexed by a
dedicated decode counter, s_ctl[10]): the shared hidden/cell BEFORE the
decode (dbg2) and pj/hidden/cell AFTER it (dbg).  The host replays the
known decision stream through run_step and compares each slot.
"""
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
from coreml.vulkan_decoder_step import (  # noqa: E402
    _EXACT_FMA16,
    pack_step_weights,
    run_step,
)
from coreml import vulkan_tdt_loop as vtl  # noqa: E402

MAX_STEPS = 16
DBG_LEN = 128 * 3200 + 2 * MAX_STEPS
DBG2_LEN = 128 * 2560 + 3 * MAX_STEPS
CTL_DECODES = 10  # free s_ctl slot used as the decode counter


def dbg_source() -> str:
    src = vtl._loop_glsl()
    init_anchor = "        s_ctl[" + str(vtl._CTL_SYM) + "] = 0;"
    if init_anchor not in src:
        raise SystemExit("init anchor not found")
    src = src.replace(
        init_anchor,
        init_anchor + "\n        s_ctl[" + str(CTL_DECODES) + "] = 0;",
        1,
    )

    pre_anchor = "                int tok = s_ctl[" + str(vtl._CTL_INPUT) + "];"
    pre_dump = (
        "                uint dslot = uint(s_ctl[" + str(CTL_DECODES)
        + "]) % " + str(MAX_STEPS) + "u;\n"
        "                for (uint i = t; i < 1280u; i += "
        + str(vtl._LT) + "u) {\n"
        "                    dbg2[dslot * 2560u + i] = s_hidden[i];\n"
        "                    dbg2[dslot * 2560u + 1280u + i] = s_cell[i];\n"
        "                }\n"
        "                if (t == 0u) {\n"
        "                    dbg2[128u * 2560u + dslot * 3u + 0u] = float("
        "s_ctl[" + str(vtl._CTL_FRAME) + "]);\n"
        "                    dbg2[128u * 2560u + dslot * 3u + 1u] = float("
        "s_ctl[" + str(vtl._CTL_INPUT) + "]);\n"
        "                    dbg2[128u * 2560u + dslot * 3u + 2u] = float("
        "s_ctl[" + str(vtl._CTL_VALID) + "]);\n"
        "                    s_ctl[" + str(CTL_DECODES) + "] = "
        "s_ctl[" + str(CTL_DECODES) + "] + 1;\n"
        "                }\n"
        "                threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        + pre_anchor
    )
    if pre_anchor not in src:
        raise SystemExit("pre anchor not found")
    src = src.replace(pre_anchor, pre_dump)

    post_anchor = (
        "                if (t == 0u) { s_ctl["
        + str(vtl._CTL_VALID)
        + "] = 1; }\n            }"
    )
    post_dump = (
        "                uint pslot = uint(s_ctl[" + str(CTL_DECODES)
        + "]) % " + str(MAX_STEPS) + "u - 1u;\n"
        "                for (uint i = t; i < 640u; i += "
        + str(vtl._LT) + "u) {\n"
        "                    dbg[pslot * 3200u + i] = float(s_pj[i]);\n"
        "                }\n"
        "                for (uint i = t; i < 1280u; i += "
        + str(vtl._LT) + "u) {\n"
        "                    dbg[pslot * 3200u + 640u + i] = s_hidden[i];\n"
        "                    dbg[pslot * 3200u + 1920u + i] = s_cell[i];\n"
        "                }\n"
        "                threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        + post_anchor
    )
    if post_anchor not in src:
        raise SystemExit("post anchor not found")
    src = src.replace(post_anchor, post_dump)
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

    dbg_kernel = mx.fast.metal_kernel(
        name="parakeet_tdt_loop_dbg2",
        input_names=[
            "embedding", "weights", "biases", "luts", "projector",
            "joint", "encoder", "hidden_in", "cell_in", "cfg",
        ],
        output_names=["emissions", "state", "ctl", "dbg", "dbg2"],
        header=_EXACT_FMA16,
        source=dbg_source(),
        compile_options={"math_mode": "safe"},
    )
    cfg = np.array(
        [FRAMES, FRAMES, int(config.blank_token_id),
         int(config.max_symbols_per_step), len(config.durations),
         *[int(d) for d in config.durations], FRAMES * 10],
        dtype=np.int32,
    )
    with mx.stream(mx.gpu):
        (emissions, state, ctl, dbg, dbg2) = dbg_kernel(
            inputs=[
                packed.embedding, packed.weights, packed.biases,
                packed.luts, packed.projector, packed.joint,
                encoder.reshape(-1),
                mx.zeros((1280,), mx.float32),
                mx.zeros((1280,), mx.float32),
                mx.array(cfg),
            ],
            output_shapes=[
                (FRAMES * 30,), (3200,), (8,), (DBG_LEN,), (DBG2_LEN,),
            ],
            output_dtypes=[
                mx.int32, mx.float32, mx.int32, mx.float32, mx.float32,
            ],
            grid=(vtl._LOOP_THREADS, 1, 1),
            threadgroup=(vtl._LOOP_THREADS, 1, 1),
            stream=mx.gpu,
        )
        mx.eval(emissions, state, ctl, dbg, dbg2)
    dbg = np.asarray(dbg)
    dbg2 = np.asarray(dbg2)
    em = np.asarray(emissions)
    count = int(np.asarray(ctl)[0])
    print("emissions:", count, em[: count * 3].reshape(count, 3).tolist())
    ndecodes = int(dbg2[128 * 2560 + 0])  # not stored; use known 2
    print("decodes recorded:", int(dbg2[128 * 2560 + 2 * 3 + 0]), "(frame of decode2 if any)")

    # host chain: known ground truth for the first two decodes
    out0, _, _, _, _ = run_step(
        packed, mx.zeros((2, 1, 640), dtype=mx.float32),
        mx.zeros((2, 1, 640), dtype=mx.float32),
        int(config.blank_token_id), encoder, 0,
    )
    mx.eval(out0)
    S0 = np.asarray(out0).ravel().copy()
    out1, _, _, _, _ = run_step(
        packed, out0[640:1920].reshape(2, 1, 640),
        out0[1920:3200].reshape(2, 1, 640), 439, encoder, 14,
    )
    mx.eval(out1)
    S1 = np.asarray(out1).ravel().copy()

    for k, expected in ((0, S0), (1, S1)):
        pre = dbg2[k * 2560:(k + 1) * 2560]
        post = dbg[k * 3200:(k + 1) * 3200]
        kframe = int(dbg2[128 * 2560 + k * 3 + 0])
        ktoken = int(dbg2[128 * 2560 + k * 3 + 1])
        kvalid = int(dbg2[128 * 2560 + k * 3 + 2])
        pre_h_d = float(np.abs(pre[0:1280] - expected[640:1920]).max())
        pre_c_d = float(np.abs(pre[1280:2560] - expected[1920:3200]).max())
        pj_d = float(np.abs(post[0:640] - expected[0:640]).max())
        h_d = float(np.abs(post[640:1920] - expected[640:1920]).max())
        c_d = float(np.abs(post[1920:3200] - expected[1920:3200]).max())
        print(
            f"decode {k}: frame {kframe} in_token {ktoken} valid {kvalid} | "
            f"pre h_d {pre_h_d} c_d {pre_c_d} | "
            f"post pj_d {pj_d} h_d {h_d} c_d {c_d}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
