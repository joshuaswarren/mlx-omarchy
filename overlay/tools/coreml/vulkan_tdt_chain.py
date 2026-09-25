# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Device-chained greedy TDT decode: per-emission slots, two fused kernels.

Each slot runs as TWO single-workgroup dispatches (Jwm1Parity10 fusion of
the former six per-slot kernels; the landed per-kernel contracts are
preserved arithmetic-for-arithmetic):

* ``slot_a``: layer-0 block chains + fold/cell.  Each of the ten 128-wide
  block chains is computed exactly as before (exact_fma16 from fp16 0,
  ascending k, one chain per (block, n)); the fold adds the ten block
  sums in ascending block order with the first term assigned, not added.
* ``slot_b``: layer-1 block chains + fold/cell + projector + joint window
  (relu, fp32 ascending-k joint, argmax) + the emission/control walk.
  All cross-thread reductions happen inside one 1024-thread workgroup,
  so the former partial-buffer round trip through ``control`` is gone.

The block chains are independent up to the fold (receipts
2026-09-15-tdt-gpu-step2), which is what makes the single-workgroup
restructure bit-exact.  The joint argmax is a max-with-tie comparison;
for non-NaN logits the winner is order-independent.  The only scan-order
sensitivity is the never-observed all-NaN pathology, which the golden
contracts would catch.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np

from .vulkan_decoder_step import _EXACT_FMA16

_WINDOW_ROWS = 7  # row 0 = emission joint, rows 1..6 = frame-entry spec
_JOINT_OUT = 8198

# Control word layout (int32, one row per slot).
_C_FRAME = 0
_C_SYM = 1
_C_COUNT = 2
_C_ERR = 3
_C_SKIP = 4  # this slot is joint-only (previous window exhausted)
_C_TOKEN = 5
_C_DONE = 6
_C_RUN = 7
_C_LAST = 8  # slot index of the last executed decoder step (-1: none)
_CTRL_WORDS = 9

# Config layout (int32).
_G_VALID = 0
_G_BLANK = 1
_G_MAXSYM = 2
_G_NDUR = 3
_G_DUR0 = 4  # five duration classes
_G_CAP = 9
_G_CAP3 = 10

# Workgroup-uniform gates, inlined (the translator emits one function).
_STEP_LIVE = "bool step_live = (ctl[_C_RUN] != 0) && (ctl[_C_DONE] == 0) && (ctl[_C_SKIP] == 0);"
_LOOP_LIVE = "bool loop_live = (ctl[_C_RUN] != 0) && (ctl[_C_DONE] == 0);"

# Ten block rounds; each round computes the 2560 block chains of one
# 128-wide k-block into sh_bs, then every (gate, lane) pair folds the
# block sum into its running bacc.  Ascending orders are the landed
# contract: ascending k inside a chain, ascending block in the fold, and
# the round-0 term assigned (not added) to preserve -0.  Thread t owns
# (gate, lane) pairs p = t + r*1024, p < 2560, so p is the flattened
# column n = lane + gate*640 the landed fold reduces over.
_CHAIN_FOLD_ROUNDS = """
    for (uint i = t; i < 1280u; i += 1024u) {
        if (i < 640u) {
            INPUT_A
        } else {
            sh_a[i] = float16_t(hidden_in[HID_BASE + (i - 640u)]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint round = 0u; round < 10u; ++round) {
        for (uint c = t; c < 2560u; c += 1024u) {
            float16_t bc = float16_t(0.0f);
            uint k = round * 128u;
            uint wbase = WB_LAYER * 3276800u + k * 2560u + c;
            for (uint j = 0u; j < 128u; ++j) {
                bc = exact_fma16(bc, sh_a[k], weights[wbase]);
                ++k;
                wbase += 2560u;
            }
            sh_bs[c] = bc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint r = 0u; r < 3u; ++r) {
            uint p = t + r * 1024u;
            if (p < 2560u) {
                if (round == 0u) {
                    bacc[r] = sh_bs[p];
                } else {
                    bacc[r] = float16_t(bacc[r] + sh_bs[p]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint r = 0u; r < 3u; ++r) {
        uint p = t + r * 1024u;
        if (p < 2560u) {
            sh_bs[p] = float16_t(bacc[r] + biases[BI_LAYER * 2560u + p]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
"""

# The landed gate-pairing epilogue, reading the four gate pre-activations
# of one lane out of sh_bs.  CELLOFF is the cell row (0 for layer 0).
_GATE_CELL = """
        uint lane = t;
        float16_t si = luts[packHalf2x16(vec2(float(sh_bs[lane]), 0.0f)) & 0xFFFFu];
        float16_t sf = luts[packHalf2x16(vec2(float(sh_bs[640u + lane]), 0.0f)) & 0xFFFFu];
        float16_t so = luts[packHalf2x16(vec2(float(sh_bs[1280u + lane]), 0.0f)) & 0xFFFFu];
        float16_t tg = luts[65536u + (packHalf2x16(vec2(float(sh_bs[1920u + lane]), 0.0f)) & 0xFFFFu)];
        float16_t c0 = float16_t(cell_in[CELLOFF * 640u + lane]);
        float16_t fprod = sf * c0;
        float16_t c1 = exact_fma16(fprod, si, tg);
        uint cb = packHalf2x16(vec2(float(c1), 0.0f)) & 0xFFFFu;
        float16_t tc = luts[65536u + cb];
        float16_t h1 = so * tc;
"""

_SLOT_A_BODY = (
    _STEP_LIVE
    + """
    uint t = thread_index_in_threadgroup.x;
    if (!step_live) {
        if (t < 640u) {
            h0_out[t] = 0.0f;
            c1_out[t] = 0.0f;
        }
        return;
    }
    threadgroup float16_t sh_a[1280];
    threadgroup float16_t sh_bs[2560];
    float16_t bacc[3];
    uint token_id = ctl[_C_TOKEN];
"""
    + _CHAIN_FOLD_ROUNDS.replace(
        "INPUT_A",
        "uint tbase = uint(token_id);\n"
        "                if (token_id < 0) { tbase = uint(token_id + 8193); }\n"
        "                sh_a[i] = embedding[tbase * 640u + i];",
    ).replace("HID_BASE", "0u")
    .replace("WB_LAYER", "0u")
    .replace("BI_LAYER", "0u")
    + """
    if (t < 640u) {
"""
    + _GATE_CELL.replace("CELLOFF", "0u")
    + """
        h0_out[lane] = float(h1);
        c1_out[lane] = float(c1);
    }
"""
)

_SLOT_B_BODY = (
    _STEP_LIVE
    + _LOOP_LIVE
    + """
    uint t = thread_index_in_threadgroup.x;
    threadgroup float16_t sh_a[1280];
    threadgroup float16_t sh_bs[2560];
    threadgroup float sh_h1[640];
    threadgroup float16_t sh_pj16[640];
    threadgroup float16_t sh_relu[ROWS6];
    threadgroup float s_val[1024];
    threadgroup uint s_idx[1024];
    threadgroup float s_dval[5];
    threadgroup int sh_tok[ROWSC];
    threadgroup int sh_duri[ROWSC];
    for (uint i = t; i < cfg[_G_CAP3]; i += 1024u) {
        emissions_next[i] = emissions_in[i];
    }
    if (!step_live) {
        if (t < 640u) {
            h_state[t] = h_state_in[t];
            h_state[640u + t] = h_state_in[640u + t];
            c_state[t] = c_state_in[t];
            c_state[640u + t] = c_state_in[640u + t];
            sh_pj16[t] = pj16_in[t];
            pj16[t] = pj16_in[t];
            pj[t] = float(pj16_in[t]);
        }
    } else {
        float16_t bacc[3];
        uint token_id = ctl[_C_TOKEN];
"""
    + _CHAIN_FOLD_ROUNDS.replace(
        "INPUT_A",
        "sh_a[i] = float16_t(x_in[i]);",
    ).replace("HID_BASE", "640u")
    .replace("WB_LAYER", "1u")
    .replace("BI_LAYER", "1u")
    + """
        if (t < 640u) {
"""
    + _GATE_CELL.replace("CELLOFF", "1u")
    + """
            h_state[lane] = float(h0_in[lane]);
            h_state[640u + lane] = float(h1);
            c_state[lane] = c0_in[lane];
            c_state[640u + lane] = float(c1);
            sh_h1[lane] = float(h1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t < 640u) {
            uint lane = t;
            precise float acc = 0.0f;
            for (uint k = 0u; k < 640u; ++k) {
                acc = acc + float(float16_t(sh_h1[k]))
                      * float(projector[k * 640u + lane]);
            }
            float16_t pjv = float16_t(float16_t(acc) + projector[640u * 640u + lane]);
            sh_pj16[lane] = pjv;
            pj16[lane] = pjv;
            pj[lane] = float(pjv);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (loop_live) {
        int base = ctl[_C_FRAME];
        int valid = cfg[_G_VALID];
        for (uint slot = 0u; slot < ROWSU; ++slot) {
            int frame = base + int(slot);
            for (uint i = t; i < 640u; i += 1024u) {
                if (frame < valid) {
                    float16_t rlv = float16_t(encoder[uint(frame) * 640u + i])
                                    + sh_pj16[i];
                    sh_relu[slot * 640u + i] =
                        (rlv > float16_t(0.0f)) ? rlv : float16_t(0.0f);
                } else {
                    sh_relu[slot * 640u + i] = float16_t(0.0f);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint nrows = (ctl[_C_COUNT] == 0u) ? ROWSU : 1u;
        for (uint row = 0u; row < nrows; ++row) {
            // Each thread reduces its own token columns (ascending j,
            // landed comparator) into (bv, bix); duration columns feed
            // s_dval.  j = 8192 (token) belongs to lane 6; j >= 8193 are
            // the five duration columns.
            float bv = -3.0e38f;
            uint bix = 0u;
            bool have = false;
            for (uint x = 0u; x < 9u; ++x) {
                uint j = t + x * 1024u;
                if (j >= 8198u) { break; }
                precise float acc = 0.0f;
                for (uint k = 0u; k < 640u; ++k) {
                    precise float w = float(joint[k * 8198u + j]);
                    acc = acc + float(sh_relu[row * 640u + k]) * w;
                }
                if (j < 8193u) {
                    float v = float(float16_t(acc) + joint[640u * 8198u + j]);
                    if (!have) {
                        bv = v; bix = j; have = true;
                    } else {
                        bool take;
                        if (v != v) { take = (bv == bv) ? true : (j < bix); }
                        else if (bv != bv) { take = false; }
                        else { take = (v > bv) || (v == bv && j < bix); }
                        if (take) { bv = v; bix = j; }
                    }
                } else {
                    uint d = j - 8193u;
                    if (d < 5u) {
                        s_dval[d] = float(float16_t(acc) + joint[640u * 8198u + j]);
                    }
                }
            }
            s_val[t] = bv;
            s_idx[t] = bix;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = 512u; stride >= 1u; stride >>= 1u) {
                if (t < stride) {
                    float v2 = s_val[t + stride];
                    uint ix2 = s_idx[t + stride];
                    bool take;
                    if (v2 != v2) { take = (bv != bv) ? (ix2 < bix) : true; }
                    else if (bv != bv) { take = false; }
                    else { take = (v2 > bv) || (v2 == bv && ix2 < bix); }
                    if (take) { bv = v2; bix = ix2; }
                    s_val[t] = bv;
                    s_idx[t] = bix;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (t == 0u) {
                sh_tok[row] = int(bv == bv ? bix : 0u);
                float dv = s_dval[0];
                uint di = 0u;
                for (uint i = 1u; i < 5u; ++i) {
                    float v2 = s_dval[i];
                    uint ix2 = i;
                    bool take;
                    if (v2 != v2) { take = (dv == dv) ? false : (ix2 < di); }
                    else if (dv != dv) { take = true; }
                    else { take = (v2 > dv) || (v2 == dv && ix2 < di); }
                    if (take) { dv = v2; di = ix2; }
                }
                sh_duri[row] = int(di);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    } else {
        if (t == 0u) {
            for (uint row = 0u; row < ROWSU; ++row) {
                sh_tok[row] = 0;
                sh_duri[row] = 0;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t == 0u) {
        int frame = ctl[_C_FRAME];
        int sym = ctl[_C_SYM];
        int count = ctl[_C_COUNT];
        int err = ctl[_C_ERR];
        int token = ctl[_C_TOKEN];
        int done = ctl[_C_DONE];
        int run = ctl[_C_RUN];
        int need = 0;
        int adv = 0;
        int last = ctl[_C_LAST];
        if (ctl[_C_SKIP] == 0 && run != 0 && done == 0) { last = sid[0]; }
        if (run != 0 && done == 0 && err == 0) {
            int r = 0;
            while (sym < cfg[_G_MAXSYM]) {
                if (frame >= cfg[_G_VALID]) { done = 1; break; }
                if (r >= ROWSC) { need = 1; break; }
                int tok = sh_tok[r];
                int duri = sh_duri[r];
                if (duri < 0 || duri >= cfg[_G_NDUR]) { err = 1; break; }
                int d = cfg[_G_DUR0 + duri];
                if (tok == cfg[_G_BLANK]) {
                    int hop = (d > 1) ? d : 1;
                    frame += hop;
                    if (count != 0) {
                        // state changes at every frame entry once a token
                        // has been emitted: the next slot must re-step
                        sym = 0;
                        break;
                    }
                    r += hop;
                    adv = 1;
                    sym = 0;
                    continue;
                }
                if (count >= cfg[_G_CAP]) { err = 2; break; }
                emissions_next[count * 3 + 0] = tok;
                emissions_next[count * 3 + 1] = frame;
                emissions_next[count * 3 + 2] = d;
                count++;
                token = tok;
                sym++;
                if (d > 0) {
                    frame += d;
                    sym = 0;
                    break;
                }
                break;  // d == 0: next slot re-runs the decoder at this frame
            }
            if (err == 0 && done == 0 && adv == 0 && sym >= cfg[_G_MAXSYM]) {
                frame += 1;
                sym = 0;
            }
            if (frame >= cfg[_G_VALID]) { done = 1; }
        }
        ctl_next[_C_FRAME] = frame;
        ctl_next[_C_SYM] = sym;
        ctl_next[_C_COUNT] = count;
        ctl_next[_C_ERR] = err;
        ctl_next[_C_SKIP] = need;
        ctl_next[_C_TOKEN] = token;
        ctl_next[_C_DONE] = done;
        ctl_next[_C_RUN] = (err == 0 && done == 0) ? 1 : 0;
        ctl_next[_C_LAST] = last;
    }
"""
)


def _subst(template: str, **repl: str) -> str:
    out = template
    for key, value in repl.items():
        out = out.replace(key, value)
    return out


def _build_slot_a() -> str:
    return _subst(
        _SLOT_A_BODY,
        _C_TOKEN=str(_C_TOKEN),
        _C_RUN=str(_C_RUN),
        _C_DONE=str(_C_DONE),
        _C_SKIP=str(_C_SKIP),
    )


def _build_slot_b() -> str:
    return _subst(
        _SLOT_B_BODY,
        ROWS6=str(_WINDOW_ROWS * 640),
        ROWSU=f"{_WINDOW_ROWS}u",
        ROWSC=str(_WINDOW_ROWS),
        _C_FRAME=str(_C_FRAME),
        _C_RUN=str(_C_RUN),
        _C_DONE=str(_C_DONE),
        _C_SKIP=str(_C_SKIP),
        _C_COUNT=str(_C_COUNT),
        _C_TOKEN=str(_C_TOKEN),
        _C_SYM=str(_C_SYM),
        _C_ERR=str(_C_ERR),
        _C_LAST=str(_C_LAST),
        _G_VALID=str(_G_VALID),
        _G_CAP3=str(_G_CAP3),
        _G_MAXSYM=str(_G_MAXSYM),
        _G_NDUR=str(_G_NDUR),
        _G_DUR0=str(_G_DUR0),
        _G_CAP=str(_G_CAP),
        _G_BLANK=str(_G_BLANK),
    )


@dataclass
class TdtChainOutput:
    """Emissions and recurrent state produced by the device-chained loop."""

    token_ids: list[int]
    frame_indices: list[int]
    durations: list[int]
    hidden: object
    cell: object
    final_frame: int
    slots_used: int = 0
    decode_path: str = "gpu-chain"


@cache
def _mx():
    import mlx.core as mx

    return mx


@cache
def _slot_a_kernel():
    mx = _mx()
    return mx.fast.metal_kernel(
        name="parakeet_tdt_slot_a",
        input_names=["embedding", "weights", "hidden_in", "cell_in",
                     "biases", "luts", "ctl"],
        output_names=["h0_out", "c1_out"],
        header=_EXACT_FMA16,
        source=_build_slot_a(),
        compile_options={"math_mode": "safe"},
    )


@cache
def _slot_b_kernel():
    mx = _mx()
    return mx.fast.metal_kernel(
        name="parakeet_tdt_slot_b",
        input_names=["embedding", "weights", "hidden_in", "x_in", "cell_in",
                     "h0_in", "c0_in", "h_state_in", "c_state_in", "pj16_in",
                     "biases", "luts", "projector", "emissions_in", "encoder",
                     "joint", "ctl", "cfg", "sid"],
        output_names=["h_state", "c_state", "pj", "pj16", "ctl_next",
                      "emissions_next"],
        header=_EXACT_FMA16,
        source=_build_slot_b(),
        compile_options={"math_mode": "safe"},
    )


@cache
def _sid_array(slot: int):
    import mlx.core as mx

    return mx.array(np.array([slot], np.int32))


def run_tdt_chain(
    packed,
    encoder,
    valid_frames: int,
    config,
    initial_hidden,
    initial_cell,
    slots_per_chunk: int = 64,
    max_chunks: int = 12,
) -> TdtChainOutput:
    """Greedy TDT decode as a device-chained slot schedule.

    ``packed`` is the shared ``StepWeights``; ``encoder`` the fp32 encoder
    output (1, F, 640); ``config`` the pinned ``TdtConfig``.  Control
    state lives in device buffers; the host syncs once per chunk and only
    to decide whether another chunk is needed.
    """
    mx = _mx()
    blank = int(config.blank_token_id)
    durs = [int(d) for d in config.durations]
    maxsym = int(config.max_symbols_per_step)
    cap = max(1, int(valid_frames) * maxsym)
    total_slots = int(valid_frames) + cap + 1
    cfg = np.zeros((_G_CAP3 + 1,), np.int32)
    cfg[_G_VALID] = int(valid_frames)
    cfg[_G_BLANK] = blank
    cfg[_G_MAXSYM] = maxsym
    cfg[_G_NDUR] = len(durs)
    for i, d in enumerate(durs):
        cfg[_G_DUR0 + i] = d
    cfg[_G_CAP] = cap
    cfg[_G_CAP3] = cap * 3
    cfg_dev = mx.array(cfg)

    ctl0 = np.zeros((_CTRL_WORDS,), np.int32)
    ctl0[_C_FRAME] = 0
    ctl0[_C_SYM] = 0
    ctl0[_C_COUNT] = 0
    ctl0[_C_ERR] = 0
    ctl0[_C_SKIP] = 1 if valid_frames == 0 else 0
    ctl0[_C_TOKEN] = blank
    ctl0[_C_DONE] = 0
    ctl0[_C_RUN] = 1
    ctl0[_C_LAST] = -1

    emissions = mx.array(np.zeros((cap * 3,), np.int32))
    enc_flat = encoder.reshape(-1)
    cap3 = cap * 3

    h_state = initial_hidden.reshape(1280,)
    c_state = initial_cell.reshape(1280,)
    pj16_prev = mx.array(np.zeros((640,), np.float16))
    folds = {}  # slot -> slot_b state outputs, for the final state readout
    ctl_rows = [mx.array(ctl0)]
    slot = 0
    chunks = 0

    with mx.stream(mx.gpu):
        while chunks < max_chunks and slot < total_slots:
            end = min(slot + slots_per_chunk, total_slots)
            for i in range(slot, end):
                ctl_i = ctl_rows[i]
                sid = _sid_array(i)
                h0, c0 = _slot_a_kernel()(
                    inputs=[packed.embedding, packed.weights, h_state,
                            c_state, packed.biases, packed.luts, ctl_i],
                    output_shapes=[(640,), (640,)],
                    output_dtypes=[mx.float32, mx.float32],
                    grid=(1024, 1, 1), threadgroup=(1024, 1, 1),
                    stream=mx.gpu,
                )
                h_s, c_s, pj, pj16, ctl_next, emissions_next = _slot_b_kernel()(
                    inputs=[packed.embedding, packed.weights, h_state, h0,
                            c_state, h0, c0, h_state, c_state, pj16_prev,
                            packed.biases, packed.luts, packed.projector,
                            emissions, enc_flat, packed.joint, ctl_i,
                            cfg_dev, sid],
                    output_shapes=[(1280,), (1280,), (640,), (640,),
                                   (_CTRL_WORDS,), (cap3,)],
                    output_dtypes=[mx.float32, mx.float32, mx.float32,
                                   mx.float16, mx.int32, mx.int32],
                    grid=(1024, 1, 1), threadgroup=(1024, 1, 1),
                    stream=mx.gpu,
                )
                ctl_rows.append(ctl_next)
                emissions = emissions_next
                h_state, c_state = h_s, c_s
                pj16_prev = pj16
                folds[i] = (h_s, c_s, pj)
            mx.eval(ctl_rows[end])
            row = np.asarray(ctl_rows[end])
            slot = end
            chunks += 1
            if int(row[_C_DONE]) or not int(row[_C_RUN]):
                break

    row = np.asarray(ctl_rows[slot])
    err = int(row[_C_ERR])
    if err != 0:
        reason = ("duration index outside class count" if err == 1
                  else "emission capacity exceeded")
        raise ValueError(f"[omarchy-parakeet] TDT chain failed: {reason}")
    if not int(row[_C_DONE]):
        raise ValueError(
            "[omarchy-parakeet] TDT chain slot budget exhausted "
            f"({slot} slots, frame {int(row[_C_FRAME])} of {valid_frames})"
        )
    count = int(row[_C_COUNT])
    em = np.asarray(emissions)[: count * 3].reshape(count, 3)
    last = int(row[_C_LAST])
    if last >= 0:
        h_s, c_s, _pj = folds[last]
        hidden = np.asarray(h_s).reshape(2, 1, 640)
        cell = np.asarray(c_s).reshape(2, 1, 640)
    else:
        hidden = np.asarray(initial_hidden).reshape(2, 1, 640)
        cell = np.asarray(initial_cell).reshape(2, 1, 640)
    return TdtChainOutput(
        token_ids=[int(x) for x in em[:, 0]],
        frame_indices=[int(x) for x in em[:, 1]],
        durations=[int(x) for x in em[:, 2]],
        hidden=hidden,
        cell=cell,
        final_frame=int(row[_C_FRAME]),
        slots_used=slot,
    )
