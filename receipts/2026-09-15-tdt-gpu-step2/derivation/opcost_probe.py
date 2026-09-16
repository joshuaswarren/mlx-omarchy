#!/usr/bin/env python3
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Cost attribution for the exact_fma16 building blocks.

Fixed geometry (20 x 512), fixed chain shape (10 chains x 128 steps per
thread == one layer of GEMV work per thread); only the inner op changes.
Every variant is its own literal source string with a unique kernel name.
"""
import json
import sys
import time

import numpy as np

FMA_HDR = """
float16_t exact_fma16(float16_t acc, float16_t x, float16_t y) {
    precise float af = float(acc);
    precise float p = float(x) * float(y);
    precise float s = af + p;
    precise float bv = s - af;
    precise float l = (af - (s - bv)) + (p - bv);
    float16_t r = float16_t(s);
    if (l == 0.0f) return r;
    uint u = floatBitsToUint(s);
    uint biased = (u >> 23) & 0xFFu;
    if (biased == 0xFFu || biased < 114u) return r;
    uint mant = u & 0x7FFFFFu;
    if (((mant >> 12) & 1u) == 0u) return r;
    if ((mant & 0xFFFu) != 0u) return r;
    precise float half16 = uintBitsToFloat((biased - 11u) << 23);
    return float16_t(s + (l > 0.0f ? half16 : -half16));
}
"""

COMMON = """
    uint gid = threadgroup_position_in_grid.x;
    uint t = thread_index_in_threadgroup.x;
    uint tid = gid * 512u + t;
    uint n = tid % 2560u;
    uint k0 = (tid % 10u) * 128u;
"""

# P2: exact_fma16 (the current production op) - fp16 chain, device fp16 weights
SRC_EXACT = COMMON + """
    float16_t outv = float16_t(0.0f);
    for (uint c = 0u; c < 10u; ++c) {
        float16_t bc = float16_t(0.0f);
        for (uint j = 0u; j < 128u; ++j) {
            bc = exact_fma16(bc, float16_t(float(j) * 1e-4f), w[k0 * 2560u + n]);
            ++k0;
        }
        outv = float16_t(outv + bc);
    }
    out[tid] = float(outv);
"""

# P1: hardware fp16 fma chain (wrong values, timing only)
SRC_HWFMA = COMMON + """
    float16_t outv = float16_t(0.0f);
    for (uint c = 0u; c < 10u; ++c) {
        float16_t bc = float16_t(0.0f);
        for (uint j = 0u; j < 128u; ++j) {
            bc = fma(bc, float16_t(float(j) * 1e-4f), w[k0 * 2560u + n]);
            ++k0;
        }
        outv = float16_t(outv + bc);
    }
    out[tid] = float(outv);
"""

# P3: convert+add chain only (fp16 <-> fp32 round trip per step)
SRC_CONV = COMMON + """
    float16_t outv = float16_t(0.0f);
    for (uint c = 0u; c < 1280u; ++c) {
        outv = float16_t(float(outv) + float(w[c % 65536u]));
    }
    out[tid] = float(outv);
"""

# P4: precise fp32 ascending chain (the projector/joint contract op)
SRC_F32P = COMMON + """
    precise float acc = 0.0f;
    for (uint c = 0u; c < 1280u; ++c) {
        acc = acc + float(w[c % 65536u]);
    }
    out[tid] = acc;
"""

# P5: plain fp32 chain (no precise) - is reassociation the only difference?
SRC_F32N = COMMON + """
    float acc = 0.0f;
    for (uint c = 0u; c < 1280u; ++c) {
        acc = acc + float(w[c % 65536u]);
    }
    out[tid] = acc;
"""

# P6: exact_fma16 in the uint domain - no float16_t converts inside the op;
# the running value stays a float16_t but every intermediate is fp32/uint and
# the rounding is manual bit work ending in unpackHalf2x16.
MANUAL_HDR = FMA_HDR + """
float16_t exact_fma16_manual(float16_t acc, float16_t x, float16_t y) {
    precise float af = float(acc);
    precise float p = float(x) * float(y);
    precise float s = af + p;
    precise float bv = s - af;
    precise float l = (af - (s - bv)) + (p - bv);
    if (l == 0.0f) return float16_t(s);
    uint u = floatBitsToUint(s);
    uint biased = (u >> 23) & 0xFFu;
    if (biased == 0xFFu || biased < 114u) return float16_t(s);
    uint mant = u & 0x7FFFFFu;
    if (((mant >> 12) & 1u) == 0u) return float16_t(s);
    if ((mant & 0xFFFu) != 0u) return float16_t(s);
    precise float half16 = uintBitsToFloat((biased - 11u) << 23);
    return float16_t(s + (l > 0.0f ? half16 : -half16));
}
uint fp32_bits_to_fp16_bits_round(float s) {
    uint u = floatBitsToUint(s);
    uint sign = u & 0x80000000u;
    uint biased = (u >> 23) & 0xFFu;
    if (biased >= 143u) { return (sign >> 16) | 0x7C00u; }
    if (biased < 103u) { return sign >> 16; }
    uint mant = u & 0x7FFFFFu;
    uint h;
    if (biased >= 113u) {
        uint keep = (mant >> 13) & 0x3FFu;
        uint rem = mant & 0x1FFFu;
        h = ((biased - 112u) << 10) | keep;
        if (rem > 0x1000u || (rem == 0x1000u && (keep & 1u) == 1u)) { h += 1u; }
    } else {
        uint shift = 126u - biased;
        uint keep = mant | 0x800000u;
        uint keep2 = keep >> (shift + 1u);
        uint rem = keep & ((2u << shift) - 1u);
        h = keep2;
        uint halfbit = 1u << shift;
        if (rem > halfbit || (rem == halfbit && (keep2 & 1u) == 1u)) { h += 1u; }
    }
    return (sign >> 16) | h;
}
float16_t round16(float s) {
    return unpackHalf2x16(fp32_bits_to_fp16_bits_round(s)).x;
}
"""

SRC_MANUAL = COMMON + """
    float16_t outv = float16_t(0.0f);
    for (uint c = 0u; c < 10u; ++c) {
        float16_t bc = float16_t(0.0f);
        for (uint j = 0u; j < 128u; ++j) {
            precise float af = float(bc);
            precise float p = float(float(j) * 1e-4f) * float(w[k0 * 2560u + n]);
            precise float s = af + p;
            bc = round16_manual(s);
            ++k0;
        }
        outv = float16_t(outv + bc);
    }
    out[tid] = float(outv);
"""

MANUAL2 = MANUAL_HDR + """
#define round16_manual round16
"""

# P7: fp16 chain carried via fp16 mul/add (no fma at all)
SRC_MULADD = COMMON + """
    float16_t outv = float16_t(0.0f);
    for (uint c = 0u; c < 10u; ++c) {
        float16_t bc = float16_t(0.0f);
        for (uint j = 0u; j < 128u; ++j) {
            bc = float16_t(bc + float16_t(float(j) * 1e-4f) * w[k0 * 2560u + n]);
            ++k0;
        }
        outv = float16_t(outv + bc);
    }
    out[tid] = float(outv);
"""


def main() -> int:
    import mlx.core as mx

    mx.set_default_device(mx.gpu)
    rng = np.random.default_rng(7)
    w = mx.array(rng.standard_normal((2 * 1280 * 2560)).astype(np.float16))
    mx.eval(w)

    variants = [
        ("exact_fma16_prod", SRC_EXACT, FMA_HDR),
        ("hw_fma16", SRC_HWFMA, ""),
        ("conv_add_chain", SRC_CONV, ""),
        ("fp32_precise_chain", SRC_F32P, ""),
        ("fp32_plain_chain", SRC_F32N, ""),
        ("manual_round_chain", SRC_MANUAL, MANUAL2),
        ("fp16_muladd_chain", SRC_MULADD, ""),
    ]
    out = mx.zeros((10240,), dtype=mx.float32)
    report = []
    for name, src, header in variants:
        k = mx.fast.metal_kernel(
            name=name,
            input_names=["w"],
            output_names=["out"],
            header=header,
            source=src,
            compile_options={"math_mode": "safe"},
        )

        def call(k=k):
            r = k(
                inputs=[w],
                output_shapes=[(10240,)],
                output_dtypes=[mx.float32],
                grid=(20, 1, 1),
                threadgroup=(512, 1, 1),
            )
            mx.eval(r[0])

        try:
            call()
            best = 1e9
            for _ in range(5):
                t0 = time.monotonic_ns()
                call()
                best = min(best, time.monotonic_ns() - t0)
            report.append({"variant": name, "ms": round(best / 1e6, 3)})
        except Exception as exc:  # noqa: BLE001 - probe reports and continues
            report.append({"variant": name, "error": str(exc)[:200]})
        print(json.dumps(report[-1]), flush=True)

    # exactness spot-check of the manual-round variant vs exact_fma16 on real
    # adversarial values happens in validate_step; here just confirm the two
    # production variants agree bit-for-bit on this synthetic run.
    def run(name, src, header):
        k = mx.fast.metal_kernel(
            name=name + "_chk",
            input_names=["w"],
            output_names=["out"],
            header=header,
            source=src,
            compile_options={"math_mode": "safe"},
        )
        r = k(
            inputs=[w],
            output_shapes=[(10240,)],
            output_dtypes=[mx.float32],
            grid=(20, 1, 1),
            threadgroup=(512, 1, 1),
        )
        return np.asarray(r[0])

    a = run("chk_exact", SRC_EXACT, FMA_HDR)
    b = run("chk_manual", SRC_MANUAL, MANUAL2)
    same = bool(np.array_equal(a, b))
    report.append({"variant": "manual_vs_exact_bitwise", "equal": same})
    print(json.dumps(report[-1]), flush=True)

    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
