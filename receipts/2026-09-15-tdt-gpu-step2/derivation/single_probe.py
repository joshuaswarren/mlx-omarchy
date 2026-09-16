#!/usr/bin/env python3
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Single-config probe: exact_fma16 layer-GEMV wall time at one geometry.

One process = one geometry, so the fork's multi-kernel shader caching can
never mix sources. argv: G T [mode]. mode=lstm sweeps the layer GEMV chain,
mode=sync sweeps N null dispatches with one final eval.
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


def main() -> int:
    grid_tg = int(sys.argv[1])
    tsize = int(sys.argv[2])
    mode = sys.argv[3] if len(sys.argv) > 3 else "lstm"
    import mlx.core as mx

    mx.set_default_device(mx.gpu)
    rng = np.random.default_rng(7)
    w = mx.array(rng.standard_normal((2 * 1280 * 2560)).astype(np.float16))
    mx.eval(w)

    if mode == "sync":
        src = """
    uint gid = threadgroup_position_in_grid.x;
    uint t = thread_index_in_threadgroup.x;
    if (gid == 4294967295u && t == 4294967295u) { out[0] = 1.0f; }
"""
        k = mx.fast.metal_kernel(
            name="probe_null_single",
            input_names=["w"],
            output_names=["out"],
            header="",
            source=src,
            compile_options={"math_mode": "safe"},
        )
        for ndisp in (1, 2, 4, 8, 16):
            outs = []
            call = lambda: [mx.eval(r) for r in (
                k(inputs=[w], output_shapes=[(16,)], output_dtypes=[mx.float32],
                  grid=(grid_tg, 1, 1), threadgroup=(tsize, 1, 1))
                for _ in range(ndisp)
            )]
            call()
            best = 1e9
            for _ in range(5):
                t0 = time.monotonic_ns()
                call()
                best = min(best, time.monotonic_ns() - t0)
            print(json.dumps({
                "mode": "sync", "grid": grid_tg, "threads": tsize,
                "dispatches": ndisp, "ms": round(best / 1e6, 3),
                "ms_per_dispatch": round(best / 1e6 / ndisp, 4),
            }), flush=True)
        return 0

    total_chains = 2560 * 40
    total_t = grid_tg * tsize
    chains = total_chains // total_t
    assert chains * total_t == total_chains, "geometry must divide evenly"
    src = f"""
    uint gid = threadgroup_position_in_grid.x;
    uint t = thread_index_in_threadgroup.x;
    uint tid = gid * {tsize}u + t;
    uint n = tid % 2560u;
    float16_t outv = float16_t(0.0f);
    for (uint c = 0u; c < {chains}u; ++c) {{
        float16_t bc = float16_t(0.0f);
        uint k0 = ((tid + c * {tsize}u) % 25600u) & ~127u;
        for (uint j = 0u; j < 128u; ++j) {{
            bc = exact_fma16(bc, float16_t(float(j) * 1e-4f), w[k0 * 2560u + n]);
            ++k0;
        }}
        outv = float16_t(outv + bc);
    }}
    out[tid] = float(outv);
"""
    k = mx.fast.metal_kernel(
        name="probe_lstm_single",
        input_names=["w"],
        output_names=["out"],
        header=FMA_HDR,
        source=src,
        compile_options={"math_mode": "safe"},
    )

    def call():
        r = k(
            inputs=[w],
            output_shapes=[(total_t,)],
            output_dtypes=[mx.float32],
            grid=(grid_tg, 1, 1),
            threadgroup=(tsize, 1, 1),
        )
        mx.eval(r[0])

    call()
    best = 1e9
    for _ in range(5):
        t0 = time.monotonic_ns()
        call()
        best = min(best, time.monotonic_ns() - t0)
    print(json.dumps({
        "mode": "lstm", "grid": grid_tg, "threads": tsize,
        "total_threads": total_t, "chains_per_thread": chains,
        "ms": round(best / 1e6, 3),
    }), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
