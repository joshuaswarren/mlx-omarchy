#!/usr/bin/env python3
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Interleaving probe: N independent exact_fma16 block-chains stepped in a
software-pipelined loop body vs sequentially. One process per config.
argv: G T INTERLEAVE  (INTERLEAVE in {1,2,4,5,8})
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
    nchain = int(sys.argv[3])
    import mlx.core as mx

    mx.set_default_device(mx.gpu)
    rng = np.random.default_rng(7)
    w = mx.array(rng.standard_normal((2 * 1280 * 2560)).astype(np.float16))
    mx.eval(w)

    total_t = grid_tg * tsize
    chains = nchain  # per thread
    decl = "\n".join(
        f"    float16_t bc{c} = float16_t(0.0f);" for c in range(chains)
    )
    decl += "\n    float16_t outv = float16_t(0.0f);"
    body = "\n".join(
        f"        bc{c} = exact_fma16(bc{c}, float16_t(float(j) * 1e-4f), w[k{c} * 2560u + n]);"
        for c in range(chains)
    )
    incs = "\n".join(f"        ++k{c};" for c in range(chains))
    fold = "\n".join(f"    outv = float16_t(outv + bc{c});" for c in range(chains))
    kdecl = "\n".join(
        f"    uint k{c} = ((tid + {c} * {total_t}u) % 256000u) & ~127u;"
        for c in range(chains)
    )
    src = f"""
    uint gid = threadgroup_position_in_grid.x;
    uint t = thread_index_in_threadgroup.x;
    uint tid = gid * {tsize}u + t;
    uint n = tid % 2560u;
{kdecl}
{decl}
    for (uint j = 0u; j < 128u; ++j) {{
{body}
{incs}
    }}
{fold}
    out[tid] = float(outv);
"""
    k = mx.fast.metal_kernel(
        name="probe_interleave",
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
        "mode": "interleave", "grid": grid_tg, "threads": tsize,
        "total_threads": total_t, "chains_per_thread": chains,
        "ms": round(best / 1e6, 3),
    }), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
