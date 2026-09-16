#!/usr/bin/env python3
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Scaling probe: exact_fma16 chain wall time vs (threadgroups x threadgroup size).

Holds TOTAL work at one layer GEMV (2560 lanes x 40 chains x 128 k) and sweeps
geometry. Also probes an fp32 ascending chain (the projector/joint contract
shape: 8198 output lanes x 640 k) and the per-dispatch sync floor.
"""
import json
import sys
import time

import numpy as np

EXACT_FMA16 = """
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

LSTM_SRC = """
    uint gid = threadgroup_position_in_grid.x;
    uint t = thread_index_in_threadgroup.x;
    uint tid = gid * THREADS + t;
    uint chains = CHAINS_PER_THREAD;
    uint n = tid % 2560u;
    float16_t outv = float16_t(0.0f);
    for (uint c = 0u; c < chains; ++c) {
        float16_t bc = float16_t(0.0f);
        uint k0 = ((tid + c) % 10u) * 128u;
        for (uint j = 0u; j < 128u; ++j) {
            bc = exact_fma16(bc, float16_t(float(j) * 1e-4f), weights[k0 * 2560u + n]);
            ++k0;
        }
        outv = float16_t(outv + bc);
    }
    out[tid] = float(outv);
"""

FP32_SRC = """
    uint gid = threadgroup_position_in_grid.x;
    uint t = thread_index_in_threadgroup.x;
    uint tid = gid * THREADS + t;
    uint lanes = LANES_PER_THREAD;
    for (uint c = 0u; c < lanes; ++c) {
        uint j = tid + c * TOTAL_T;
        if (j >= OUT_N) { continue; }
        precise float acc = 0.0f;
        for (uint k = 0u; k < 640u; ++k) {
            acc = acc + float(joint[k * OUT_N + j]);
        }
        out[j] = acc;
    }
"""

NULL_SRC = """
    uint gid = threadgroup_position_in_grid.x;
    uint t = thread_index_in_threadgroup.x;
    if (gid == 4294967295u && t == 4294967295u) { out[0] = 1.0f; }
"""


def main() -> int:
    import mlx.core as mx

    mx.set_default_device(mx.gpu)
    rng = np.random.default_rng(7)
    weights = mx.array(rng.standard_normal((2, 1280, 2560)).astype(np.float16).ravel())
    joint = mx.array(rng.standard_normal((640 * 8198)).astype(np.float16).ravel())
    mx.eval(weights, joint)

    results = []

    def dispatch(k, args, n_out, grid_tg, tsize):
        state = k(
            inputs=args,
            output_shapes=[(n_out,)],
            output_dtypes=[mx.float32],
            grid=(grid_tg, 1, 1),
            threadgroup=(tsize, 1, 1),
        )
        return state[0]

    def timed(call, reps=5):
        call()
        best = 1e9
        for _ in range(reps):
            t0 = time.monotonic_ns()
            call()
            best = min(best, time.monotonic_ns() - t0)
        return best / 1e6

    # ---- dispatch + sync floor (nop kernel) ----
    kn = mx.fast.metal_kernel(
        name="probe_null",
        input_names=["weights", "joint"],
        output_names=["out"],
        header="",
        source=NULL_SRC,
        compile_options={"math_mode": "safe"},
    )
    for grid_tg, tsize in ((1, 256), (64, 256)):
        wall = timed(lambda g=grid_tg, t=tsize: mx.eval(
            dispatch(kn, [weights, joint], 16, g, t)))
        results.append({"probe": "null_sync", "grid": grid_tg, "ms": round(wall, 4)})

    # ---- exact_fma16 layer-GEMV geometry sweep ----
    TOTAL_CHAINS = 2560 * 40  # one layer
    for grid_tg, tsize in (
        (1, 640), (4, 640), (10, 512), (20, 512), (40, 320),
        (64, 200), (80, 160), (40, 640), (40, 160), (80, 80),
    ):
        total_t = grid_tg * tsize
        per_thread = TOTAL_CHAINS // total_t
        if per_thread * total_t != TOTAL_CHAINS or per_thread == 0:
            continue
        src = LSTM_SRC.replace("THREADS", str(tsize)).replace(
            "CHAINS_PER_THREAD", str(per_thread)
        )
        k = mx.fast.metal_kernel(
            name=f"probe_lstm_{grid_tg}x{tsize}",
            input_names=["weights", "joint"],
            output_names=["out"],
            header=EXACT_FMA16,
            source=src,
            compile_options={"math_mode": "safe"},
        )
        wall = timed(lambda k=k, g=grid_tg, t=tsize, n=total_t: mx.eval(
            dispatch(k, [weights, joint], n, g, t)))
        print(json.dumps(results[-1:]), flush=True)
        results.append({
            "probe": "fma16_layer", "grid": grid_tg, "threads": tsize,
            "total_threads": total_t, "chains_per_thread": per_thread,
            "ms": round(wall, 3),
        })

    # ---- fp32 ascending chain (joint contract shape) ----
    OUT_N = 8198
    jm_host = np.asarray(joint, dtype=np.float16).astype(np.float32).reshape(640, OUT_N)
    for grid_tg, tsize in ((1, 640), (2, 1024), (17, 512), (34, 256), (66, 128)):
        total_t = grid_tg * tsize
        lanes_pt = -(-OUT_N // total_t)
        src = (FP32_SRC.replace("TOTAL_T", str(total_t))
               .replace("LANES_PER_THREAD", str(lanes_pt))
               .replace("OUT_N", str(OUT_N)))
        k = mx.fast.metal_kernel(
            name=f"probe_fp32_{grid_tg}x{tsize}",
            input_names=["weights", "joint"],
            output_names=["out"],
            header="",
            source=src,
            compile_options={"math_mode": "safe"},
        )
        wall = timed(lambda k=k, g=grid_tg, t=tsize: mx.eval(
            dispatch(k, [weights, joint], OUT_N, g, t)))
        # exactness sanity on the first 64 lanes: fp32 ascending numpy fp32 order
        ref = np.zeros((64,), dtype=np.float32)
        for j in range(64):
            acc = np.float32(0.0)
            for kk in range(640):
                acc = np.float32(acc + jm_host[kk, j])
            ref[j] = acc
        got = np.empty((64,), dtype=np.float32)
        state = k(
            inputs=[weights, joint],
            output_shapes=[(OUT_N,)],
            output_dtypes=[mx.float32],
            grid=(grid_tg, 1, 1),
            threadgroup=(tsize, 1, 1),
        )
        got[:] = np.asarray(state[0])[:64]
        exact = bool(np.array_equal(got, ref))
        print(json.dumps(results[-1:]), flush=True)
        results.append({
            "probe": "fp32_joint", "grid": grid_tg, "threads": tsize,
            "total_threads": total_t, "ms": round(wall, 3), "exact_first64": exact,
        })

    print(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
