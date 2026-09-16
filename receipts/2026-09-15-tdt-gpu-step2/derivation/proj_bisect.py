#!/usr/bin/env python3
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Bisect the proj kernel: which construct kills the shared-memory chain?

Variants (single dispatch each, synthetic h1 input):
  v1: load sh_dec from device, barrier, chain, pj            (no branch)
  v2: v1 + the jmode branch around load and chain
  v3: v1 + relu/encoder tail
  v4: barrier BEFORE load too (double barrier)
"""
import sys

import numpy as np

PKG = "/var/tmp/TdtGpuStep2/pkg"


def run_variant(mx, packed, encoder_dev, name, src):
    k = mx.fast.metal_kernel(
        name=f"probe_{name}",
        input_names=["h1_in", "projector", "encoder", "flags"],
        output_names=["pj_out", "relu_out"],
        header="",
        source=src,
        compile_options={"math_mode": "safe"},
    )
    h1 = mx.array(np.random.default_rng(5).standard_normal(640).astype(np.float32) * 0.3)
    mx.eval(h1)
    flags = mx.array(np.array([3, 0, 0], np.int32))
    pj, rl = k(
        inputs=[h1, packed.projector, encoder_dev.reshape(-1), flags],
        output_shapes=[(640,), (640,)],
        output_dtypes=[mx.float32, mx.float16],
        grid=(1, 1, 1),
        threadgroup=(640, 1, 1),
        stream=mx.gpu,
    )
    mx.eval(pj, rl)
    pjh = np.asarray(pj)
    print(f"{name}: pj finite={np.isfinite(pjh).all()} nonzero={np.count_nonzero(pjh)} "
          f"max_abs={np.abs(pjh).max():.4f}")
    return pjh


LOAD = """
    uint t = thread_index_in_threadgroup.x;
    int jmode = flags[1];
    threadgroup float16_t sh_dec[640];
"""


def main() -> int:
    import mlx.core as mx

    mx.set_default_device(mx.gpu)
    sys.path.insert(0, PKG)
    sys.path.insert(0, PKG + "/coreml")
    from coreml.vulkan_decoder import load_decoder
    from coreml.vulkan_decoder_step import pack_step_weights

    decoder = load_decoder(
        "/home/joshuawarren/.cache/mlx-omarchy/parakeet-reference/mweinbach1/"
        "parakeet-tdt-0.6b-v3-coreml/b650695c2322ee5281dff48d7345b2f3a58ff018"
        "/decoder.mlpackage"
    )
    packed = pack_step_weights(decoder, None or (
        "/home/joshuawarren/.cache/mlx-omarchy/parakeet-reference/mweinbach1/"
        "parakeet-tdt-0.6b-v3-coreml/b650695c2322ee5281dff48d7345b2f3a58ff018"
        "/joint.mlpackage"
    ))
    frames = np.random.default_rng(9).standard_normal((1, 24, 640)).astype(np.float32)
    encoder_dev = mx.array(frames)
    mx.eval(encoder_dev)

    CHAIN = """
    precise float acc = 0.0f;
    for (uint k = 0u; k < 640u; ++k) {
        acc = acc + float(sh_dec[k]) * float(projector[k * 640u + t]);
    }
    float16_t pj = float16_t(float16_t(acc) + projector[640u * 640u + t]);
    pj_out[t] = float(pj);
"""
    v1 = LOAD + """
    sh_dec[t] = float16_t(h1_in[t]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
""" + CHAIN
    v2 = LOAD + """
    if (jmode == 0) {
        sh_dec[t] = float16_t(h1_in[t]);
    } else {
        sh_dec[t] = float16_t(h1_in[t]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (jmode == 0) {
""" + CHAIN + """
    } else {
        pj_out[t] = 0.0f;
    }
"""
    v3 = LOAD + """
    sh_dec[t] = float16_t(h1_in[t]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
""" + CHAIN + """
    float16_t rlv = float16_t(encoder[uint(flags[0]) * 640u + t]) + pj;
    relu_out[t] = (rlv > float16_t(0.0f)) ? rlv : float16_t(0.0f);
"""
    v4 = LOAD + """
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sh_dec[t] = float16_t(h1_in[t]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
""" + CHAIN
    for name, src in (("v1_plain", v1), ("v2_branch", v2), ("v3_relu", v3), ("v4_doublebarrier", v4)):
        try:
            run_variant(mx, packed, encoder_dev, name, src)
        except Exception as exc:  # noqa: BLE001
            print(f"{name}: ERROR {str(exc)[:160]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
