# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# Isolation 2: the exact fused-kernel layer-0 GEMV (shared a, 4-lane loop,
# barriers) as a standalone kernel, dumped through logits.
import sys
from pathlib import Path

import numpy as np

PKG = Path("/var/tmp/TdtGpuStep/pkg")
MODEL = Path(
    "/home/joshuawarren/.cache/mlx-omarchy/parakeet-reference/mweinbach1/"
    "parakeet-tdt-0.6b-v3-coreml/b650695c2322ee5281dff48d7345b2f3a58ff018"
)


def main() -> int:
    import mlx.core as mx

    sys.path.insert(0, str(PKG.resolve()))
    sys.path.insert(0, str((PKG / "coreml").resolve()))
    import coreml.vulkan_decoder as vd
    from coreml.vulkan_decoder import load_decoder

    decoder = load_decoder(MODEL / "decoder.mlpackage")
    w64, bias = decoder._layers[0]
    weights16 = np.ascontiguousarray(w64.astype(np.float16))
    token = 122
    embedding = np.asarray(decoder._weights.embedding)
    x = embedding[token].ravel()
    h0 = np.zeros(640, np.float16)
    a = np.concatenate([x, h0]).astype(np.float16)
    want = vd._add16(vd._blocked_gemv(a, w64), bias.astype(np.float16))

    header = open("header.txt").read()
    kernel = mx.fast.metal_kernel(
        name="layer0_iso",
        input_names=["a_in", "weights", "biases", "hidden_in"],
        output_names=["state_out", "logits"],
        header=header,
        source="""
            uint t = thread_index_in_threadgroup.x;
            int token_id = 122;
            threadgroup float16_t sh_a[1280];
            uint tbase = uint(token_id);
            sh_a[t] = a_in[tbase * 640u + t];
            sh_a[640u + t] = float16_t(hidden_in[t]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float16_t pr[4];
            for (uint q = 0u; q < 4u; ++q) {
                uint n = t + q * 640u;
                float16_t bacc = float16_t(0.0f);
                float16_t bc = float16_t(0.0f);
                for (uint k = 0u; k < 1280u; ++k) {
                    bc = exact_fma16(bc, sh_a[k], weights[k * 2560u + n]);
                    if ((k & 127u) == 127u) {
                        bacc = ((k == 127u) ? bc : float16_t(bacc + bc));
                        bc = float16_t(0.0f);
                    }
                }
                pr[q] = bacc + biases[n];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint q = 0u; q < 4u; ++q) {
                logits[t + q * 640u] = float(pr[q]);
            }
            state_out[t] = float(pr[0]);
        """,
        compile_options={"math_mode": "safe"},
    )
    out = kernel(
        inputs=[
            mx.array(embedding), mx.array(weights16),
            mx.array(bias.astype(np.float16)), mx.array(np.zeros(640, np.float32)),
        ],
        output_shapes=[(640,), (8198,)],
        output_dtypes=[mx.float32, mx.float32],
        grid=(640, 1, 1),
        threadgroup=(640, 1, 1),
        stream=mx.gpu,
    )
    mx.eval(*out)
    got = np.asarray(out[1][:2560]).astype(np.float16)
    bads = np.flatnonzero(got != want)
    print(f"layer0 preact: {len(bads)}/2560 mismatch")
    for i in bads[:6]:
        print(f"  lane {i}: got {got[i].view(np.uint16)[()]:#06x} want {want[i].view(np.uint16)[()]:#06x}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
