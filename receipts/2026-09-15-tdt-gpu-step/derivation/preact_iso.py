# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# Preact isolation: same exact_fma16 chain as the fused step, layer 0 only,
# compared lane-by-lane against the numpy contract.
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

    rng = np.random.default_rng(5)
    decoder = load_decoder(MODEL / "decoder.mlpackage")
    w64, bias = decoder._layers[0]
    weights16 = np.ascontiguousarray(w64.astype(np.float16))
    token = 321
    embedding = np.asarray(decoder._weights.embedding)
    x = embedding[token].ravel()
    h0 = (rng.standard_normal(640) * 0.4).astype(np.float16)
    a = np.concatenate([x, h0]).astype(np.float16)

    preact_np = vd._add16(vd._blocked_gemv(a, w64), bias.astype(np.float16))
    print("numpy preact[0:4]", preact_np[:4])
    print("numpy preact[638:642]", preact_np[638:642])

    header = open("header.txt").read()
    kernel = mx.fast.metal_kernel(
        name="preact_iso",
        input_names=["a", "w", "bias"],
        output_names=["op"],
        header=header,
        source="""
            uint n = thread_position_in_grid.x;
            float16_t bacc = float16_t(0.0f);
            float16_t bc = float16_t(0.0f);
            for (uint k = 0u; k < 1280u; ++k) {
                bc = exact_fma16(bc, a[k], w[k * 2560u + n]);
                if ((k & 127u) == 127u) {
                    bacc = ((k == 127u) ? bc : float16_t(bacc + bc));
                    bc = float16_t(0.0f);
                }
            }
            op[n] = bacc + bias[n];
        """,
        compile_options={"math_mode": "safe"},
    )
    out = kernel(
        inputs=[mx.array(a), mx.array(weights16), mx.array(bias.astype(np.float16))],
        output_shapes=[(2560,)],
        output_dtypes=[mx.float16],
        grid=(2560, 1, 1),
        threadgroup=(256, 1, 1),
        stream=mx.gpu,
    )
    mx.eval(out)
    got = np.asarray(out[0])
    bad = np.flatnonzero(got != preact_np)
    print(f"preact mismatches {len(bad)}/2560")
    for i in bad[:5]:
        print(f"  lane {i}: got {got[i].view(np.uint16)[()]:#06x} want {preact_np[i].view(np.uint16)[()]:#06x}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
