#!/usr/bin/env python3
"""Bit-identity probe for the Bf16PrefillGap candidate.

Runs the same fixed-input cases through whatever wheel this interpreter
has and prints one NDJSON row per case: {"case", "sha256"} over the raw
output bytes. The window script runs it under the base and candidate
venvs and diffs. Cases cover the three candidate mechanisms:

  sdpa_*   bf16 f32-composition sdpa, causal (the additive-mask ->
           softmax-causal-mode change), including offset>0, the ragged
           k_len<q_len fallback, and the decode shape (q_len=1)
  mm_bf16  dense bf16 matmul (pair-load staging): aligned production
           shape plus a misaligned slice that must keep the scalar path
  mm_f32   dense f32 matmul with k%8 tail (the coopmat k-tail) vs base's
           16x16 tile route, plus k%8==0 and k<8 controls
"""
import hashlib
import json
import os
import sys

import numpy as np

os.environ.setdefault("MLX_DISABLE_COMPILE", "1")

import mlx.core as mx  # noqa: E402

def sha(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        mx.eval(a)
        b = a
        if b.dtype in (mx.bfloat16, mx.float16):
            b = b.view(mx.uint16)
        h.update(np.asarray(b).tobytes())
    return h.hexdigest()


def main():
    out = []
    rng = np.random.default_rng(1234)

    def bf16(shape):
        return mx.array(rng.standard_normal(shape).astype(np.float32)).astype(
            mx.bfloat16)

    def f32(shape):
        return mx.array(rng.standard_normal(shape).astype(np.float32))

    scale = 0.125
    # ---- sdpa causal cases (bf16 -> f32 composition in the fork) ----
    for tag, q_len, k_len in (
            ("sq257", 257, 257), ("s1k", 1052, 1052),
            ("off", 64, 1052), ("ragged", 64, 32), ("dec", 1, 1053)):
        q = bf16((1, 14, q_len, 64))
        k = bf16((1, 14, k_len, 64))
        v = bf16((1, 14, k_len, 64))
        y = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=scale, mask="causal")
        out.append({"case": f"sdpa_{tag}", "sha256": sha(y)})

    # ---- dense bf16 matmul (pair staging) ----
    x = bf16((1052, 896))
    w = bf16((896, 896))
    out.append({"case": "mm_bf16_aligned", "sha256": sha(x @ w.T)})
    xs = x[:, 2:]  # element offset 2 -> k%8!=0 -> plain tile on both
    out.append({"case": "mm_bf16_off2", "sha256": sha(xs @ w[:, :894].T)})
    ws = w[:, 2:]  # weight k-slice -> scalar path on both wheels
    out.append({"case": "mm_bf16_gap894", "sha256": sha(x[:, :894] @ ws.T)})
    wv = bf16((4864, 896))
    out.append({"case": "mm_bf16_gate_up", "sha256": sha(x @ wv.T)})

    # ---- dense f32 matmul (k tail) ----
    p = mx.abs(f32((14, 1052, 1052))) * 0.001
    vv = f32((14, 1052, 64))
    out.append({"case": "mm_f32_ktail4", "sha256": sha(
        mx.matmul(p, vv))})
    p0 = f32((14, 64, 64))
    v0 = f32((14, 64, 128))
    out.append({"case": "mm_f32_kmod0", "sha256": sha(mx.matmul(p0, v0))})
    p7 = f32((14, 33, 7))
    v7 = f32((14, 7, 64))
    out.append({"case": "mm_f32_ktail7", "sha256": sha(mx.matmul(p7, v7))})

    tag = sys.argv[1] if len(sys.argv) > 1 else "wheel"
    for row in out:
        row["wheel"] = tag
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
