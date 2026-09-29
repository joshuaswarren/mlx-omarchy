#!/usr/bin/env python3
"""Jw16DecodeGap2 group-level bit check for the RMSNorm-prologue fold.

Run the SAME script under two venvs (base wheel = unfused, candidate wheel
with the np fold) and diff the JSON: identical hashes == bit-identical
outputs on identical numpy-seeded inputs. Covers the group classes of the
Qwen3.8-2B decode tape:

  q4x4_add.*   4-member group, Add epilogues on every member (in_proj
               class with residual addends)
  swiglu.*     2-member group folded into the SwiGLU store epilogue, with
               the norm feeding it (gate/up class)
  noNorm.*     2-member group whose x is NOT a norm output (out/down
               control - must be untouched by the fold)
  normOnly.*   the standalone fast rms_norm itself (control)

The kv-direct sum-window interaction is exercised by the model-level
digests in the window cells (this script's hashes cannot replace them).
usage: np_bitcheck.py OUT.json
"""
import hashlib
import json
import sys

import mlx.core as mx
import numpy as np

out_path = sys.argv[1]
rows = {}


def h(*arrs):
    m = hashlib.sha256()
    for a in arrs:
        mx.eval(a)
        if a.dtype in (mx.float32, mx.int32, mx.uint32):
            v = np.array(a.view(mx.uint32))
        elif a.dtype in (mx.bfloat16, mx.float16, mx.int16, mx.uint16):
            v = np.array(a.view(mx.uint16))
        else:
            v = np.array(a)
        m.update(v.tobytes())
    return m.hexdigest()[:16]


def seeded(shape, seed, dtype=mx.bfloat16, scale=1.0):
    rng = np.random.default_rng(seed)
    a = (rng.standard_normal(shape) * scale).astype(np.float32)
    return mx.array(a).astype(dtype)


def quant(seed, n, k):
    w = seeded((n, k), seed, mx.bfloat16, 0.05)
    q, s, b = mx.quantize(w, group_size=64, bits=4)
    return q, s, b


K = 2048
for seed in (11, 12, 13):
    # Pre-norm row, norm weight, residual addends: identical bytes both sides.
    h_in = seeded((1, K), seed, mx.bfloat16)
    nw = seeded((K,), seed + 100, mx.bfloat16, 0.5)
    addends = [seeded((1, n), seed + 200 + i, mx.bfloat16)
               for i, n in enumerate((2048, 2048, 2048, 1024))]

    # Class q4x4_add: four qmms on one normed row, each with an Add.
    xn = mx.fast.rms_norm(h_in, nw, 1e-6)
    outs = []
    for i, n in enumerate((2048, 2048, 2048, 1024)):
        wq, s, b = quant(seed + 300 + i, n, K)
        y = mx.quantized_matmul(
            xn, wq, s, b, transpose=True, group_size=64, bits=4)
        outs.append(y + addends[i])
    rows[f"q4x4_add.s{seed}"] = h(*outs)
    rows[f"q4x4_add.xn.s{seed}"] = h(xn)

    # Class swiglu: gate/up pair, the tail silu(gate)*up, norm feeding.
    xn2 = mx.fast.rms_norm(h_in, nw, 1e-6)
    wg = quant(seed + 500, 6144, K)
    wu = quant(seed + 501, 6144, K)
    gate = mx.quantized_matmul(
        xn2, *wg, transpose=True, group_size=64, bits=4)
    up = mx.quantized_matmul(xn2, *wu, transpose=True, group_size=64, bits=4)
    prod = mx.silu(gate) * up
    rows[f"swiglu.s{seed}"] = h(prod, gate + seeded((1, 6144), seed + 7))

    # Class noNorm: same shapes, x is the raw row (no RMSNorm producer).
    outs2 = []
    for i, n in enumerate((2048, 6144)):
        wq, s, b = quant(seed + 600 + i, n, K)
        y = mx.quantized_matmul(
            h_in, wq, s, b, transpose=True, group_size=64, bits=4)
        outs2.append(y + addends[i])
    rows[f"noNorm.s{seed}"] = h(*outs2)

    # Standalone norm control across dtypes.
    rows[f"normOnly.bf16.s{seed}"] = h(
        mx.fast.rms_norm(h_in, nw, 1e-6))
    rows[f"normOnly.f32.s{seed}"] = h(
        mx.fast.rms_norm(h_in.astype(mx.float32),
                         nw.astype(mx.float32), 1e-6))

with open(out_path, "w") as f:
    json.dump(rows, f, indent=1, sort_keys=True)
print(f"wrote {out_path}: {len(rows)} rows")
