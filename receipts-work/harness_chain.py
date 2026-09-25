#!/usr/bin/env python3
"""Chained-reality harness: the in-model structure the per-class micro lacked.

Per token (T tokens, per-token replanning like decode):
  xin (residual; step 0 constant, steps 1+ the previous token's final
      epilogue-sum buffer)
  n1  = fast RMSNorm(xin)                       <- deferred norm
  trio: qmm(n1, A1..A3)                          <- GDN/attn trio class
  single: d = qmm(n1, Wd); h = d + addend(xin)   <- epilogue sum = next norm input
  n2  = fast RMSNorm(h)                          <- norm fed by ANOTHER group's sum
  trio2: qmm(n2, B1..B3)
  out = h + qmm(n2, Wd2) + h                     <- chained residual epilogue
  state[:, :64] slice-updated from the trio output (in-place persistent write)

Toggles via env:
  NH_NOCHAIN=1 : norm inputs are materialized constants (kills the chain, L-chain)
  NH_ONETOK=1  : single token (kills replanning, L2)
  NH_NOSLICE=1 : no persistent slice-update state (kills L3 in-place flavor)
Dumps per-step sha256 of every observable. Compare -on vs -off.
"""
import hashlib
import json
import os
import sys

import mlx.core as mx
import numpy as np

K = 2048
T = 8 if os.environ.get("NH_ONETOK") != "1" else 1
seed = 0
out_path = sys.argv[1] if len(sys.argv) > 1 else "/var/tmp/normfold-out/chain.json"

rng = np.random.default_rng(seed)

def bf(a):
    a = a.astype(np.float32)
    return (a.view(np.uint32) & np.uint32(0xFFFF0000)).view(np.float32)

def qpack(m):
    n, k = m.shape
    g = m.astype(np.float32).reshape(n, k // 64, 64)
    s = np.maximum(np.abs(g).max(axis=2) / 127.0, 1e-8)
    q = np.clip(np.abs(g / s[:, :, None]).round(), 0, 127).astype(np.uint32)
    scales = mx.array(np.ascontiguousarray(s, dtype=np.float32)).astype(mx.bfloat16)
    qw = q.reshape(n, k // 8, 8)
    packed = (qw[..., 0] | (qw[..., 1] << 4) | (qw[..., 2] << 8) | (qw[..., 3] << 12) |
              (qw[..., 4] << 16) | (qw[..., 5] << 20) | (qw[..., 6] << 24) | (qw[..., 7] << 28))
    return mx.array(np.ascontiguousarray(packed.astype(np.uint32))), scales, scales * 0

def mat(rows):
    return qpack(rng.standard_normal((rows, K)) * 0.05)

w = mx.array(bf(rng.standard_normal(K) * 0.5 + 1.0)).astype(mx.bfloat16)
mx.eval(w)

W_A1, W_A2, W_A3 = mat(512), mat(256), mat(192)
W_AD = mat(2048)  # single + epilogue: rows = hidden so h = d + xin broadcasts
W_B1, W_B2, W_B3 = mat(384), mat(320), mat(256)
W_BD = mat(2048)  # epilogue producing the next residual
mx.eval(W_A1[1], W_A2[1], W_A3[1], W_AD[1],
        W_B1[1], W_B2[1], W_B3[1], W_BD[1])
mx.eval(W_A1[0], W_A2[0], W_A3[0], W_AD[0],
        W_B1[0], W_B2[0], W_B3[0], W_BD[0])

def qmm(t, p):
    return mx.quantized_matmul(t, p[0], p[1], p[2], transpose=True)

def h(a):
    return hashlib.sha256(np.asarray(a.astype(mx.float32)).tobytes()).hexdigest()[:16]

x0 = mx.array(bf(rng.standard_normal(K) * 2.0)).reshape(1, K).astype(mx.bfloat16)
mx.eval(x0)

NOCHAIN = os.environ.get("NH_NOCHAIN") == "1"
NOSLICE = os.environ.get("NH_NOSLICE") == "1"

state = mx.zeros((1, 64), dtype=mx.bfloat16)
mx.eval(state)

steps = []
xin = x0
for t in range(T):
    if not NOCHAIN and t > 0:
        pass  # xin is the previous token's epilogue-sum array (already set)
    n1 = mx.fast.rms_norm(xin, w, 1e-6)
    a1, a2, a3 = qmm(n1, W_A1), qmm(n1, W_A2), qmm(n1, W_A3)
    # Single-member group on its OWN norm so n1 keeps exactly three qmm
    # consumers (uses==3): a fourth qmm on n1 chunks [3,1] and the
    # reader-safety matcher refuses both chunks, exactly like the GDN
    # in_proj in-model.
    ng = mx.fast.rms_norm(xin, w, 1e-6)
    d = qmm(ng, W_AD)
    if NOCHAIN:
        din = x0  # constant input to the epilogue add (kills the chained buffer)
    else:
        din = xin
    hs = d + din
    n2 = mx.fast.rms_norm(hs, w, 1e-6)
    b1, b2, b3 = qmm(n2, W_B1), qmm(n2, W_B2), qmm(n2, W_B3)
    d2 = qmm(n2, W_BD)
    out = hs + d2 + hs
    if not NOSLICE:
        state = mx.slice_update(state, b3[:, :64], mx.array([0, 0]), [0, 1])
    mx.eval(a1, a2, a3, d, h, n2 if False else b1, b2, b3, d2, out, state)
    steps.append({
        "a1": h(a1), "a2": h(a2), "a3": h(a3), "d": h(d), "hs": h(hs),
        "b1": h(b1), "b2": h(b2), "b3": h(b3), "d2": h(d2), "out": h(out),
        "state": h(state),
    })
    xin = out  # chain: next token's norm input is this token's epilogue sum

rec = {"env": os.environ.get("MLX_OMARCHY_FUSED_GEMV_NORM", "default"),
       "nochain": NOCHAIN, "onetok": T == 1, "noslice": NOSLICE,
       "steps": steps}
print(json.dumps(rec))
with open(out_path, "w") as f:
    json.dump(rec, f)
