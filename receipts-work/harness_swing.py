#!/usr/bin/env python3
"""Commissioned round: SwiGLU-store groups (bit 16) carrying a chained norm
prologue + the RMSNormGated-fed out_proj path.

Per token (T=8, per-token replanning):
  xin  residual (chained from the previous token's epilogue sum)
  na   = RMSNorm(xin);  a1,a2,a3 = qmm trio          (uses==3, folds)
  nb   = RMSNorm(xin);  g,u = qmm(nb); sw = silu(g)*u (uses==2, folds WITH
        the swiglu store epilogue: bit 15 + bit 16 + chained input)
  dn   = qmm(sw, W_DN); hs = dn + xin                (down group; epilogue sum)
  gn   = RMSNormGated(feat, z) if the binding exists else RMSNorm(feat)
  og   = qmm(gn, W_OG) + hs                          (out_proj class)
  n2   = RMSNorm(hs);  b1,b2,b3 = qmm trio
  out  = hs + qmm(n2, W_BD) + hs                     (next residual)
  state[:, :64] slice-updated from b3

Toggles: NH_NOCHAIN=1 (na/nb read materialized constants), NH_NOGATED=1
(drop the gated stage). Compare -on vs -off per step.
"""
import hashlib
import json
import os
import sys

import mlx.core as mx
import numpy as np

K = 2048
T = 8
seed = 0
out_path = sys.argv[1] if len(sys.argv) > 1 else "/var/tmp/normfold-out/swing.json"

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

def matk(rows, k):
    return qpack(rng.standard_normal((rows, k)) * 0.05)

w = mx.array(bf(rng.standard_normal(K) * 0.5 + 1.0)).astype(mx.bfloat16)
mx.eval(w)

W_A1, W_A2, W_A3 = mat(512), mat(256), mat(192)
W_G, W_U = mat(512), mat(512)
W_DN = matk(2048, 512)
W_OG = matk(2048, 128)
W_B1, W_B2, W_B3 = mat(384), mat(320), mat(256)
W_BD = mat(2048)
mx.eval(W_A1[0], W_A1[1], W_A2[0], W_A2[1], W_A3[0], W_A3[1],
        W_G[0], W_G[1], W_U[0], W_U[1], W_DN[0], W_DN[1],
        W_OG[0], W_OG[1], W_B1[0], W_B1[1], W_B2[0], W_B2[1],
        W_B3[0], W_B3[1], W_BD[0], W_BD[1])

def qmm(t, p):
    return mx.quantized_matmul(t, p[0], p[1], p[2], transpose=True)

def h(a):
    return hashlib.sha256(np.asarray(a.astype(mx.float32)).tobytes()).hexdigest()[:16]

x0 = mx.array(bf(rng.standard_normal(K) * 2.0)).reshape(1, K).astype(mx.bfloat16)
feat0 = mx.array(bf(rng.standard_normal(128) * 1.5)).reshape(1, 128).astype(mx.bfloat16)
z0 = mx.array(bf(rng.standard_normal(128))).reshape(1, 128).astype(mx.bfloat16)
wg128 = mx.array(bf(np.ones(128))).astype(mx.bfloat16)
mx.eval(x0, feat0, z0, wg128)

NOCHAIN = os.environ.get("NH_NOCHAIN") == "1"
NOGATED = os.environ.get("NH_NOGATED") == "1"
HAS_GATED = hasattr(mx.fast, "rms_norm_gated")

state = mx.zeros((1, 64), dtype=mx.bfloat16)
mx.eval(state)

steps = []
xin = x0
feat = feat0
for t in range(T):
    if NOCHAIN:
        na_in = x0
        nb_in = x0
    else:
        na_in = xin
        nb_in = xin
    na = mx.fast.rms_norm(na_in, w, 1e-6)
    a1, a2, a3 = qmm(na, W_A1), qmm(na, W_A2), qmm(na, W_A3)
    nb = mx.fast.rms_norm(nb_in, w, 1e-6)
    g = qmm(nb, W_G)
    u = qmm(nb, W_U)
    sw = (mx.sigmoid(g) * g) * u
    dn = qmm(sw, W_DN)
    hs = dn + xin
    if not NOGATED:
        try:
            gn_ = mx.fast.rms_norm_gated(feat, z0, wg128, 1e-6)
        except Exception:
            gn_ = mx.fast.rms_norm(feat, z0, 1e-6)
    else:
        gn_ = None
    og = qmm(gn_, W_OG) + hs if gn_ is not None else hs
    n2 = mx.fast.rms_norm(hs, w, 1e-6)
    b1, b2, b3 = qmm(n2, W_B1), qmm(n2, W_B2), qmm(n2, W_B3)
    d2 = qmm(n2, W_BD)
    out = hs + d2 + hs
    state = mx.slice_update(state, b3[:, :64], mx.array([0, 0]), [0, 1])
    mx.eval(a1, a2, a3, g, u, sw, dn, hs, og if gn_ is not None else b1,
            b1, b2, b3, d2, out, state)
    rec = {
        "a1": h(a1), "a2": h(a2), "a3": h(a3),
        "g": h(g), "u": h(u), "sw": h(sw), "dn": h(dn), "hs": h(hs),
        "b1": h(b1), "b2": h(b2), "b3": h(b3), "d2": h(d2), "out": h(out),
        "state": h(state),
    }
    if gn_ is not None:
        rec["og"] = h(og)
        rec["gn"] = h(gn_)
    steps.append(rec)
    xin = out
    feat = b3[:, :128]

rec = {"env": os.environ.get("MLX_OMARCHY_FUSED_GEMV_NORM", "default"),
       "nochain": NOCHAIN, "nogated": NOGATED, "has_gated": HAS_GATED,
       "steps": steps}
print(json.dumps(rec))
with open(out_path, "w") as f:
    json.dump(rec, f)
