"""GDN synthetic v2: rep hypothesis + sane scales + first-divergence token.

rep in {3 (27B), 2 (9B), 1}: Hk=16, Hv=16*rep, Dk=Dv=128, scalar decay.
Sane input scales (the recurrence must stay f32-stable): k,v ~ N(0,0.25),
beta in [0.1,0.5], g decay in [0.85,0.99].
Primary route: NO_COOPMAT=1 (scan-only) to isolate from the coopmat kernels.
Reports the FIRST TOKEN where maskless and all-True-mask outputs diverge,
plus fp64 reference and the composed ops (gated_delta_ops) fp32 oracle.
"""
import os, sys
import numpy as np

HK, DK, DV = 16, 128, 128
rng = np.random.default_rng(20261004)

def bf16_round(a):
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    return (u >> 16 << 16).view(np.float32)

def make_inputs(T, rep):
    q = bf16_round(rng.standard_normal((1, T, HK, DK)) * 0.25)
    k = bf16_round(rng.standard_normal((1, T, HK, DK)) * 0.25)
    v = bf16_round(rng.standard_normal((1, T, HK * rep, DV)) * 0.25)
    g = rng.uniform(0.85, 0.99, (1, T, HK * rep)).astype(np.float32)
    beta = bf16_round(0.1 + 0.4 * rng.random((1, T, HK * rep)))
    return q, k, v, g, beta

def ref_scan(q, k, v, g, beta, state0):
    T = q.shape[1]
    HV = k.shape[2]
    st = state0[0].astype(np.float64)
    ys = []
    for t in range(T):
        outs = np.zeros((HV, DV), dtype=np.float64)
        for h in range(HV):
            dec = st[h] * g[0, t, h]
            kv = (dec * k[0, t, h][None, :]).sum(axis=1)
            delta = (v[0, t, h] - kv) * beta[0, t, h]
            ns = dec + delta[:, None] * k[0, t, h][None, :]
            outs[h] = (ns * q[0, t, h][None, :]).sum(axis=1)
            st[h] = ns
        ys.append(outs)
    return np.stack(ys)[None], st[None]

def run_rep(rep, mode):
    import mlx.core as mx
    HV = HK * rep
    for T in (64, 65):
        q, k, v, g, beta = make_inputs(T, rep)
        kr = np.repeat(k, rep, axis=2)
        qr = np.repeat(q, rep, axis=2)
        yref, _ = ref_scan(qr.astype(np.float64), kr.astype(np.float64),
                        v.astype(np.float64), g.astype(np.float64),
                        beta.astype(np.float64), np.zeros((1, HV, DV, DK)))
        _, stref = ref_scan(qr.astype(np.float64), kr.astype(np.float64),
                         v.astype(np.float64), g.astype(np.float64),
                         beta.astype(np.float64), np.zeros((1, HV, DV, DK)))
        outs = {}
        for mm in ("maskless", "allvalid"):
            qm = mx.array(qr).astype(mx.bfloat16)
            km = mx.array(kr).astype(mx.bfloat16)
            vm = mx.array(v).astype(mx.bfloat16)
            gm = mx.array(g)
            bm = mx.array(beta).astype(mx.bfloat16)
            st0 = mx.zeros((1, HV, DV, DK), dtype=mx.float32)
            mask = mx.ones((1, T), dtype=mx.bool_) if mm == "allvalid" else None
            y, st = mx.fast.gated_delta_update(qm, km, vm, gm, bm, st0, mask)
            mx.eval(y, st)
            outs[mm] = np.array(y.astype(mx.float32)).astype(np.float64)
            stm = np.array(st).astype(np.float32)
            outs[mm + "_stdiff"] = np.abs(stm.astype(np.float64) - stref).max()
            outs[mm + "_stsha"] = hashlib.sha256(stm.tobytes()).hexdigest()[:12]
        bit = np.array_equal(outs["maskless"], outs["allvalid"])
        d0 = np.abs(outs["maskless"] - yref).max()
        d1 = np.abs(outs["allvalid"] - yref).max()
        tok_diff = -1
        if not bit:
            per = np.abs(outs["maskless"] - outs["allvalid"]).max(axis=(0, 2, 3))
            tok_diff = int(np.argmax(per > 1e-6))
        print(f"REPW {mode} rep={rep} T={T} bitsame={bit} first_diff_tok={tok_diff} "
              f"maskless_vs_ref={d0:.3g} allvalid_vs_ref={d1:.3g} "
              f"st_diff_maskless={outs['maskless_stdiff']:.3g} "
              f"st_diff_allvalid={outs['allvalid_stdiff']:.3g} "
              f"st_m={outs['maskless_stsha']} st_a={outs['allvalid_stsha']}", flush=True)

def oracle(rep):
    """Primitive-level composed reference: gated_delta_ops in f32 on the same
    inputs; validates the fp64 reference and arbitrates the two GPU routes."""
    import mlx.core as mx
    import mlx_lm.models.gated_delta as gd
    HV = HK * rep
    T = 64
    q, k, v, g, beta = make_inputs(T, rep)
    kr = np.repeat(k, rep, axis=2)
    qr = np.repeat(q, rep, axis=2)
    yref64, stref = ref_scan(qr.astype(np.float64), kr.astype(np.float64),
                      v.astype(np.float64), g.astype(np.float64),
                      beta.astype(np.float64), np.zeros((1, HV, DV, DK)))
    qo = mx.array(qr).astype(mx.float32)
    ko = mx.array(kr).astype(mx.float32)
    vo = mx.array(v).astype(mx.float32)
    go = mx.array(g)
    bo = mx.array(beta).astype(mx.float32)
    st0 = mx.zeros((1, HV, DV, DK), dtype=mx.float32)
    y_ops, _ = gd.gated_delta_ops(qo, ko, vo, go, bo, st0, None)
    mx.eval(y_ops)
    yops = np.array(y_ops).astype(np.float64)
    d_ref = np.abs(yops - yref64).max()
    print(f"ORACLE rep={rep} ops_vs_fp64ref_max={d_ref:.3g}", flush=True)

import hashlib
mode = sys.argv[1] if len(sys.argv) > 1 else "scan"
print("ENV NO_COOPMAT", os.environ.get("MLX_OMARCHY_NO_COOPMAT", "<unset>"),
      "GDN_BATCH", os.environ.get("MLX_OMARCHY_GDN_BATCH", "<unset=on>"), flush=True)
if mode.startswith("scan") or mode == "default":
    for rep in (3, 1):
        run_rep(rep, mode)
elif mode == "oracle":
    oracle(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
