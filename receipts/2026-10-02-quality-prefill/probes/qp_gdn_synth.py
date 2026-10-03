"""Single GDN layer @ 27B dims: mask=None vs all-True vs fp64 reference.

Dims (Qwen3.8-27B): Hk=16, Hv=48 (repeat 3), Dk=Dv=128, scalar decay.
Calls mx.fast.gated_delta_update directly (the primitive the backend
implements). The all-True mask is arithmetically a no-op, so maskless vs
allvalid must be BIT-IDENTICAL; both must track the fp64 reference.

Sweeps T across the route boundary (kGdnCoopmatMinTokens=64: coopmat at
T>=64, two-pass scan below). Inputs are drawn once per T and bf16-rounded
before both the GPU routes and the fp64 reference.
"""
import os, sys
import numpy as np

HK, HV, DK, DV = 16, 48, 128, 128
rng = np.random.default_rng(20261003)

def bf16_round(a):
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    return (u >> 16 << 16).view(np.float32)

def make_inputs(T):
    q = bf16_round(rng.standard_normal((1, T, HK, DK)) * 0.5)
    k = bf16_round(rng.standard_normal((1, T, HK, DK)) * 0.5)
    v = bf16_round(rng.standard_normal((1, T, HV, DV)) * 0.5)
    g = rng.uniform(0.9, 1.0, (1, T, HV)).astype(np.float32)
    beta = bf16_round(1.0 / (1.0 + rng.standard_normal((1, T, HV)) ** 2))
    return q, k, v, g, beta

def ref_scan(q, k, v, g, beta, state0):
    T = q.shape[1]
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
    return np.stack(ys)[None]

def gpu_run(T, q, k, v, g, beta, mask_mode, layout="contig"):
    import mlx.core as mx
    st0 = mx.zeros((1, HV, DV, DK), dtype=mx.float32)
    mask = mx.ones((1, T), dtype=mx.bool_) if mask_mode == "allvalid" else None
    if layout == "strided":
        # mirror the model: one fused projection holds UNREPEATED q/k + v;
        # q/k/v are SLICE VIEWS (non-contiguous, offset != 0); the repeat
        # happens after slicing, inside the wrapper.
        QD, KD, VD = HK * DK, HK * DK, HV * DV
        fused = mx.array(np.concatenate(
            [q.reshape(1, T, QD), k.reshape(1, T, KD), v.reshape(1, T, VD)],
            axis=2)).astype(mx.bfloat16)
        qm = mx.repeat(fused[:, :, :QD].reshape(1, T, HK, DK), HV // HK, axis=2)
        km = mx.repeat(fused[:, :, QD:QD + KD].reshape(1, T, HK, DK), HV // HK, axis=2)
        vm = fused[:, :, QD + KD:].reshape(1, T, HV, DV)
    else:
        qm = mx.repeat(mx.array(q).astype(mx.bfloat16), HV // HK, axis=2)
        km = mx.repeat(mx.array(k).astype(mx.bfloat16), HV // HK, axis=2)
        vm = mx.array(v).astype(mx.bfloat16)
    gm = mx.array(g)
    bm = mx.array(beta).astype(mx.bfloat16)
    y, st = mx.fast.gated_delta_update(qm, km, vm, gm, bm, st0, mask)
    mx.eval(y, st)
    return np.array(y.astype(mx.float32)), np.array(st)

mode = sys.argv[1] if len(sys.argv) > 1 else "sweep"
print("ENV GDN_BATCH", os.environ.get("MLX_OMARCHY_GDN_BATCH", "<unset=on>"),
      "NO_COOPMAT", os.environ.get("MLX_OMARCHY_NO_COOPMAT", "<unset>"),
      "MASKLESS", os.environ.get("MLX_OMARCHY_SSM_MASKLESS", "<unset=on>"), flush=True)

if mode == "sweep":
    for T in (4, 8, 16, 32, 63, 64, 65, 96, 352):
        q, k, v, g, beta = make_inputs(T)
        y0, s0 = gpu_run(T, q, k, v, g, beta, "maskless")
        y1, s1 = gpu_run(T, q, k, v, g, beta, "allvalid")
        y2, s2 = gpu_run(T, q, k, v, g, beta, "maskless", "strided")
        y3, s3 = gpu_run(T, q, k, v, g, beta, "allvalid", "strided")
        same_y = np.array_equal(y0, y1)
        same_s = np.array_equal(s0, s1)
        strid_ok = np.array_equal(y0, y2) and np.array_equal(y1, y3)
        yref = ref_scan(np.repeat(q, HV // HK, axis=2).astype(np.float64),
                        np.repeat(k, HV // HK, axis=2).astype(np.float64),
                        np.repeat(v, HV // HK, axis=2).astype(np.float64),
                        g.astype(np.float64), beta.astype(np.float64),
                        np.zeros((1, HV, DV, DK)))
        yref = np.repeat(yref, 1, axis=0)
        d0 = np.abs(y0.astype(np.float64) - yref).max()
        d1 = np.abs(y1.astype(np.float64) - yref).max()
        ds = np.abs(y2.astype(np.float64) - yref).max()
        print(f"SWEEP T={T} y_bitsame={same_y} st_bitsame={same_s} "
              f"strid_ok={strid_ok} strid_vs_ref_max={ds:.4g} "
              f"maskless_vs_ref_max={d0:.4g} allvalid_vs_ref_max={d1:.4g} "
              f"maskless_nan={int(np.isnan(y0).sum())}", flush=True)
elif mode == "tokens":
    import mlx.core as mx
    T = 352
    q, k, v, g, beta = make_inputs(T)
    for mm in ("maskless", "allvalid"):
        y, st = gpu_run(T, q, k, v, g, beta, mm)
        ynp = np.array(y.astype(mx.float32))
        tnorm = np.linalg.norm(ynp[0], axis=-1).mean(axis=1)
        print(f"TOKENS route={mm} first12={np.round(tnorm[:12], 4).tolist()} "
              f"nan_tokens={int(np.isnan(ynp[0]).any(axis=(1, 2)).sum())} "
              f"state_nan={int(np.isnan(st).sum())}", flush=True)
