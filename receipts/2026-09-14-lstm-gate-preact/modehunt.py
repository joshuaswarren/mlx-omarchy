# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Mode hunt against captured native gate values (probe v7).

Anchors the capture (selftest row = bias only), characterizes unresolved
lanes, then scores candidate GEMM accumulation forms by how often
LUT[fp16(z_form)] equals the captured native gate value per lane.
Writes only into ./out7/.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "out7"
MAC = Path("/private/tmp/mac-activation-capture/out5")
F16 = np.dtype("<f2")
TANH_CLAMP = np.float16(1.0 - 2.0**-10)
O_C0 = [0.25, 0.5, 1.0]
F_C0 = [0.25, 0.5, 1.0]
I_BG = [0.5, 1.0, 2.0]
G_BI = [1.0, 2.0, 4.0]
OFF = {"i": 0, "f": 640, "o": 1280, "g": 1920}


def r16(v):
    return np.asarray(v, np.float64).astype(F16)


def build_lut(name):
    z = np.load(MAC / f"{name}_full.npz")
    lut = np.zeros(65536, dtype=F16)
    args = z["args"].astype(F16)
    lut[args.view(np.uint16).astype(np.int64)] = z[name].astype(F16)
    allbits = np.arange(65536, dtype=np.uint16)
    arg = allbits.view(np.float16).astype(np.float64)
    nanmask = np.isnan(lut)
    if name == "sigma":
        out = np.full(65536, np.float16(0.0), dtype=F16)
        out[:] = lut
        with np.errstate(over="ignore"):
            out[nanmask] = r16(1.0 / (1.0 + np.exp(-arg[nanmask])))
        out[arg > 30.0] = np.float16(1.0)
        out[arg < -30.0] = np.float16(0.0)
        out[0x0000] = np.float16(0.5)
        out[0x8000] = np.float16(0.5)
        return out
    lut[arg >= 12.0] = TANH_CLAMP
    lut[arg <= -12.0] = np.float16(-TANH_CLAMP)
    return lut


def ob(x):
    b = np.asarray(x, F16).view(np.uint16).astype(np.int32)
    return np.where(b & 0x8000, 0x8000 - (b & 0x7FFF), 0x8000 + (b & 0x7FFF))


def main() -> int:
    sig = build_lut("sigma")
    tan = build_lut("tan") if (MAC / "tan_full.npz").exists() else build_lut("tanh")
    w = np.load(HERE / "weights.npz")
    tr = np.load(HERE / "traces.npz")
    gd = np.load(HERE / "gate_dump.npz")
    nat = np.load(OUT / "native_gates.npz")
    idxs = sorted({int(k[1:5]) for k in tr.files})
    grid = np.unique(np.arange(65536, dtype=np.uint16).view(np.float16)
                     [np.isfinite(np.arange(65536, dtype=np.uint16).view(np.float16))])
    names = {"L0": ("concat_1_to_fp16", "concat_2_to_fp16", "concat_0_to_fp16"),
             "L1n": ("concat_4_to_fp16", "concat_5_to_fp16", "concat_3_to_fp16")}
    report = {}

    # ---- anchor: selftest row = bias only --------------------------------- #
    anchor = {}
    for lay in names:
        b = w[names[lay][2]]
        for target, mults in (("i", I_BG), ("f", F_C0), ("o", O_C0), ("g", G_BI)):
            lut = sig if target != "g" else tan
            expect = lut[np.asarray(b[OFF[target]:OFF[target] + 640], F16).view(np.uint16)]
            eqs = []
            for m in mults:
                z = np.load(OUT / f"obs_{lay}_{target}_{m}.npz")
                u = np.float16(m)
                if target in ("o", "i"):
                    u = tan[u.view(np.uint16)]
                if target == "g":
                    u = sig[u.view(np.uint16)]
                obs = z["h" if target == "o" else "c"][15].ravel()
                prod = r16(float(u) * grid.astype(np.float64))
                eqs.append((prod, obs))
            # lanes whose v5 raw read is NaN (ambiguous args, cr-filled) are exempt
            name = "sigma" if target != "g" else "tanh"
            raw = np.load(MAC / f"{name}_full.npz")
            raw_nan_args = raw["args"].astype(F16)[np.isnan(raw[name].astype(F16))]
            bts = np.asarray(b, F16)[OFF[target]:OFF[target] + 640]
            nanfill = np.isin(bts.view(np.uint16), raw_nan_args.view(np.uint16))
            resolved = exact = singleton = checked = 0
            for lane in range(640):
                if nanfill[lane]:
                    continue
                checked += 1
                s = None
                for prod, obs in eqs:
                    ok = grid[prod == obs[lane]]
                    s = ok if s is None else np.intersect1d(s, ok)
                if s.size:
                    resolved += 1
                    exact += int(np.any(s == expect[lane]))
                singleton += int(s.size == 1)
            anchor[f"{lay}_{target}"] = {"checked": checked, "resolved": resolved,
                                         "expect_in_cands": exact, "singleton": singleton}
            assert resolved == checked and exact == checked, \
                f"anchor failed {lay}_{target}: {anchor[f'{lay}_{target}']}"
    print("anchor:", json.dumps(anchor), flush=True)

    # ---- mode hunt per (layer, target) ------------------------------------ #
    hunt = {}
    for lay, (n_ih, n_hh, n_b) in names.items():
        wih, whh, bias = (np.asarray(w[n], np.float64) for n in (n_ih, n_hh, n_b))
        for target in ("i", "f", "o", "g"):
            key = f"{lay}_{target}"
            lut = sig if target != "g" else tan
            nt = nat[key + "_gate16"]
            single = nat[key + "_sizes"] == 1
            X = np.concatenate([tr[f"t{i:04d}_x0"] for i in idxs]).astype(np.float64)
            H = np.concatenate([tr[f"t{i:04d}_h"][0:1] if lay == "L0" else tr[f"t{i:04d}_h"][1:2]
                                for i in idxs]).astype(np.float64)
            wt = wih[OFF[target]:OFF[target] + 640]
            wh = whh[OFF[target]:OFF[target] + 640]
            bt = bias[OFF[target]:OFF[target] + 640]
            Px = np.einsum("rk,lk->rlk", X, wt)             # exact (f64)
            Ph = np.einsum("rk,lk->rlk", H, wh)
            P = np.concatenate([Px, Ph], axis=2)            # (n, lanes, 1280) k-order x|h
            P32 = P.astype(np.float32)
            P16 = P.astype(F16)
            f32 = lambda a: a.astype(np.float32)
            forms = {}
            forms["f64_exact"] = P.sum(axis=2) + bt
            forms["f32_pairwise"] = P32.sum(axis=2, dtype=np.float32)
            forms["f32_seq"] = np.cumsum(P32, axis=2, dtype=np.float32)[:, :, -1]
            forms["f16_pairwise"] = P16.sum(axis=2, dtype=F16).astype(np.float64)
            forms["f16_seq"] = np.cumsum(P16, axis=2, dtype=F16)[:, :, -1].astype(np.float64)
            sx = P32[:, :, :640].sum(axis=2, dtype=np.float32)
            sh = P32[:, :, 640:].sum(axis=2, dtype=np.float32)
            forms["two_f32"] = sx + sh
            forms["two_f32_r16each"] = r16(sx.astype(np.float64) + sh.astype(np.float64))
            sx16 = sx.astype(F16).astype(np.float64)
            sh16 = sh.astype(F16).astype(np.float64)
            b16 = np.asarray(bt, F16).astype(np.float64)
            forms["vulkan_3op"] = r16(r16(sx16 + sh16) + b16)
            forms["vulkan_bias_in"] = r16(sx16 + sh16 + b16)
            Pv = np.empty_like(P)
            Pv[:, :, 0::2] = Px
            Pv[:, :, 1::2] = Ph
            forms["packed_f32"] = Pv.astype(np.float32).sum(axis=2, dtype=np.float32)
            forms["hidfirst_f32"] = sh + sx
            q = P32.reshape(P32.shape[0], P32.shape[1], 4, 320)
            forms["splitk4_f32"] = q.sum(axis=3, dtype=np.float32).sum(axis=2, dtype=np.float32)
            gpu = np.concatenate([gd[f"t{i:04d}_{target}{'0' if lay == 'L0' else '1n'}"].ravel()
                                  for i in idxs])
            ok = single & np.isfinite(nt.astype(np.float32))
            scores = {}
            for fname, zv in forms.items():
                z16 = np.asarray(zv, np.float64).astype(F16).ravel()
                gate = lut[z16.view(np.uint16)]
                scores[fname] = int(np.count_nonzero(gate[ok] == nt[ok]))
            gate = lut[np.asarray(gpu, F16).view(np.uint16)]
            scores["gpu_dump"] = int(np.count_nonzero(gate[ok] == nt[ok]))
            hunt[key] = {"resolved": int(single.sum()), "scores": scores}
            print(key, json.dumps(scores), flush=True)
    report["hunt"] = hunt
    (OUT / "modehunt.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print("WROTE", OUT / "modehunt.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
