# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Host-only numpy hunt for the BNNS fused-LSTM accumulator form.

Scores candidate gate-preact accumulation forms against the probe-v7 captured
native gate values (receipts/2026-09-14-lstm-fused-impl2.md follow-up).

Captured native gate value per lane = the native fp16 sigmoid/tanh OUTPUT,
pinned uniquely by the v7 three-multiplier intersection (singleton lanes only).
A candidate form z is scored by LUT[fp16(z)] == captured output.

Correction vs modehunt.py: for L1n the capture probes fed native next_hidden[0]
(traces.npz tXXXX_nh[0]) as the sequence row (run_gate_preact7.py load_cases),
not the embedding row x0. modehunt scored L1n against x0, which is why every
host form scored ~0 there; this script uses nh[0] for L1n.

No MLX, no GPU, no device. Writes bnns_acc_hunt.json next to this file.
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
F16 = np.dtype("<f2")
TANH_CLAMP = np.float16(1.0 - 2.0**-10)
MAC = Path.home() / "src/mlx-omarchy/receipts/2026-09-14-lstm-unary-mac"
IMPL2 = Path.home() / "src/mlx-omarchy/receipts/2026-09-14-lstm-fused-impl2"
ART = Path("/tmp/gate-preact-art")
OFF = {"i": 0, "f": 640, "o": 1280, "g": 1920}
NAMES = {"L0": ("concat_1_to_fp16", "concat_2_to_fp16", "concat_0_to_fp16"),
         "L1n": ("concat_4_to_fp16", "concat_5_to_fp16", "concat_3_to_fp16")}
HID = 640


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
        out = np.zeros(65536, dtype=F16)
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
    b = np.asarray(x, F16).view(np.uint16).astype(np.int64)
    return np.where(b & 0x8000, 0x8000 - (b & 0x7FFF), 0x8000 + (b & 0x7FFF))

def round_mantissa(x_f32, bits):
    """RNE-round float32 array to `bits` mantissa bits (bf16=7, tf32=10)."""
    mb = 23 - bits
    mask_low = np.uint32((1 << mb) - 1)
    half = np.uint32(1 << (mb - 1))
    xi = x_f32.view(np.uint32)
    low = xi & mask_low
    base = xi & ~mask_low
    lsb = (base >> np.uint32(mb)) & np.uint32(1)
    up = (low > half) | ((low == half) & (lsb == 1))
    out = np.where(up, base + (np.uint32(1) << np.uint32(mb)), base)
    return out.astype(np.uint32).view(np.float32).astype(np.float64)


def p_round(P, kind):
    if kind == "exact":
        return P
    if kind == "p16":
        return P.astype(F16).astype(np.float64)
    P32 = P.astype(np.float32)
    if kind == "pbf16":
        return round_mantissa(P32, 7)
    if kind == "ptf32":
        return round_mantissa(P32, 10)
    raise ValueError(kind)


def acc_f64(P):
    return P.sum(axis=2)


def acc_f64_seq(P):
    return np.cumsum(P, axis=2)[:, :, -1]


def acc_f32_np(P32):
    return P32.sum(axis=2, dtype=np.float32).astype(np.float64)


def acc_f32_seq(P32):
    return np.cumsum(P32, axis=2, dtype=np.float32)[:, :, -1].astype(np.float64)


def acc_blocks(P32, B, combine):
    n, lanes, k = P32.shape
    nb = k // B
    q = P32.reshape(n, lanes, nb, B)
    part = np.cumsum(q, axis=3, dtype=np.float32)[:, :, :, -1]  # f32 seq within block
    if combine == "f32seq":
        return np.cumsum(part, axis=2, dtype=np.float32)[:, :, -1].astype(np.float64)
    if combine == "f32np":
        return part.sum(axis=2, dtype=np.float32).astype(np.float64)
    if combine == "f64":
        return part.astype(np.float64).sum(axis=2)
    raise ValueError(combine)


def acc_tree(P32, base):
    n, lanes, k = P32.shape
    cur = P32
    while cur.shape[2] > base:
        m = (cur.shape[2] // base) * base
        head, tail = cur[:, :, :m], cur[:, :, m:]
        groups = head.reshape(n, lanes, -1, base)
        agg = groups.sum(axis=3, dtype=np.float32)
        cur = np.concatenate([agg.reshape(n, lanes, -1), tail], axis=2)
    return cur.sum(axis=2, dtype=np.float32).astype(np.float64)

def acc_fp16carry(P, N):
    """f32 products; per-N chunk the chunk sum joins an fp16-carried accumulator."""
    n, lanes, k = P32shape = P.shape
    P32 = P.astype(np.float32)
    nb = k // N
    q = P32.reshape(n, lanes, nb, N)
    chunk = q.sum(axis=3, dtype=np.float32)  # f32 chunk sum (exact products)
    carry = np.zeros((n, lanes), np.float64)
    for j in range(nb):
        carry = r16(chunk[:, :, j].astype(np.float64) + carry)
    return carry


def packed_order(Px, Ph, s):
    """Interleave x and h k-blocks of size s: x[0:s], h[0:s], x[s:2s], ..."""
    n, lanes, k = Px.shape
    idx = []
    for j in range(0, k, s):
        idx.append(np.arange(j, j + s))            # x part
        idx.append(k + np.arange(j, j + s))        # h part
    return np.concatenate([Px, Ph], axis=2)[:, :, np.concatenate(idx)]


def main() -> int:
    t0 = time.perf_counter()
    SIG = build_lut("sigma")
    TAN = build_lut("tanh")
    W = np.load(IMPL2 / "weights.npz")
    TR = np.load(ART / "traces.npz")
    NAT = np.load(ART / "native_gates.npz")
    GD = np.load(IMPL2 / "gate_dump.npz")
    idxs = sorted({int(k[1:5]) for k in TR.files})

    results = {"schema": "mlx-omarchy.bnns-acc-hunt/1",
               "host": {"hostname": platform.node(), "python": sys.version.split()[0],
                        "numpy": np.__version__, "kernel": platform.release()},
               "groups": {}, "forms": {}}

    def forms_for(Px, Ph, bt):
        """Return dict name -> z (f64), given exact f64 products and fp16 bias."""
        out = {}
        k = Px.shape[2]
        XH = np.concatenate([Px, Ph], axis=2)
        HX = np.concatenate([Ph, Px], axis=2)
        b64 = bt.astype(np.float64)
        b16 = np.asarray(bt, F16).astype(np.float64)
        # --- exact-product, order xh ---
        out["xh_f64"] = acc_f64(XH) + b64
        out["xh_f64_f32f16"] = (acc_f64(XH).astype(np.float32)).astype(np.float64) + b64
        P32 = XH.astype(np.float32)
        out["xh_f32np"] = acc_f32_np(P32) + b64
        out["xh_f32seq"] = acc_f32_seq(P32) + b64
        for B in (4, 8, 16, 32, 64, 128, 160, 320):
            out[f"xh_blk{B}_f32seq"] = acc_blocks(P32, B, "f32seq") + b64
            out[f"xh_blk{B}_f64"] = acc_blocks(P32, B, "f64") + b64
        out["xh_tree8"] = acc_tree(P32, 8) + b64
        out["xh_tree16"] = acc_tree(P32, 16) + b64
        for N in (4, 8, 16, 32, 64, 128, 320):
            out[f"xh_c16_{N}"] = acc_fp16carry(XH, N) + b64
        # --- order hx ---
        out["hx_f64"] = acc_f64(HX) + b64
        HXP = HX.astype(np.float32)
        out["hx_f32np"] = acc_f32_np(HXP) + b64
        out["hx_f32seq"] = acc_f32_seq(HXP) + b64
        # --- packed interleaves (f64 + f32np) ---
        for s in (1, 2, 4, 8, 16, 32, 64):
            PK = packed_order(Px, Ph, s)
            out[f"pk{s}_f64"] = acc_f64(PK) + b64
            out[f"pk{s}_f32np"] = acc_f32_np(PK.astype(np.float32)) + b64
            out[f"pk{s}_f32seq"] = acc_f32_seq(PK.astype(np.float32)) + b64
        # --- product rounding (order xh) ---
        for kind in ("p16", "pbf16", "ptf32"):
            PR = p_round(XH, kind)
            out[f"xh_{kind}_f64"] = acc_f64(PR) + b64
            PR32 = PR.astype(np.float32)
            out[f"xh_{kind}_f32np"] = acc_f32_np(PR32) + b64
            out[f"xh_{kind}_f32seq"] = acc_f32_seq(PR32) + b64
            for B in (8, 16, 32):
                out[f"xh_{kind}_blk{B}_f64"] = acc_blocks(PR32, B, "f64") + b64
        # --- split x/h combine + bias placement ---
        sxf = Px.sum(axis=2)
        shf = Ph.sum(axis=2)
        sx32 = sxf.astype(np.float32).astype(np.float64)
        sh32 = shf.astype(np.float32).astype(np.float64)
        sx16 = r16(sxf)
        sh16 = r16(shf)
        out["sep_f64"] = sxf + shf + b64
        out["sep_f32_x_h_b"] = sx32 + sh32 + b64
        out["sep_f32_b_x_h"] = b64 + sx32 + sh32
        out["sep_vulkan3op"] = r16(r16(sx16 + sh16) + b16)
        out["sep_vulkan_bias_in"] = r16(sx16 + sh16 + b16)
        out["sep_sx16_sh32"] = sx16 + sh32 + b64
        out["sep_sx32_sh16"] = sx32 + sh16 + b64
        out["sep_sx16_sh16_f64_b"] = sx16 + sh16 + b64
        out["sep_bx_h"] = (sxf + b64).astype(np.float32).astype(np.float64) + sh32
        out["sep_x_bh"] = sx32 + (shf + b64).astype(np.float32).astype(np.float64)
        out["sep_xh_b16end_f64"] = sxf + shf + b16
        out["sep_16_16_16"] = r16(r16(sx16 + sh16) + b16)
        # descending k within parts
        XR = np.concatenate([Px[:, :, ::-1], Ph[:, :, ::-1]], axis=2)
        out["xhrev_f64"] = acc_f64(XR) + b64
        out["xhrev_f32np"] = acc_f32_np(XR.astype(np.float32)) + b64
        return out

    agg = Counter()
    totals = {}
    for lay in ("L0", "L1n"):
        wih, whh, bias = (np.asarray(W[n], np.float64) for n in NAMES[lay])
        if lay == "L0":
            X = np.concatenate([TR[f"t{i:04d}_x0"] for i in idxs]).astype(np.float64)
            H = np.concatenate([TR[f"t{i:04d}_h"][0:1] for i in idxs]).astype(np.float64)
        else:
            X = np.concatenate([TR[f"t{i:04d}_nh"][0:1] for i in idxs]).astype(np.float64)
            H = np.concatenate([TR[f"t{i:04d}_h"][1:2] for i in idxs]).astype(np.float64)
        for target in ("i", "f", "o", "g"):
            key = f"{lay}_{target}"
            o = OFF[target]
            wt, wh, bt = wih[o:o + HID], whh[o:o + HID], bias[o:o + HID]
            Px = np.einsum("rk,lk->rlk", X, wt)
            Ph = np.einsum("rk,lk->rlk", H, wh)
            nt = NAT[key + "_gate16"]
            sizes = NAT[key + "_sizes"]
            ok = (sizes == 1) & np.isfinite(nt.astype(np.float32))
            lut = TAN if target == "g" else SIG
            nt_ok = nt[ok]
            n_ok = int(ok.sum())
            forms = forms_for(Px, Ph, bt)
            gd = np.concatenate([GD[f"t{i:04d}_{target}{'0' if lay == 'L0' else '1n'}"].ravel()
                                 for i in idxs])
            forms["gpu_dump"] = np.asarray(gd, np.float64)
            forms["f64_f32round_plus_b16"] = None  # placeholder removed below
            forms.pop("f64_f32round_plus_b16")
            gres = {}
            for fname, zv in forms.items():
                z16 = np.asarray(zv, np.float64).astype(F16).ravel()
                gate = lut[z16.view(np.uint16)]
                eq = gate[ok] == nt_ok
                sc = int(np.count_nonzero(eq))
                gres[fname] = sc
                agg[fname] += sc
            # offset histogram (output ulps) for a few reference forms
            hist = {}
            for fname in ("xh_f64", "sep_vulkan3op", "gpu_dump"):
                z16 = np.asarray(forms[fname], np.float64).astype(F16).ravel()
                gate = lut[z16.view(np.uint16)]
                d = np.abs(ob(gate[ok]) - ob(nt_ok))
                hist[fname] = {str(k): int(v) for k, v in
                               sorted(Counter(d.tolist()).items())[:12]}
            results["groups"][key] = {"ok": n_ok, "scores": gres, "offsets": hist}
            best = max(gres, key=gres.get)
            print(key, "ok", n_ok, "best", best, gres[best],
                  "gpu", gres["gpu_dump"], "f64", gres["xh_f64"], flush=True)
    totals = {"ok": sum(v["ok"] for v in results["groups"].values()),
              "scores": dict(agg)}
    results["totals"] = totals
    top = sorted(agg.items(), key=lambda kv: -kv[1])[:25]
    results["top"] = top
    print("TOTALS ok", totals["ok"])
    for name, sc in top:
        print(f"  {name:32s} {sc:6d}  {100.0*sc/totals['ok']:.2f}%")
    (HERE / "bnns_acc_hunt.json").write_text(json.dumps(results, indent=1) + "\n")
    print("WROTE", HERE / "bnns_acc_hunt.json", f"elapsed {time.perf_counter()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
