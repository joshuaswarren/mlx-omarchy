# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Analysis for probe v7: invert the exposure equations to native gate
values, compare against the jwm1 GPU gate values through the dense macstudio
unary LUTs, and report per-gate offset histograms.
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


def r16(v):
    return np.asarray(v, np.float64).astype(F16)


def build_sigma_lut():
    z = np.load(MAC / "sigma_full.npz")
    lut = np.full(65536, np.float16(0.0), dtype=F16)
    args = z["args"].astype(F16)
    lut[args.view(np.uint16).astype(np.int64)] = z["sigma"].astype(F16)
    allbits = np.arange(65536, dtype=np.uint16)
    arg = allbits.view(np.float16).astype(np.float64)
    nanmask = np.isnan(lut)
    with np.errstate(over="ignore"):
        lut[nanmask] = r16(1.0 / (1.0 + np.exp(-arg[nanmask])))
    lut[arg > 30.0] = np.float16(1.0)
    lut[arg < -30.0] = np.float16(0.0)
    lut[0x0000] = np.float16(0.5)
    lut[0x8000] = np.float16(0.5)
    return lut


def build_tanh_lut():
    z = np.load(MAC / "tanh_full.npz")
    lut = np.zeros(65536, dtype=F16)
    args = z["args"].astype(F16)
    lut[args.view(np.uint16).astype(np.int64)] = z["tanh"].astype(F16)
    allbits = np.arange(65536, dtype=np.uint16)
    arg = allbits.view(np.float16).astype(np.float64)
    lut[arg >= 12.0] = TANH_CLAMP
    lut[arg <= -12.0] = np.float16(-TANH_CLAMP)
    return lut


def ordered_bits(x):
    """Map fp16 to monotonically ordered ints (+0 and -0 share a key)."""
    b = np.asarray(x, F16).view(np.uint16).astype(np.int32)
    return np.where(b & 0x8000, 0x8000 - (b & 0x7FFF), 0x8000 + (b & 0x7FFF))


def ulp_diff(a, b):
    return np.abs(ordered_bits(a) - ordered_bits(b))


def bucket(prod, grid):
    """fp16 product value -> list of grid args producing it."""
    d = {}
    for g, p in zip(grid, prod):
        d.setdefault(p.view(np.uint16), []).append(g)
    return d


def main() -> int:
    sig_lut = build_sigma_lut()
    tan_lut = build_tanh_lut()
    allbits = np.arange(65536, dtype=np.uint16)
    allargs = allbits.view(np.float16)
    finite = np.isfinite(allargs)
    sig_pre = {}
    tan_pre = {}
    for a, v in zip(allargs[finite], sig_lut[finite]):
        sig_pre.setdefault(v.view(np.uint16), []).append(a)
    for a, v in zip(allargs[finite], tan_lut[finite]):
        tan_pre.setdefault(v.view(np.uint16), []).append(a)
    grid = np.unique(allargs[finite])

    gd = np.load(HERE / "gate_dump.npz")
    tr = np.load(HERE / "traces.npz")
    idxs = sorted({int(k[1:5]) for k in tr.files})

    native = {}
    summary = {}
    for lay, suf in (("L0", "0"), ("L1n", "1n")):
        for target in ("i", "f", "o", "g"):
            cands_per_lane = []
            for ci, i in enumerate(idxs):
                eqs = []
                if target in ("o", "f"):
                    obs_key = "h" if target == "o" else "c"
                    for m in (O_C0 if target == "o" else F_C0):
                        z = np.load(OUT / f"obs_{lay}_{target}_{m}.npz")
                        u = np.float16(m)
                        if target == "o":
                            u = tan_lut[u.view(np.uint16)]
                        obs = z[obs_key][ci].ravel()
                        prod = r16(grid.astype(np.float64) * float(u))
                        eqs.append((bucket(prod, grid), obs))
                elif target == "i":
                    for m in I_BG:
                        z = np.load(OUT / f"obs_{lay}_i_{m}.npz")
                        K = tan_lut[np.float16(m).view(np.uint16)]
                        obs = z["c"][ci].ravel()
                        prod = r16(grid.astype(np.float64) * float(K))
                        eqs.append((bucket(prod, grid), obs))
                else:  # g
                    for m in G_BI:
                        z = np.load(OUT / f"obs_{lay}_g_{m}.npz")
                        iK = sig_lut[np.float16(m).view(np.uint16)]
                        obs = z["c"][ci].ravel()
                        prod = r16(float(iK) * grid.astype(np.float64))
                        eqs.append((bucket(prod, grid), obs))
                for lane in range(640):
                    s = None
                    for bk, obs in eqs:
                        ok = np.array(bk.get(obs[lane].view(np.uint16), []), F16)
                        s = ok if s is None else np.intersect1d(s, ok)
                    cands_per_lane.append(s)

            lut = sig_lut if target != "g" else tan_lut
            pre = sig_pre if target != "g" else tan_pre
            nl = len(cands_per_lane)
            nat = np.full(nl, np.float16(np.nan), F16)
            sizes = np.zeros(nl, np.int32)
            for li, s in enumerate(cands_per_lane):
                sizes[li] = s.size
                if s.size == 1:
                    nat[li] = s[0]
            key = f"{lay}_{target}"
            native[key] = {"gate16": nat, "sizes": sizes}
            n0 = int((sizes == 0).sum())
            n1 = int((sizes == 1).sum())
            summary[key] = {"lanes": nl, "empty": n0, "singleton": n1,
                            "multi": nl - n0 - n1}
            print(key, json.dumps(summary[key]), flush=True)

    # ---- compare native gate16 vs GPU gate16 ------------------------------ #
    offsets = {}
    compare = {}
    for lay, suf in (("L0", "0"), ("L1n", "1n")):
        for target in ("i", "f", "o", "g"):
            key = f"{lay}_{target}"
            nat = native[key]["gate16"]
            gpu = np.concatenate([gd[f"t{i:04d}_{target}{suf}"].ravel() for i in idxs])
            gpu16 = (sig_lut if target != "g" else tan_lut)[
                np.asarray(gpu, F16).view(np.uint16)]
            valid = ~np.isnan(nat.astype(np.float32))
            d = ulp_diff(nat[valid], gpu16[valid])
            hist = Counter(d.tolist())
            offsets[key] = {str(k): int(v) for k, v in sorted(hist.items())}
            compare[key] = {"valid": int(valid.sum()), "exact": int((d == 0).sum()),
                            "within2": int((d <= 2).sum())}
            print(key, "exact", compare[key]["exact"], "/", compare[key]["valid"],
                  "offsets:", json.dumps(offsets[key]), flush=True)

    np.savez(OUT / "native_gates.npz",
             **{f"{k}_gate16": v["gate16"] for k, v in native.items()},
             **{f"{k}_sizes": v["sizes"] for k, v in native.items()})
    report = {"schema": "mlx-omarchy.fused-lstm-gate-preact-analysis/7",
              "singleton": summary, "gate_offsets": offsets, "compare": compare}
    (OUT / "analysis_v7.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print("WROTE", OUT / "analysis_v7.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
