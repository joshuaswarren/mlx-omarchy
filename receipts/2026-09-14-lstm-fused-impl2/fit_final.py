# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Final fit: fused ios18.lstm composition over ALL native-input decoder traces.

Emits unary_tables2.npz: sigma/tanh LUTs indexed by fp16 bit pattern, built
from the dense macstudio direct reads (sigma_full.npz, tanh_full.npz; see
receipts/2026-09-14-lstm-unary-mac.md). No interpolation anywhere.

Fills (from the macstudio contract):
  sigmoid   arg +0/-0 -> 0.5 (two candidates, identical exposures)
            17 ambiguous args (NaN in the read-out) -> correctly rounded
            arg > 30 -> saturate 1.0; arg < -30 -> 0.0
  tanh      arg +0/-0 -> +0.0 (-0.0 collapses to +0.0)
            |arg| > 12 (incl. the (12,16) unmeasured gap) -> clamp
            +-(1 - 2^-10) = 0.9990234375 (verified constant 8..30)

Composition under test (per lane, operands fp16, arithmetic f64):
  c1   = round16(sigma_f * c0 + sigma_i * tanh16(g))
  h    = round16(sigma_o * tanh16(round16(c1)))
The winning hidden form was decided on trace 0 layer 0 (637/640 vs ~400 for
raw-tanh variants); gates are the jwm1 GPU values from dump_gates.py, layer 1
injected with the native layer-0 output (t????_i1n), so a bit-exact lane
pins the composition, not the matmuls.

Baseline: the correctly-rounded overlay contract (numpy CR sigmoid/tanh at
the same fp16 gate args) for drift comparison.
"""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MAC = Path.home() / "src/mlx-omarchy/receipts/2026-09-14-lstm-unary-mac"
CAPTURE = (
    Path.home()
    / ".cache/mlx-omarchy/parakeet-reference/captures/b650695c-75aec2a/"
    "20260913T105550Z-librispeech-tdt-tensors/ane"
)
F16 = np.dtype("<f2")
TANH_CLAMP = np.float16(1.0 - 2.0**-10)


def r16(values):
    return np.asarray(values, np.float64).astype(F16)


def build_sigma_lut():
    z = np.load(MAC / "sigma_full.npz")
    args = z["args"].astype(F16)
    values = z["sigma"].astype(F16)
    lut = np.full(65536, np.float16(0.0), dtype=F16)
    lut[args.view(np.uint16).astype(np.int64)] = values
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
    args = z["args"].astype(F16)
    values = z["tanh"].astype(F16)
    lut = np.zeros(65536, dtype=F16)
    lut[args.view(np.uint16).astype(np.int64)] = values
    allbits = np.arange(65536, dtype=np.uint16)
    arg = allbits.view(np.float16).astype(np.float64)
    lut[arg >= 12.0] = TANH_CLAMP
    lut[arg <= -12.0] = np.float16(-TANH_CLAMP)
    lut[0x0000] = np.float16(0.0)
    lut[0x8000] = np.float16(0.0)
    return lut


def cr_sigmoid(x_f16):
    x = np.asarray(x_f16, np.float64)
    return r16(1.0 / (1.0 + np.exp(-x)))


def cr_tanh(x_f16):
    return r16(np.tanh(np.asarray(x_f16, np.float64)))


def main() -> int:
    sigma_lut = build_sigma_lut()
    tanh_lut = build_tanh_lut()
    np.savez(HERE / "unary_tables2.npz", sigma_lut=sigma_lut, tanh_lut=tanh_lut)

    gates = np.load(HERE / "gate_dump.npz")
    traces = json.loads((CAPTURE / "tdt_tensors.json").read_text())["traces"]
    ran = [t["index"] for t in traces if t.get("ran_decoder")]

    def lut_lookup(lut, x):
        return lut[np.asarray(x, F16).view(np.uint16)].astype(np.float64)

    def fused_stage(ii, ff, oo, gg, c0):
        sig_i = lut_lookup(sigma_lut, ii)
        sig_f = lut_lookup(sigma_lut, ff)
        sig_o = lut_lookup(sigma_lut, oo)
        tau_g = lut_lookup(tanh_lut, gg)
        c1 = r16(sig_f * c0.astype(np.float64) + sig_i * tau_g)
        tau_c1 = lut_lookup(tanh_lut, c1)
        h = r16(sig_o * tau_c1)
        return c1, h

    def cr_stage(ii, ff, oo, gg, c0):
        sig_i = cr_sigmoid(ii).astype(np.float64)
        sig_f = cr_sigmoid(ff).astype(np.float64)
        sig_o = cr_sigmoid(oo).astype(np.float64)
        tau_g = cr_tanh(gg).astype(np.float64)
        c1 = r16(sig_f * c0.astype(np.float64) + sig_i * tau_g)
        h = r16(sig_o * cr_tanh(c1).astype(np.float64))
        return c1, h

    report = {
        "schema": "mlx-omarchy.lstm-fused-fit2/1",
        "host": platform.node(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "traces": ran,
        "lanes_per_stage": len(ran) * 640 * 2,
        "fills": {
            "sigma_zero": 0.5,
            "sigma_saturate": "1.0 / 0.0 outside [-30, 30]",
            "sigma_ambiguous": "correctly rounded (17 args)",
            "tanh_clamp": "+-(1 - 2^-10) outside [-12, 12]",
        },
    }
    for name, stage in (("fused_dense", fused_stage), ("cr_baseline", cr_stage)):
        cell_exact = hid_exact = total = 0
        cell_max = cell_sum = 0.0
        hid_max = hid_sum = 0.0
        per_trace = {}
        for t in ran:
            tc = th = 0
            for layer in (0, 1):
                suffix = "0" if layer == 0 else "1n"
                ii = gates[f"t{t:04d}_i{suffix}"].ravel()
                ff = gates[f"t{t:04d}_f{suffix}"].ravel()
                oo = gates[f"t{t:04d}_o{suffix}"].ravel()
                gg = gates[f"t{t:04d}_g{suffix}"].ravel()
                c0 = gates[f"t{t:04d}_cell16_{layer}"].ravel()
                gold_c = np.load(
                    CAPTURE / f"tdt_trace_{t:04d}_decoder_next_cell.npy"
                )[layer].ravel()
                gold_h = np.load(
                    CAPTURE / f"tdt_trace_{t:04d}_decoder_next_hidden.npy"
                )[layer].ravel()
                c1, h = stage(ii, ff, oo, gg, c0)
                dc = np.abs(c1.astype(np.float64) - gold_c.astype(np.float64))
                dh = np.abs(h.astype(np.float64) - gold_h.astype(np.float64))
                tc += int((dc == 0).sum())
                th += int((dh == 0).sum())
                cell_max = max(cell_max, float(dc.max()))
                cell_sum += float(dc.sum())
                hid_max = max(hid_max, float(dh.max()))
                hid_sum += float(dh.sum())
            per_trace[str(t)] = {
                "cell_exact": f"{tc}/1280",
                "hidden_exact": f"{th}/1280",
            }
            cell_exact += tc
            hid_exact += th
            total += 1280
        report[name] = {
            "cell_exact": f"{cell_exact}/{total}",
            "hidden_exact": f"{hid_exact}/{total}",
            "cell_max_abs": cell_max,
            "cell_mean_abs": cell_sum / total,
            "hidden_max_abs": hid_max,
            "hidden_mean_abs": hid_sum / total,
            "per_trace": per_trace,
        }
    summary = {
        k: report[k] for k in ("schema", "host", "python", "numpy", "traces",
                               "lanes_per_stage", "fills")
    }
    for name in ("fused_dense", "cr_baseline"):
        summary[name] = {
            k: v for k, v in report[name].items() if k != "per_trace"
        }
    print(json.dumps(summary, indent=2, sort_keys=True))
    (HERE / "fit_report2.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
