#!/usr/bin/env python3
"""CPU f64 truth + ULP statistics for the captured BF16 GEMV cells.

Rebuilds the identical inputs (real capture tensors + model weights +
deterministic patterned sets), computes RNE(f64) of x@W.T (+b) with the
root-cause convention, and compares every cell's captured bits against it.

Usage: m1_ulp_truth.py OUT.json CELL=DIR [CELL=DIR ...]
"""
import json
import sys
from pathlib import Path

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MLX_DISABLE_COMPILE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

import mlx.core as mx
mx.set_default_device(mx.cpu)
from mlx_lm.utils import load
from ulp_common import (capture_tensor, patterned_bits, rne_bf16, synthetic_cancel,
                        to_f64)


def truth_for(name, model):
    """f64 RNE truth (bits, acc) plus the native Metal output bits."""
    layer = model.model.layers[0]
    phase, proj = name.split(".")
    x = capture_tensor(f"{phase}.{proj}.input")
    native = capture_tensor(f"{phase}.{proj}.output").reshape(-1)
    lin = getattr(layer.self_attn, proj)
    k = lin.weight.shape[1]
    acc = to_f64(x).reshape(-1, k) @ to_f64(
        np.asarray(lin.weight.view(mx.uint16), dtype=np.uint16)).T
    if getattr(lin, "bias", None) is not None:
        acc = acc + to_f64(np.asarray(
            lin.bias.astype(mx.bfloat16).view(mx.uint16), dtype=np.uint16))
    return rne_bf16(acc).reshape(-1), acc.reshape(-1), native


def gemv_truth(x_bits, w_bits, w_view):
    x = np.asarray(x_bits)
    w = np.asarray(w_bits)
    xf = to_f64(x).reshape(1, -1)
    wf = to_f64(w).reshape(-1, x.size) if w.ndim == 1 else to_f64(w)
    acc = xf @ wf.T
    return rne_bf16(acc).reshape(-1), acc.reshape(-1)


def stats(bits, truth, truth_f):
    bits = bits.reshape(-1).astype(np.int64)
    truth = truth.reshape(-1).astype(np.int64)
    d = np.abs(bits - truth)
    mm = d > 0
    cancel = np.abs(truth_f) < 1e-3
    return {
        "elements": int(d.size),
        "mismatches": int(mm.sum()),
        "max_bit_distance": int(d.max()),
        "mismatch_median_distance": (float(np.median(d[mm])) if mm.any() else 0.0),
        "mismatch_mean_distance": (float(d[mm].mean()) if mm.any() else 0.0),
        "sign_flips": int(((bits < 0) != (truth < 0)).sum()),
        "cancel_elements_lt1e-3": int(cancel.sum()),
        "cancel_mismatches": int((d[cancel] > 0).sum()),
        "cancel_max_bit_distance": int(d[cancel].max()) if cancel.any() else 0,
        "cancel_median_distance": (float(np.median(d[cancel][d[cancel] > 0]))
                                   if (d[cancel] > 0).any() else 0.0),
    }


def main():
    out_json = Path(sys.argv[1])
    cells = dict(a.split("=", 1) for a in sys.argv[2:])
    model, _ = load(str(Path.home() / "models/Qwen2.5-0.5B-Instruct-bf16-mlx"))
    layer = model.model.layers[0]
    w_gate = np.asarray(layer.mlp.gate_proj.weight.view(mx.uint16), dtype=np.uint16).ravel()
    w_down = np.asarray(layer.mlp.down_proj.weight.view(mx.uint16), dtype=np.uint16)
    w_emb = np.asarray(model.model.embed_tokens.weight.view(mx.uint16), dtype=np.uint16).ravel()

    truths = {}
    native_out = {}
    for phase in ("decode", "prefill"):
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            t = truth_for(f"{phase}.{proj}", model)
            truths[f"{phase}.{proj}"] = t[:2]
            native_out[f"{phase}.{proj}"] = t[2]
    for trial in range(8):
        x896 = patterned_bits(896, 11 + trial * 101)
        x4864 = patterned_bits(4864, 11 + trial * 101)
        truths[f"gate_t{trial}"] = gemv_truth(x896, w_gate, None)
        truths[f"down_t{trial}"] = gemv_truth(x4864, w_down, None)
        truths[f"lmhead_t{trial}"] = gemv_truth(x896, w_emb, None)
    for salt, k, n in ((23, 896, 128), (29, 4864, 896)):
        xb, wb, partial = synthetic_cancel(k, n, salt)
        t = gemv_truth(xb, wb, None)
        truths[f"cancel{k}x{n}"] = t
        np.savez(out_json.with_suffix("").parent / f"cancel{k}x{n}-partials.npz",
                 partial=partial)

    report = {}
    for cell, cdir in cells.items():
        cdir = Path(cdir)
        report[cell] = {}
        for name, (truth, truth_f) in truths.items():
            f = cdir / f"{name}.npz"
            if not f.exists():
                continue
            bits = np.load(f)["result"]
            rec = {"vs_rne_f64": stats(bits, truth, truth_f)}
            if name in native_out:
                nat = native_out[name]
                rec["vs_native_mismatches"] = int(
                    (bits.reshape(-1).astype(np.int64) != nat.astype(np.int64)).sum())
                rec["vs_native_max_distance"] = int(np.abs(
                    bits.reshape(-1).astype(np.int64) - nat.astype(np.int64)).max())
            report[cell][name] = rec

    # v_proj worst element, for the receipt narrative
    truth, truth_f = truths["decode.v_proj"]
    print("native mismatch counts (root-cause reference: q106 k13 v46 o0):")
    for name, nat in native_out.items():
        if name.startswith("decode"):
            row = {c: report[c][name]["vs_native_mismatches"] for c in report}
            print(" ", name, row)
    worst = {}
    if "cand_fork" in cells:
        b = np.load(Path(cells["cand_fork"]) / "decode.v_proj.npz")["result"].reshape(-1).astype(np.int64)
        d = np.abs(b - truth.astype(np.int64))
        i = int(np.argmax(d))
        worst = {"index": i, "truth_f": float(truth_f[i]),
                 "truth_bits": int(truth[i]), "cand_bits": int(b[i]),
                 "distance": int(d[i])}
    out_json.write_text(json.dumps(
        {"cells": report, "decode_v_proj_worst_cand_fork": worst}, indent=2) + "\n")
    print("DONE", out_json)


if __name__ == "__main__":
    main()
