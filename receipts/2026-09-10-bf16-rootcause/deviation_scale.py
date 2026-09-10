#!/usr/bin/env python3
"""Characterize Metal's deviation on decode.v_proj: for the deviating
elements, compare deviation against partial-sum scale to identify the
accumulation precision Metal used."""
import json
import os
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["MLX_DISABLE_COMPILE"] = "1"
import mlx.core as mx
import numpy as np
from mlx_lm.utils import load

mx.set_default_device(mx.cpu)
MODEL = Path.home() / "models/Qwen2.5-0.5B-Instruct-bf16-mlx"
CAPTURE = Path("/tmp/bf16chain3-native-fixed/import-2/capture")


def to_f32(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def tensor(name):
    meta = json.loads((CAPTURE / "capture.json").read_text())["tensors"][name]
    return np.asarray(np.load(CAPTURE / meta["path"]), dtype=np.uint16)


model, _ = load(str(MODEL))
layer = model.model.layers[0]

for proj in ("v_proj", "k_proj", "q_proj"):
    xb = tensor(f"decode.{proj}.input")
    native = tensor(f"decode.{proj}.output").reshape(-1)
    lin = getattr(layer.self_attn, proj)
    Wb = np.asarray(lin.weight.view(mx.uint16), dtype=np.uint16)
    xf = to_f32(xb).astype(np.float64).reshape(-1, xb.shape[-1])
    Wf = to_f32(Wb).astype(np.float64)
    acc = xf @ Wf.T[0:0] if False else xf @ Wf.T
    bb = np.asarray(lin.bias.astype(mx.bfloat16).view(mx.uint16), dtype=np.uint16)
    bias = to_f32(bb).astype(np.float64)
    truth = acc + bias

    def val(bits):
        return to_f32(bits.reshape(-1)).astype(np.float64)

    nv = val(native)
    dev = np.abs(nv - truth).reshape(-1)
    idx = np.argsort(-dev)[:6]
    print(f"\ndecode.{proj}: worst Metal deviations vs f64 truth")
    for i in idx:
        if dev[i] == 0:
            break
        # scales: |acc| (pre-bias dot), max |x_k*w_k| term, sum|terms|
        terms = np.abs(xf[0] * Wf[i])
        print(f"  [{i}] truth={truth.reshape(-1)[i]:+.6g} metal={nv[i]:+.6g} "
              f"|acc|={abs(acc.reshape(-1)[i]):.4g} sum|terms|={terms.sum():.4g} "
              f"maxterm={terms.max():.4g} dev={dev[i]:.4g} dev/sum|terms|={dev[i]/terms.sum():.3g}")
