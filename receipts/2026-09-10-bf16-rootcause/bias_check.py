#!/usr/bin/env python3
"""Bias-corrected root cause: f64 truth (x@W.T + b, rounded RNE to bf16)
vs (a) native Metal capture, (b) CPU mlx. Reveals which accumulation
order is closer to truth and where the Linux-vs-Metal 106/13/46 mismatches
actually come from."""
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


def rne_bf16(a):
    b = np.asarray(a, dtype=np.float32).view(np.uint32)
    return ((b + 0x7FFF + ((b >> 16) & 1)) >> 16).astype(np.uint16)


def tensor(name):
    meta = json.loads((CAPTURE / "capture.json").read_text())["tensors"][name]
    data = np.load(CAPTURE / meta["path"])
    return np.asarray(data, dtype=np.uint16)


model, _ = load(str(MODEL))
layer = model.model.layers[0]

print(f"{'op':22s} {'|native-f64|':>14s} {'|cpu-f64|':>14s} "
      f"{'native==rne(f64)':>18s} {'cpu==rne(f64)':>14s} {'native==cpu':>12s}")
for phase in ("decode", "prefill"):
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        name = f"{phase}.{proj}"
        x = tensor(name + ".input")
        native = tensor(name + ".output").reshape(-1)
        W = np.asarray(layer.self_attn, ).__getattr__(proj).weight if False else getattr(layer.self_attn, proj).weight
        Wb = np.asarray(W.view(mx.uint16), dtype=np.uint16)
        bias = getattr(layer.self_attn, proj)
        has_bias = getattr(bias, "bias", None) is not None
        xf = to_f32(x).astype(np.float64).reshape(x.shape[0], -1)
        Wf = to_f32(Wb).astype(np.float64)
        acc = xf @ Wf.T
        if has_bias:
            bb = np.asarray(has_bias.view(mx.uint16), dtype=np.uint16) if False else np.asarray(
                bias.bias.astype(mx.bfloat16).view(mx.uint16), dtype=np.uint16)
            acc = acc + to_f32(bb).astype(np.float64)
        truth = rne_bf16(acc).reshape(-1)

        # CPU mlx with full model module (bias included)
        cpu_out = getattr(layer.self_attn, proj)(mx.array(np.uint16(x)).view(mx.bfloat16))
        mx.eval(cpu_out)
        cpu = np.asarray(cpu_out.view(mx.uint16), dtype=np.uint16).reshape(-1)

        dn = np.abs(native.astype(np.int64) - truth.astype(np.int64))
        dc = np.abs(cpu.astype(np.int64) - truth.astype(np.int64))
        dm = np.abs(native.astype(np.int64) - cpu.astype(np.int64))
        print(f"{name:22s} max={dn.max():6d} n>1={int((dn>1).sum()):6d} "
              f"max={dc.max():6d} n>1={int((dc>1).sum()):6d} "
              f"eq={int((native==truth).sum()):6d} "
              f"eq={int((cpu==truth).sum()):6d} "
              f"max={dm.max():6d}")
