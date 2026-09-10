#!/usr/bin/env python3
"""Pin the exact worst v_proj element: bits, f64 acc, bias, Metal value."""
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
    return np.asarray(np.load(CAPTURE / meta["path"]), dtype=np.uint16)


model, _ = load(str(MODEL))
layer = model.model.layers[0]
lin = layer.self_attn.v_proj
xb = tensor("decode.v_proj.input")
native = tensor("decode.v_proj.output").reshape(-1)
Wb = np.asarray(lin.weight.view(mx.uint16), dtype=np.uint16)
bb = np.asarray(lin.bias.astype(mx.bfloat16).view(mx.uint16), dtype=np.uint16)

xf = to_f32(xb).astype(np.float64).reshape(-1, xb.shape[-1])
Wf = to_f32(Wb).astype(np.float64)
bf = to_f32(bb).astype(np.float64)
acc = (xf @ Wf.T).reshape(-1)
truth_f = acc + bf
truth = rne_bf16(truth_f)
d = np.abs(native.astype(np.int64) - truth.astype(np.int64))
i = int(np.argmax(d))
print(f"element {i}:")
print(f"  f64 acc (no bias) = {acc[i]:.10g}")
print(f"  f64 bias          = {bf[i]:.10g}")
print(f"  f64 truth         = {truth_f[i]:.10g} -> bf16 {truth[i]:#06x}")
print(f"  metal bits        = {native[i]:#06x} = {float(to_f32(native[i:i+1])[0]):.10g}")
print(f"  bit distance      = {int(d[i])}")
# top-5 for context
for j in np.argsort(-d)[:5]:
    print(f"  [{j}] acc={acc[j]:+.6g} bias={bf[j]:+.6g} truth={truth[j]:#06x} "
          f"metal={native[j]:#06x}({float(to_f32(native[j:j+1])[0]):+.6g}) d={int(d[j])}")
# how many elements have |truth_f| < 0.001 and mismatch
small = np.abs(truth_f) < 0.001
print(f"elements with |truth|<1e-3: {int(small.sum())}, mismatched among them: "
      f"{int((native[small] != truth[small]).sum())}")
mm = native != truth
print(f"total mismatched: {int(mm.sum())}; |truth|<0.005 among mismatched: "
      f"{int((np.abs(truth_f[mm]) < 0.005).sum())}")
