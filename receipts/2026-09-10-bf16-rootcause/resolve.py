#!/usr/bin/env python3
"""One-process resolution: model Linear vs bare nn.Linear vs f64 matmul,
with SHAs of every operand and result."""
import hashlib
import json
import os
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["MLX_DISABLE_COMPILE"] = "1"
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.utils import load

mx.set_default_device(mx.cpu)
MODEL = Path.home() / "models/Qwen2.5-0.5B-Instruct-bf16-mlx"
CAPTURE = Path("/tmp/bf16chain3-native-fixed/import-2/capture")


def sha(a, label):
    mx.eval(a)
    bits = np.asarray(a.view(mx.uint16), dtype=np.uint16)
    h = hashlib.sha256(bits.tobytes()).hexdigest()[:12]
    print(f"  sha[{label}] = {h} dtype={a.dtype} shape={a.shape}")
    return bits


def tensor(name):
    meta = json.loads((CAPTURE / "capture.json").read_text())["tensors"][name]
    data = np.load(CAPTURE / meta["path"])
    return mx.array(np.uint16(data)).view(mx.bfloat16)


x = tensor("decode.q_proj.input")
native = np.load(CAPTURE / "decode_q_proj_output.npy").reshape(-1).astype(np.uint16)
print("input:")
xb = sha(x, "x")

model, _ = load(str(MODEL))
layer = model.model.layers[0]
w_model = layer.self_attn.q_proj.weight
print("model weights:")
wb1 = sha(w_model, "model_q")

out1 = layer.self_attn.q_proj(x)
b1 = sha(out1, "model_result")
m1 = int((np.asarray(b1).reshape(-1) != native).sum())
print(f"  model q_proj vs native: {m1}/896")

raw = mx.load(str(MODEL / "model.safetensors"))["model.layers.0.self_attn.q_proj.weight"]
print("raw weights:")
wb2 = sha(raw, "raw_q")
print("  weights identical:", np.array_equal(np.asarray(wb1), np.asarray(wb2)))

linear = nn.Linear(raw.shape[1], raw.shape[0], bias=False)
linear.weight = raw
out2 = linear(x)
b2 = sha(out2, "bare_linear_result")
m2 = int((np.asarray(b2).reshape(-1) != native).sum())
print(f"  bare nn.Linear vs native: {m2}/896")
print("  model result == bare result:", np.array_equal(np.asarray(b1), np.asarray(b2)))

out3 = x @ raw.T
b3 = sha(out3, "x@W.T")
m3 = int((np.asarray(b3).reshape(-1) != native).sum())
print(f"  x@W.T vs native: {m3}/896")

# f64
x64 = np.float64(np.asarray(x.astype(mx.float32), dtype=np.float32))
w64 = np.float64(np.asarray(raw.astype(mx.float32), dtype=np.float32))
ref = to_bf16 = ((w64 @ x64[0]))
print("f64 first 4:", ref[:4], " native first 4:",
      [(int(n), float((int(n) << 16).view(np.float32) if False else np.float32(np.uint32(int(n) << 16)))) for n in native[:4]])
