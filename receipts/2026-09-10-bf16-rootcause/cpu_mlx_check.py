#!/usr/bin/env python3
"""Third opinion: run the same ops through mlx CPU backend on the M1
and compare to the captured (macOS GPU) outputs."""
import json
from pathlib import Path

os_free = Path("/tmp/bf16chain3-native-fixed/import-2/capture")
MODEL = Path.home() / "models/Qwen2.5-0.5B-Instruct-bf16-mlx"

import mlx.core as mx
import numpy as np

mx.set_default_device(mx.cpu)
CAPTURE = os_free


def tensor(name):
    meta = json.loads((CAPTURE / "capture.json").read_text())["tensors"][name]
    data = np.load(CAPTURE / meta["path"])
    if meta["encoding"] == "raw-bfloat16-bits":
        return mx.array(np.uint16(data)).view(mx.bfloat16)
    if meta["encoding"] == "raw-float16-bits":
        return mx.array(np.uint16(data)).view(mx.float16)
    return mx.array(data)


def bits_of(value):
    return np.asarray(value.view(mx.uint16), dtype=np.uint16)


def compare(name, result):
    got = bits_of(result).reshape(-1)
    exp = bits(tensor(name + ".output")).reshape(-1)
    d = np.abs(got.astype(np.int64) - exp.astype(np.int64))
    print(f"{name}: cpu-vs-native mismatches={int((got != exp).sum())}/{got.size} "
          f"max_bit_delta={int(d.max())}")
    return got


layer_w = {}
for f in sorted(MODEL.glob("model*.safetensors")):
    sd = mx.load(str(f))
    for k in sd:
        if ".layers.0." in k and ("self_attn" in k or "mlp" in k or "layernorm" in k):
            layer_w[k] = sd[k]
    if len(layer_w) >= 12:
        break
print("weight files:", [f.name for f in sorted(MODEL.glob('model*.safetensors'))])
print("layer0 keys sample:", sorted(layer_w)[:4], "...", len(layer_w), "keys")

# RMSNorm via mlx CPU
norm_in = tensor("decode.input_norm.input")
w = layer_w["model.layers.0.input_layernorm.weight"]
import mlx.nn as nn
rms = nn.RMSNorm(w.shape[0], 1e-5)
rms.weight = w
compare("decode.input_norm", rms(norm_in))

for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
    x = tensor(f"decode.{proj}.input")
    wk = layer_w[f"model.layers.0.self_attn.{proj}.weight"]
    linear = nn.Linear(wk.shape[1], wk.shape[0], bias=False)
    linear.weight = wk
    got = compare(f"decode.{proj}", linear(x))
    # also raw f64 for the same op
    x64 = np.float64(np.asarray(x.astype(mx.float32), dtype=np.float32))
    w64 = np.float64(np.asarray(wk.astype(mx.float32), dtype=np.float32))
    ref = (x64 @ w64.T)
    ref_bf = ((ref.astype(np.float32).view(np.uint32) + 0x7FFF + ((ref.astype(np.float32).view(np.uint32) >> 16) & 1)) >> 16).astype(np.uint16).reshape(-1)
    d2 = np.abs(got.astype(np.int64) - ref_bf.astype(np.int64))
    print(f"    cpu-vs-f64: mismatches={int((got != ref_bf).sum())}/{got.size} max_bit_delta={int(d2.max())}")
