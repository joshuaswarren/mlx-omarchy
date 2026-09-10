#!/usr/bin/env python3
"""Decisive CPU test: does the mlx_lm-loaded model's layer-0 q/k/v Linear
hold the same weight bits as raw safetensors? And does running the model's
own Linear on CPU reproduce the captured output?"""
import hashlib
import json
from pathlib import Path
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["MLX_DISABLE_COMPILE"] = "1"

import mlx.core as mx
import numpy as np
import sys

sys.path.insert(0, str(Path.home() / "venv-bf16chain3/lib/python3.14/site-packages"))
from mlx_lm.utils import load

mx.set_default_device(mx.cpu)
MODEL = Path.home() / "models/Qwen2.5-0.5B-Instruct-bf16-mlx"
CAPTURE = Path("/tmp/bf16chain3-native-fixed/import-2/capture")


def sha_bits(a):
    mx.eval(a)
    bits = np.asarray(a.view(mx.uint16), dtype=np.uint16)
    return hashlib.sha256(bits.tobytes()).hexdigest(), bits


model, _ = load(str(MODEL))
layer = model.model.layers[0]

for f in sorted(MODEL.glob("model*.safetensors")):
    sd = mx.load(str(f))
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        key = f"model.layers.0.self_attn.{proj}.weight"
        raw = sd[key]
        held = getattr(layer.self_attn, proj).weight
        s_raw, b_raw = sha_bits(raw.astype(mx.bfloat16))
        s_held, b_held = sha_bits(held)
        print(f"{proj}: raw={s_raw[:12]} held={s_held[:12]} equal={s_raw == s_held} "
              f"raw_dtype={raw.dtype} held_dtype={held.dtype} raw_shape={raw.shape}")

    # run the model's own Linear on CPU with the captured input and compare
    meta = json.loads((CAPTURE / "capture.json").read_text())["tensors"]["decode.q_proj.output"]

    def tensor(name):
        m = json.loads((CAPTURE / "capture.json").read_text())["tensors"][name]
        data = np.load(CAPTURE / m["path"])
        return mx.array(np.uint16(data)).view(mx.bfloat16)

    x = tensor("decode.q_proj.input")
    out = layer.self_attn.q_proj(x)
    got = np.asarray(out.view(mx.uint16), dtype=np.uint16).reshape(-1)
    exp = np.asarray(
        mx.array(np.uint16(np.load(CAPTURE / meta["path"]))).view(mx.uint16)
        if False else np.load(CAPTURE / meta["path"]).reshape(-1), dtype=np.uint16)
    d = np.abs(got.astype(np.int64) - exp.astype(np.int64))
    print(f"model q_proj(captured input) CPU vs native capture: "
          f"mismatch={int((got != exp).sum())}/{got.size} max_bit_delta={int(d.max())}")
    break
