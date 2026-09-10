#!/usr/bin/env python3
"""Round 2: test capture self-consistency hypotheses for decode q/k/v."""
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

CAPTURE = Path("/tmp/bf16chain3-native-fixed/import-2/capture")
MODEL = Path.home() / "models/Qwen2.5-0.5B-Instruct-bf16-mlx"


def bits(name):
    meta = json.loads((CAPTURE / "capture.json").read_text())["tensors"][name]
    return np.asarray(np.load(CAPTURE / meta["path"]), dtype=np.uint16)


def to_f32(b):
    return (b.astype(np.uint32) << 16).view(np.float32)


def f64(b):
    return to_f32(b).astype(np.float64)


def load_w():
    w = {}
    for f in sorted(MODEL.glob("model*.safetensors")):
        sd = mx.load(str(f))
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            key = f"model.layers.0.self_attn.{proj}.weight"
            if key in sd:
                w[proj] = f64(np.asarray(sd[key].view(mx.uint16), dtype=np.uint16))
        key = "model.layers.0.input_layernorm.weight"
        if key in sd:
            w["input_norm"] = f64(np.asarray(sd[key].astype(mx.float32).view(mx.uint32), dtype=np.float32))
        if len(w) == 5:
            break
    return w


w = load_w()

# 1. RMSNorm consistency (no mean subtraction)
x = f64(bits("decode.input_norm.input")).reshape(-1)
ref = x / np.sqrt((x ** 2).mean() + 1e-5) * w["input_norm"]
got = f64(bits("decode.input_norm.output")).reshape(-1)
print("decode RMSNorm f64 vs captured: max_abs_delta =", np.abs(ref - got).max())

xp = f64(bits("prefill.input_norm.input")).reshape(262, 896)
refp = xp / np.sqrt((xp ** 2).mean(axis=-1, keepdims=True) + 1e-5) * w["input_norm"]
gotp = f64(bits("prefill.input_norm.output")).reshape(262, 896)
print("prefill RMSNorm f64 vs captured: max_abs_delta =", np.abs(refp - gotp).max())

# 2. What input would produce the captured native decode q output?
q_out = f64(bits("decode.q_proj.output")).reshape(-1)
q_in = f64(bits("decode.q_proj.input")).reshape(-1)
Wq = w["q_proj"]  # (896, 896)
try:
    x_real = np.linalg.solve(Wq, q_out)  # Wq @ x_real = q_out  <=> q_out = Wq @ x_real
    # (Linear computes x @ W.T; out[j] = dot(x, W[j]) for each row j; so out = W @ x)
    print("\nsolve(Wq, q_out): rms =", np.sqrt((x_real ** 2).mean()),
          " captured q_in rms =", np.sqrt((q_in ** 2).mean()))
    print("corr(x_real, q_in) =", np.corrcoef(x_real, q_in)[0, 1])
    print("max|x_real - q_in| =", np.abs(x_real - q_in).max())
except np.linalg.LinAlgError as e:
    print("solve failed:", e)

# 3. Does decode q_out match q_proj on the prefill LAST row?
qp_in = f64(bits("prefill.q_proj.input")).reshape(262, 896)
ref_last = Wq @ qp_in[261]
print("\nf64 q_proj(prefill row 261) vs decode.q_out: max delta =",
      np.abs(ref_last - q_out).max(), " rms delta =",
      np.sqrt(((ref_last - q_out) ** 2).mean()))

# 4. sanity: f64 q_proj on captured decode input vs captured output
ref_q = Wq @ q_in
print("f64 q_proj(captured decode q_in) vs decode.q_out: max delta =",
      np.abs(ref_q - q_out).max(), " corr = ", np.corrcoef(ref_q, q_out)[0, 1])
print("scale ratio median(q_out/ref_q on |ref|>1e-3):",
      np.median(np.abs(q_out[np.abs(ref_q) > 1e-3]) / np.abs(ref_q[np.abs(ref_q) > 1e-3])))

# 5. v/k same solve
for proj in ("v_proj", "k_proj"):
    out = f64(bits(f"decode.{proj}.output")).reshape(-1)
    xin = f64(bits(f"decode.{proj}.input")).reshape(-1)
    W = w[proj]
    x_real = np.linalg.solve(W, out)
    print(f"\nsolve(W{proj}, out): rms={np.sqrt((x_real**2).mean()):.4g} "
          f"in-rms={np.sqrt((xin**2).mean()):.4g} corr={np.corrcoef(x_real, xin)[0,1]:.4f}")
