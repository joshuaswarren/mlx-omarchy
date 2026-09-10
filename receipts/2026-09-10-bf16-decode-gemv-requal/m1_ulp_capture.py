#!/usr/bin/env python3
"""One GPU cell of the BF16 GEMV ULP capture: the installed wheel x the
caller-chosen driver. Dumps raw bf16 output bits per op-set.

Op-sets:
- decode/prefill x {q,k,v,o}_proj: REAL captured inputs (native-fixed capture,
  capture.json sha256 2defcc6a...) through the nn.Linear modules (bias
  included), layer 0 weights - exact continuation of the root-cause chain.
- synthetic large-N: gate/up (K=896,N=4864), down (K=4864,N=896),
  lm_head (K=896,N=151936) - real weights, patterned x (fixed_bf16.py
  generator), the scope the candidate adds over the old N>=4096-only vec path.
- cancel896x128 / cancel4864x896: deep-cancellation patterned sets.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MLX_DISABLE_COMPILE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

import mlx.core as mx
from mlx_lm.utils import load
from ulp_common import (MODEL, capture_tensor, patterned_bits, synthetic_cancel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--trials", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model, _ = load(str(MODEL))
    layer = model.model.layers[0]
    meta = {"mlx_version": mx.__version__,
            "driver_env": os.environ.get("VK_DRIVER_FILES", "fork-default"),
            "sets": {}}

    def run_set(name, fn):
        bits = fn()
        np.savez(out / f"{name}.npz", result=bits)
        meta["sets"][name] = {
            "shape": list(bits.shape),
            "sha256": __import__("hashlib").sha256(bits.tobytes()).hexdigest(),
        }
        print(name, bits.shape, flush=True)

    for phase in ("decode", "prefill"):
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            x = capture_tensor(f"{phase}.{proj}.input")
            xb = mx.array(x).view(mx.bfloat16)
            run_set(f"{phase}.{proj}",
                    lambda xb=xb, proj=proj: np.asarray(
                        getattr(layer.self_attn, proj)(xb).view(mx.uint16),
                        dtype=np.uint16))

    mlp = layer.mlp
    emb = model.model.embed_tokens

    def gemv(x_bits, w_bits):
        y = mx.array(x_bits).view(mx.bfloat16) @ mx.array(w_bits).view(mx.bfloat16).T
        mx.eval(y)
        return np.asarray(y.view(mx.uint16), dtype=np.uint16)

    w_gate = np.asarray(mlp.gate_proj.weight.view(mx.uint16), dtype=np.uint16)
    w_down = np.asarray(mlp.down_proj.weight.view(mx.uint16), dtype=np.uint16)
    w_emb = np.asarray(emb.weight.view(mx.uint16), dtype=np.uint16)
    for trial in range(args.trials):
        x896 = patterned_bits(896, 11 + trial * 101)
        x4864 = patterned_bits(4864, 11 + trial * 101)
        run_set(f"gate_t{trial}", lambda x=x896: gemv(x, w_gate))
        run_set(f"down_t{trial}", lambda x=x4864: gemv(x, w_down))
        run_set(f"lmhead_t{trial}", lambda x=x896: gemv(x, w_emb))

    for salt, k, n in ((23, 896, 128), (29, 4864, 896)):
        xb, wb, _ = synthetic_cancel(k, n, salt)
        run_set(f"cancel{k}x{n}", lambda xb=xb, wb=wb: gemv(xb, wb))

    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
