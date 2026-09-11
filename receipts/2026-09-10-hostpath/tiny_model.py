#!/usr/bin/env python3
"""Build a dispatch-identical, GPU-trivial Qwen2.5-0.5B-shaped tiny model.

Method-B instrument for receipts/2026-09-10-hostpath: the same layer
count, op classes, and dispatch sequence as the pinned q4 0.5B model,
with every per-op payload shrunk to a few workgroups, so a decode token
measures the HOST-paced floor (Python + graph walk + record + submit +
bookkeeping) plus near-zero GPU work. Identity of dispatch count with
the real model is what makes wall(tiny) vs wall(real) a GPU-side vs
host-side split without any GPU timestamp.

Usage: tiny_model.py --real <real-snapshot-dir> --out <tiny-model-dir>
The tiny dir gets the real tokenizer files (same chat template, same
30-token prompt) and a randomly-initialized 4-bit group-64 model.
"""
import argparse
import json
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models import qwen2

    real = Path(args.real)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((real / "config.json").read_text())
    cfg.update({
        "hidden_size": 64,
        "intermediate_size": 172,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "intermediate_size": 256,
    })
    args_ns = qwen2.ModelArgs.from_dict(cfg)
    model = qwen2.Model(args_ns)
    mx.eval(model.parameters())
    # Same quantization recipe as the pinned model: affine 4-bit group-64.
    nn.quantize(model, group_size=64, bits=4)
    cfg["quantization"] = {"group_size": 64, "bits": 4}
    model.save_weights(str(out / "model.safetensors"))
    (out / "config.json").write_text(json.dumps(cfg, indent=1) + "\n")
    for name in ["tokenizer.json", "tokenizer_config.json",
                 "vocab.json", "merges.txt", "special_tokens_map.json",
                 "generation_config.json"]:
        src = real / name
        if src.exists():
            shutil.copy(src, out / name)
    print("tiny model written to", out)


if __name__ == "__main__":
    main()
