#!/usr/bin/env python3
# No-cache greedy decode: mlx-lm bf16, compile ON, cache=None — every
# step re-runs the FULL sequence through the model, so no KV cache
# exists. If garbage disappears here vs the cached run, the KV cache
# path is implicated in the bf16 tape defect.
import os
import sys

import mlx.core as mx
import numpy as np


def banner(m):
    sys.stderr.write(f"[bf16tape_nocache] {m}\n")
    sys.stderr.flush()


def main():
    n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    from mlx_lm.utils import load
    model, tokenizer = load("/home/joshuawarren/models/Qwen2.5-0.5B-Instruct-bf16-mlx")
    messages = [{"role": "user", "content": "Hi"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)

    seq = list(prompt)
    banner(f"prompt tokens={len(prompt)} DISABLE={os.environ.get('MLX_DISABLE_COMPILE')}")
    for step in range(n_steps):
        logits = model(mx.array([seq]), cache=None)
        mx.eval(logits)
        t = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        seq.append(t)
        banner(f"step {step}: token {t} {tokenizer.decode([t])!r}")
    text = tokenizer.decode(seq[len(prompt):])
    banner(f"TEXT: {text!r}")


if __name__ == "__main__":
    main()