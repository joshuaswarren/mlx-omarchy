#!/usr/bin/env python3
"""Qwen3.5-9B prefill A/B harness. Run with --venv-python <path> --label <name>."""
import argparse, json, os, statistics, sys, time, hashlib

ap = argparse.ArgumentParser()
ap.add_argument("--venv-python", required=True)
ap.add_argument("--label", required=True)
args = ap.parse_args()

import mlx.core as mx
import mlx_lm
from mlx_lm.utils import load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler, make_logits_processors

label = args.label
mlx_v = mx.__version__
mlxlm_v = getattr(mlx_lm, "__version__", "?")

model_path = ("~/.cache/huggingface/hub/"
              "models--mlx-community--Qwen3.5-9B-MLX-4bit/snapshots/"
              "938d8919941c6e7efd3c7150eff7fe9d12afa631")
import os
model_path = os.path.expanduser(model_path)

t0 = time.perf_counter()
model, tok = load(model_path)
mx.eval(model.parameters())
load_s = time.perf_counter() - t0

prompt = ("The history of computing begins with early mechanical "
           "calculators, telescopes, and clocks. Today the field "
           "continues to evolve with machine learning, specialized "
           "accelerators, and a growing emphasis on privacy, open "
           "weights, and local execution. Open source communities "
           "publish reproducible code and reproducible measurements.")
ids = tok.encode(prompt)
while len(ids) < 4096: ids = ids + ids
ids_512 = ids[:512]
ids_2048 = ids[:2048]

sampler = make_sampler(temp=0.0, top_p=1.0)
lps = make_logits_processors(repetition_penalty=1.1)

mx.eval(model(mx.array(ids_512[:16])[None]))
mx.synchronize()

def prefill_n(seq_len, n=3):
    times = []
    for _ in range(n):
        s = ids_512 if seq_len == 512 else ids_2048
        t0 = time.perf_counter()
        out = model(mx.array(s)[None])
        mx.eval(out)
        mx.synchronize()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)

t512 = prefill_n(512, 3)
t2048 = prefill_n(2048, 3)

sys_p = "You are a careful assistant."
user = "Describe the lifecycle of a software bug from report to fix in one paragraph."

def ttft_n(n=3):
    times = []
    for _ in range(n):
        msgs = [{"role":"system","content":sys_p},{"role":"user","content":user}]
        prompt_str = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        p_ids = tok.encode(prompt_str)
        t0 = time.perf_counter()
        for tok_id, _ in generate_step(mx.array(p_ids), model, max_tokens=256,
                                        sampler=sampler, logits_processors=lps):
            mx.eval(tok_id)
            mx.synchronize()
            times.append(time.perf_counter() - t0)
            break
    return statistics.median(times)

tftt = ttft_n(3)

def decode_n(tokens=32, reps=3):
    per_token = []
    for _ in range(reps):
        gen = generate_step(mx.array(ids_512), model, max_tokens=tokens,
                            sampler=sampler, logits_processors=lps)
        toks = []
        t0 = time.perf_counter()
        for tok_id, _ in gen:
            mx.eval(tok_id)
            toks.append(tok_id.item())
            if len(toks) == tokens: break
        t_end = time.perf_counter()
        per_token.append((t_end - t0) / max(1, len(toks)-1))
    return statistics.median(per_token)

dec_per_tok = decode_n(32, 3)
decode_tps = 1.0 / dec_per_tok

def greedy_32():
    gen = generate_step(mx.array(ids_512), model, max_tokens=32,
                        sampler=sampler, logits_processors=lps)
    toks = []
    for tok_id, _ in gen:
        mx.eval(tok_id)
        toks.append(tok_id.item())
        if len(toks) == 32: break
    return toks

greedy_toks = greedy_32()
greedy_text = tok.decode(greedy_toks)
greedy_hash = hashlib.sha256(bytes(greedy_toks)).hexdigest()[:16]

result = {
    "label": label,
    "mlx": mlx_v, "mlx_lm": mlxlm_v, "load_s": round(load_s, 2),
    "prefill_512_tok_per_s": round(512/t512, 1),
    "prefill_512_wall_s": round(t512, 3),
    "prefill_2048_tok_per_s": round(2048/t2048, 1),
    "prefill_2048_wall_s": round(t2048, 3),
    "ttft_sys_256tok_s": round(tftt, 3),
    "decode_tok_per_s": round(decode_tps, 1),
    "greedy_32_hash": greedy_hash,
    "greedy_32_text": greedy_text,
}
print("===JSON===")
print(json.dumps(result, indent=2))
