#!/usr/bin/env python3
"""Per-step greedy top-2 logit margins for the long-prompt workload
(262 prompt tokens, 128 generated) on the default GPU device.

Quantifies whether the historical token-20 divergence (native
254d73fd93164b98 vs Linux 4cc08910089477fd) sits on a near-tie argmax.
Uses the mlx-lm prompt cache path, identical to bench_decode's engine.

Usage (cwd = repo root):
  python3 margin-probe.py MODEL_SNAPSHOT --max-print 40 > margins.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "scripts"))
import bench_matrix

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler
from mlx_lm.utils import load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--prompt-id", default="long")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--max-print", type=int, default=40)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    manifest = json.loads(
        (Path.cwd() / "scripts" / "bench_matrix.json").read_text())
    prompt_text = bench_matrix.prompt_text(manifest, args.prompt_id)

    model, tokenizer = load(args.model)
    mx.random.seed(args.seed)
    sampler = make_sampler(temp=args.temp)

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        add_generation_prompt=True)
    saved_eos = getattr(tokenizer, "eos_token_ids", None)
    tokenizer.eos_token_ids = set()

    cache = make_prompt_cache(model)
    logits = model(mx.array([prompt]), cache=cache)

    steps = []
    for i in range(args.tokens):
        last = logits[0, -1]
        top2 = mx.argpartition(last, -2)[-2:]
        vals = sorted((float(last[int(t)]), int(t)) for t in top2,
                      reverse=True)
        token = sampler(last)
        tid = int(token.item()) if hasattr(token, "item") else int(token)
        steps.append({
            "index": i,
            "token_id": tid,
            "top1_id": vals[0][1], "top1_logit": vals[0][0],
            "top2_id": vals[1][1], "top2_logit": vals[1][0],
            "margin": vals[0][0] - vals[1][0],
        })
        tid_arr = mx.array([tid]) if not hasattr(token, "item") else token
        logits = model(tid_arr[None], cache=cache)

    tokenizer.eos_token_ids = saved_eos
    digest_ids = [s["token_id"] for s in steps]
    import hashlib
    digest = hashlib.sha256(
        ",".join(str(t) for t in digest_ids).encode()).hexdigest()[:16]
    out = {
        "schema": "margin-probe/1",
        "prompt_id": args.prompt_id,
        "prompt_tokens": len(prompt),
        "ids_digest_16": digest,
        "ids": digest_ids,
        "min_margin_step": min(steps, key=lambda s: s["margin"]),
        "steps": steps[:args.max_print],
    }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
