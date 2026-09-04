#!/usr/bin/env python3
"""Decompose the per-layer KV-cache copy count in a Qwen2 decode step.

Runs one transformer block's decode-path components in isolation under
MLX_OMARCHY_GPU_PROFILE and reports CopyGeneralF16 dispatch sizes for
each stage, so every copy is attributed to the graph op that caused it:

  stage A: cache.update_and_fetch(k, v) alone
  stage B: A + attention over the fetched (sliced) k/v (sdpa)
  stage C: B with rope on q and k before the update (full step)

Usage (profile path fixed to /tmp/kvcopy-prof.jsonl):
  MLX_OMARCHY_ALLOW_NON_APPLE=1 MLX_OMARCHY_GPU_PROFILE=/tmp/kvcopy-prof.jsonl \
    python3 scripts/kvcopy_decompose.py --model PATH
"""

import argparse
import json
import re
import time
from collections import Counter

import mlx.core as mx


def load_model(path):
    from mlx_lm.utils import load

    return load(path)


class Marker:
    def __init__(self, path):
        self.f = open(path, "w") if path else None

    def mark(self, phase):
        if self.f:
            self.f.write(
                json.dumps({"t": time.monotonic_ns(), "p": phase}) + "\n")
            self.f.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-tokens", type=int, default=7)
    ap.add_argument("--decode-steps", type=int, default=8)
    ap.add_argument("--markers", default="/tmp/kvcopy.markers.jsonl")
    args = ap.parse_args()

    mk = Marker(args.markers)
    model, tokenizer = load_model(args.model)
    layer = model.model.layers[0]
    attn = layer.self_attn
    from mlx_lm.models import cache as cache_mod

    # ---- prefill one real cache to decode against ----
    tokens = mx.arange(args.prompt_tokens)[None]
    caches = [cache_mod.KVCache() for _ in range(model.args.num_hidden_layers)]
    h = model.model.embed_tokens(tokens)
    for i, lay in enumerate(model.model.layers):
        h = lay(h, mask=None, cache=caches[i])
    mx.eval(h)

    c = cache_mod.KVCache()
    c.keys = caches[0].keys
    c.values = caches[0].values
    c.offset = caches[0].offset

    def attn_step(x_embed, stage):
        B, L, D = x_embed.shape
        q = attn.q_proj(x_embed)
        k = attn.k_proj(x_embed)
        v = attn.v_proj(x_embed)
        q = q.reshape(B, L, attn.n_heads, -1).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
        if stage >= 2:
            q = attn.rope(q, offset=c.offset)
            k = attn.rope(k, offset=c.offset)
        if stage >= 1:
            k, v = c.update_and_fetch(k, v)
        if stage >= 2:
            out = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=attn.scale, mask=None)
            return attn.o_proj(
                out.transpose(0, 2, 1, 3).reshape(B, L, -1))
        return k.sum() + v.sum()

    x = model.model.embed_tokens(mx.array([[args.prompt_tokens]]))

    labels = {0: "A-update-only", 1: "B-update+sdpa", 2: "C-full-step"}
    for stage in (0, 1, 2):
        # restore the prefilled state for every stage
        c.keys = caches[0].keys
        c.values = caches[0].values
        c.offset = caches[0].offset
        mk.mark("decode_start")
        for _ in range(args.decode_steps):
            mk.mark("tok")
            out = attn_step(x, stage)
            mx.eval(out)
            c.offset += 1
        mk.mark("decode_done")

    if mk.f:
        mk.f.close()

    # ---- analyze the profile stream ----
    names = []
    pat = re.compile(r"^\s*([A-Za-z0-9_]+)\s*,?\s*$")
    inside = False
    for line in open(".work/mlx/mlx/backend/omarchy/compute.h"):
        if "enum class ComputeKernel" in line:
            inside = True
            continue
        if inside:
            if "}" in line:
                break
            m = pat.match(line)
            if m and m.group(1) != "Count":
                names.append(m.group(1))

    sub_t = {}
    events = []
    for line in open("/tmp/kvcopy-prof.jsonl"):
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("k") == "s":
            sub_t[ev["s"]] = ev["t"]
    for line in open("/tmp/kvcopy-prof.jsonl"):
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("k") == "d":
            events.append(ev)
    starts, dones, ts = [], [], []
    for line in open(args.markers):
        ev = json.loads(line)
        if ev["p"] == "decode_start":
            starts.append(ev["t"])
        elif ev["p"] == "decode_done":
            dones.append(ev["t"])
        elif ev["p"] == "tok":
            ts.append(ev["t"])

    for s in range(3):
        lo, hi = starts[s], dones[s]
        nsteps = sum(1 for t in ts if lo <= t < hi)
        window = [ev for ev in events if lo <= sub_t.get(ev["s"], 0) < hi]
        copies = Counter()
        for ev in window:
            if names[ev["e"]] == "CopyGeneralF16":
                copies[ev["n"]] += 1
        per_tok = {n: v // max(nsteps, 1) for n, v in copies.items()}
        print(f"{labels[s]}: CopyGeneralF16 per step = "
              f"{dict(sorted(per_tok.items()))} "
              f"total={sum(copies.values()) // max(nsteps, 1)}")


if __name__ == "__main__":
    main()
