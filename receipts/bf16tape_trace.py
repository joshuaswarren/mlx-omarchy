#!/usr/bin/env python3
# bf16 differential activation trace — mlx-lm Qwen2.5-0.5B bf16 on jwm1.
# M1Bf16Tape, 2026-09-02. No commits.
#
# Runs the model forward on the same prompt twice (compile ON vs
# MLX_DISABLE_COMPILE=1), dumping each transformer block's output tensor
# to .npy so they can be diffed offline. Compares only the FIRST generated
# token to stay deterministic.
#
# Usage:
#   mode=trace MLX_DISABLE_COMPILE=1 python bf16tape_trace.py /tmp/base.npz
#   mode=trace python bf16tape_trace.py /tmp/compiled.npz
#   mode=diff   python bf16tape_trace.py /tmp/base.npz /tmp/compiled.npz
import os
import sys

import mlx.core as mx
import numpy as np

MODEL = "/home/joshuawarren/models/Qwen2.5-0.5B-Instruct-bf16-mlx"
PROMPT = "Hi"


def banner(msg):
    sys.stderr.write(f"[bf16tape_trace] {msg}\n")
    sys.stderr.flush()


def load_model_and_tokenizer():
    from mlx_lm.utils import load
    model, tokenizer = load(MODEL)
    return model, tokenizer


def trace(path):
    from mlx_lm.models import cache as lmcache
    model, tokenizer = load_model_and_tokenizer()

    # Tokenize with chat template so the first-token input matches mlx-lm's
    # default greedy path (the receipts' command uses the default template).
    if tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": PROMPT}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True)
    else:
        prompt = tokenizer.encode(PROMPT)
    banner(f"prompt tokens = {len(prompt)} first5={prompt[:5]}")

    # Hook every transformer block: record the block output after each runs.
    layers = model.model.layers
    captured = [None] * len(layers)
    originals = [l.__call__ for l in layers]

    import types
    def make_hook(idx, orig):
        def call(x, mask=None, cache=None):
            out = orig(x, mask, cache)
            mx.eval(out)
            captured[idx] = np.asarray(out.astype(mx.float32))
            return out
        return call

    for i, layer in enumerate(layers):
        layer.__call__ = make_hook(i, layer.__call__)

    # Build the prompt cache the way mlx-lm does.
    cache = lmcache.make_prompt_cache(model)
    input_array = mx.array(prompt)[None]

    logits = model(input_array, cache=cache)
    mx.eval(logits)
    logits_np = np.asarray(logits.astype(mx.float32))
    last_logits = logits[:, -1, :]
    token = int(mx.argmax(last_logits, axis=-1).item())
    banner(f"first token = {token} decoded={tokenizer.decode([token])!r}")

    np.savez(path,
             logits=logits_np,
             token=np.int64(token),
             **{f"layer_{i:02d}": captured[i] for i in range(len(layers))})
    banner(f"wrote {path} with {len(layers)} layer dumps")
    banner(f"MLX_DISABLE_COMPILE={os.environ.get('MLX_DISABLE_COMPILE')}")


def diff(base_path, compiled_path):
    base = np.load(base_path, allow_pickle=True)
    compiled = np.load(compiled_path, allow_pickle=True)
    banner(f"base first token = {int(base['token'])}")
    banner(f"compiled first token = {int(compiled['token'])}")

    n_layers = len(model.model.layers) if False else max(
        int(k.split("_")[1]) for k in base.files if k.startswith("layer_")) + 1
    first_bad = None
    for i in range(n_layers):
        key = f"layer_{i:02d}"
        a = base[key]
        b = compiled[key]
        if a.shape != b.shape:
            banner(f"layer {i}: SHAPE differs base={a.shape} compiled={b.shape}")
            first_bad = i
            break
        d = (a != b)
        diff_count = int(d.sum())
        if diff_count > 0:
            first_idx = tuple(int(v) for v in idx[0].tolist())
            banner(
                f"layer {i}: FIRST divergence at index {first_idx} "
                f"count={diff_count}/{a.size} "
                f"base={a[first_idx]!r} compiled={b[first_idx]!r}")
            if first_bad is None:
                first_bad = i
            break
    if first_bad is None:
        banner("ALL LAYERS MATCH at first token.")
    else:
        banner(f"first diverging layer = {first_bad}")
    a = base["logits"]
    b = compiled["logits"]
    d = (a != b)
    diff_count = int(d.sum()) - int((np.isnan(a) & ~np.isnan(b)).sum())
    banner(f"logits diff_count = {diff_count}/{a.size}")


def trace_decode(path, n_steps):
    from mlx_lm.models import cache as lmcache
    model, tokenizer = load_model_and_tokenizer()

    if tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": PROMPT}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True)
    else:
        prompt = tokenizer.encode(PROMPT)
    banner(f"prompt tokens = {len(prompt)} first5={prompt[:5]}")

    layers = model.model.layers
    captured = []
    step_state = {"i": -1}

    def make_hook(idx, orig):
        def call(x, mask=None, cache=None):
            out = orig(x, mask, cache)
            mx.eval(out)
            s = step_state["i"]
            if s >= 0:
                while len(captured) <= s:
                    captured.append([None] * len(layers))
                captured[s][idx] = np.asarray(out.astype(mx.float32))
            return out
        return call

    for i, layer in enumerate(layers):
        layer.__call__ = make_hook(i, layer.__call__)

    cache = lmcache.make_prompt_cache(model)
    input_array = mx.array(prompt)[None]

    tokens = []
    for step in range(n_steps):
        step_state["i"] = step
        logits = model(input_array, cache=cache)
        mx.eval(logits)
        last_logits = logits[:, -1, :]
        token = int(mx.argmax(last_logits, axis=-1).item())
        tokens.append(token)
        input_array = mx.array([[token]])

    banner(f"tokens = {tokens}")
    banner(f"decoded = {tokenizer.decode(tokens)!r}")
    arrays = {}
    for s in range(len(captured)):
        for i in range(len(layers)):
            arrays[f"step{s:02d}_layer{i:02d}"] = captured[s][i]
    np.savez(path, tokens=np.array(tokens, dtype=np.int64), **arrays)
    banner(f"wrote {path} with {len(captured)} steps x {len(layers)} layers")
    banner(f"MLX_DISABLE_COMPILE={os.environ.get('MLX_DISABLE_COMPILE')}")


def diff_decode(base_path, compiled_path):
    base = np.load(base_path, allow_pickle=True)
    compiled = np.load(compiled_path, allow_pickle=True)
    base_tokens = base["tokens"].tolist()
    comp_tokens = compiled["tokens"].tolist()
    banner(f"base tokens     = {base_tokens}")
    banner(f"compiled tokens = {comp_tokens}")

    first_bad = None
    for s in range(min(len(base_tokens), len(comp_tokens))):
        layer_keys = sorted(
            k for k in base.files if k.startswith(f"step{s:02d}_layer"))
        for key in layer_keys:
            a = base[key]
            b = compiled[key]
            if a.shape != b.shape:
                banner(f"{key}: SHAPE differs base={a.shape} compiled={b.shape}")
                first_bad = (s, int(key.split("_layer")[1]))
                break
            d = (a != b)
            diff_count = int(d.sum())
            if diff_count > 0:
                idx = np.argwhere(d)
                first_idx = tuple(int(v) for v in idx[0].tolist())
                banner(
                    f"{key}: FIRST divergence at index {first_idx} "
                    f"count={diff_count}/{a.size} "
                    f"base={a[first_idx]!r} compiled={b[first_idx]!r}")
                first_bad = (s, int(key.split("_layer")[1]))
                break
        if first_bad is not None:
            break
    if first_bad is None:
        banner("ALL STEPS ALL LAYERS MATCH.")
    else:
        banner(f"first diverging (step, layer) = {first_bad}")


if __name__ == "__main__":
    if sys.argv[1] == "trace_decode":
        trace_decode(sys.argv[2], int(sys.argv[3]))
    elif sys.argv[1] == "diff_decode":
        diff_decode(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "trace":
        trace(sys.argv[2])
    elif sys.argv[1] == "diff":
        diff(sys.argv[2], sys.argv[3])
    else:
        sys.stderr.write("usage: bf16tape_trace.py trace OUT.npz | diff A.npz B.npz | trace_decode OUT.npz N | diff_decode A.npz B.npz\n")
        sys.exit(2)