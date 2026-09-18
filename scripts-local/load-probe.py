#!/usr/bin/env python3
"""Device load-matrix probe for the 0.32.3 bump candidate (jw16 Vulkan).

Run inside the candidate venv with the GPU lock held. One JSON line per
model: id, arch, status (LOADED/GENERATED/FAILED), load_s, gen_s, gen
sample, peak memory, error.
"""
import json
import sys
import time
import traceback

import mlx.core as mx


def peak_mem():
    for name in ("get_peak_memory", "metal_get_peak_memory"):
        fn = getattr(mx, name, None)
        if fn:
            try:
                return fn()
            except Exception:
                pass
    return None


def reset_mem():
    for name in ("clear_cache", "metal_clear_cache"):
        fn = getattr(mx, name, None)
        if fn:
            try:
                fn()
            except Exception:
                pass


def text_probe(repo, max_tokens=16):
    from mlx_lm import load
    t0 = time.time()
    model, tokenizer = load(repo)
    load_s = time.time() - t0
    arch = getattr(getattr(model, "config", None), "model_type", "?")
    from mlx_lm import generate
    prompt = "The capital of France is"
    t1 = time.time()
    try:
        text = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)
    except TypeError:
        text = generate(model, tokenizer, prompt, max_tokens=max_tokens)
    gen_s = time.time() - t1
    return {
        "arch": arch,
        "status": "GENERATED",
        "load_s": round(load_s, 2),
        "gen_s": round(gen_s, 2),
        "sample": text[:120].replace("\n", " "),
        "peak_mem_gb": round((peak_mem() or 0) / 1e9, 2),
    }


def bonsai8_probe(repo, max_tokens=16):
    # Gen-1 pack: plain qwen3 affine 2-bit group-128, loads via mlx_lm stock.
    return text_probe(repo, max_tokens)


def main():
    repo = sys.argv[1]
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    reset_mem()
    try:
        if "Ternary-Bonsai-8B" in repo:
            result = bonsai8_probe(repo, max_tokens)
        else:
            result = text_probe(repo, max_tokens)
    except Exception as e:
        result = {
            "status": "FAILED",
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc(limit=4),
        }
    result["id"] = repo
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
