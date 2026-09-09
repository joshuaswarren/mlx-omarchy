#!/usr/bin/env python3
"""Time the host work around a prompt-sized forward: graph build, eval,
mx.clear_cache(), the decode graph build, and its eval, at the default
allocator cache limit and at a large one.
Usage: clear_cache_probe.py MODEL PROMPT_FILE [cache_limit_bytes ...]"""
import sys
import time

import mlx.core as mx
from mlx_lm.utils import load
from mlx_lm.models.cache import make_prompt_cache

model, tokenizer = load(sys.argv[1])
prompt = open(sys.argv[2]).read()
ids = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}], add_generation_prompt=True)
ids = mx.array(ids)
print(f"prompt tokens {ids.size}")
limits = [None] + [int(x) for x in sys.argv[3:]]
for limit in limits:
    if limit is not None:
        mx.set_cache_limit(limit)
    for rep in range(3):
        cache = make_prompt_cache(model)
        t0 = time.perf_counter()
        model(ids[:-1][None], cache=cache)
        t1 = time.perf_counter()
        mx.eval([c.state for c in cache])
        t2 = time.perf_counter()
        mx.clear_cache()
        t3 = time.perf_counter()
        logits = model(ids[-1:][None], cache=cache)
        t4 = time.perf_counter()
        mx.eval(logits)
        t5 = time.perf_counter()
        print(f"limit={limit} rep {rep}: prefill build {1e3*(t1-t0):.1f} ms, "
              f"prefill eval {1e3*(t2-t1):.1f} ms, clear_cache {1e3*(t3-t2):.1f} ms, "
              f"decode build {1e3*(t4-t3):.1f} ms, decode eval {1e3*(t5-t4):.1f} ms")
