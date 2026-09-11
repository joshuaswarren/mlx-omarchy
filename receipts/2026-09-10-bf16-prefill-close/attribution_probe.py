#!/usr/bin/env python3
"""BF16 prefill attribution at the three canonical legs, uninstrumented.

Loads the pinned bf16 model through mlx_lm (same path as bench_decode),
then times with wall clock around mx.eval + mx.synchronize:

  * whole-model prefill passes at the three canonical prompt leg scales
    (short=30, long=262, ctx1024=1053 prompt tokens);
  * isolated dense bf16 matmuls at the per-layer projection shapes
    (k=896 n=896/128/4864, k=4864 n=896) - on the fork driver these are
    the MatmulBF16Coopmat dispatches, with MLX_OMARCHY_NO_COOPMAT=1 the
    16x16 tile;
  * fast.scaled_dot_product_attention at the attention shapes (the f32
    composition), fast.rope, and rms norm.

No profiler, no per-dispatch timers: every number is a wall-clock median
over repeated full evaluations. Prints NDJSON and writes it to --out.
"""

import argparse
import json
import statistics
import time


def nn_silu(x):
    return x / (1.0 + mx.exp(-x))


def timed(fn, reps=9, discard=2):
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(samples[discard:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=9)
    args = ap.parse_args()

    from mlx_lm.utils import load

    import mlx.core as mx

    model, tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-bf16")
    model.eval()
    out = open(args.out, "w")
    rows = []

    def emit(row):
        rows.append(row)
        print(json.dumps(row), flush=True)
        out.write(json.dumps(row) + "\n")
        out.flush()

    prompts = {
        "short": "Hi",
        "long": ("You are reviewing a measurement protocol for a small "
                 "language-model benchmark. The benchmark runs the same "
                 "workload matrix on two operating systems and compares "
                 "prefill and decode throughput. Each leg loads one pinned "
                 "model snapshot, applies the model chat template to the "
                 "prompt, runs greedy decoding with a fixed random seed, "
                 "and records the prompt phase, the per-token phase, the "
                 "generated token count, a hash of the generated token "
                 "ids, and peak memory. A leg is valid only when the model "
                 "snapshot revision matches the manifest pin, the power "
                 "state is recorded, and no other model-serving process is "
                 "holding the accelerator."),
    }
    short_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompts["short"]}],
        add_generation_prompt=True)
    long_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompts["long"]}],
        add_generation_prompt=True)
    # The ctx1024 leg repeats the long prompt to ~1053 chat-template
    # tokens; token content does not affect prefill timing, only the
    # count does.
    filler = (long_ids * 8)[:1053]
    legs = {"short": short_ids, "long": long_ids, "ctx1024": filler}

    for name, ids in legs.items():
        x = mx.array(ids)[None]
        logits = model(x)
        mx.eval(logits)
        mx.synchronize()
        us = timed(lambda: (mx.eval(model(x)), mx.synchronize()),
                   reps=args.reps)
        emit({"k": "whole_prefill", "leg": name, "tokens": int(x.shape[1]),
              "us_median": round(us, 1),
              "tok_s": round(x.shape[1] / (us / 1e6), 1)})

    h = 896
    for m in (30, 262, 1053):
        for (k, n, tag) in ((896, 896, "q_o"), (896, 128, "kv"),
                            (896, 4864, "gate_up"), (4864, 896, "down")):
            xb = mx.random.normal((m, k)).astype(mx.bfloat16)
            w = mx.random.normal((n, k)).astype(mx.bfloat16)
            mx.eval(xb)
            mx.eval(w)
            us = timed(
                lambda: (mx.eval(xb @ w.T), mx.synchronize()),
                reps=args.reps)
            emit({"k": "matmul_bf16", "m": m, "kn": f"{k}x{n}", "tag": tag,
                  "us_median": round(us, 1),
                  "tflops": round(2 * m * k * n / (us * 1e6), 3)})
        # Attention at the f32 composition shapes (14 heads, head_dim 64).
        q = mx.random.normal((1, 14, m, 64)).astype(mx.bfloat16)
        kk = mx.random.normal((1, 14, m, 64)).astype(mx.bfloat16)
        v = mx.random.normal((1, 14, m, 64)).astype(mx.bfloat16)
        us = timed(
            lambda: (mx.eval(mx.fast.scaled_dot_product_attention(
                q, kk, v, scale=0.125)), mx.synchronize()),
            reps=args.reps)
        emit({"k": "sdpa", "m": m, "us_median": round(us, 1)})
        # Rope + norm scale proxies.
        t = mx.random.normal((1, m, h)).astype(mx.bfloat16)
        us = timed(
            lambda: (mx.eval(mx.fast.rope(
                t, 64, traditional=False, base=10000.0, scale=1.0,
                offset=0)),
                mx.synchronize()),
            reps=args.reps)
        emit({"k": "rope", "m": m, "us_median": round(us, 1)})
        us = timed(
            lambda: (mx.eval(mx.fast.rms_norm(
                t, 1e-6, mx.ones((h,)).astype(mx.bfloat16))),
                mx.synchronize()),
            reps=args.reps)
        emit({"k": "rms_norm", "m": m, "us_median": round(us, 1)})

    out.close()


if __name__ == "__main__":
    main()
