#!/usr/bin/env python3
"""BF16 prefill component attribution at the canonical legs, wall-clock only.

Loads the pinned bf16 model through mlx_lm (same path as bench_decode), then
times with wall clock around mx.eval + mx.synchronize (device timestamps are
unusable on this driver: they undercount by ~2.07x).

Legs: short=30, long=262, ctx1024=1053 chat-template prompt tokens.

Components, each timed in isolation at the exact per-layer attention and
projection shapes (B=1, H=14, KV=2, D=64, hidden=896, mlp=4864, vocab=151936):

  whole_prefill      full model pass, MLX_OMARCHY_SDPA_BF16_FAST unset (the
                     shipped f32 attention composition) and =1 (the bf16
                     storage composition). The env is flipped in-process; the
                     gate is read per call via getenv in primitives.cpp.
  sdpa_whole         mx.fast.scaled_dot_product_attention at (14, m, 64),
                     both env settings.
  f32 composition    q/k/v upcasts, q*scale multiply, scores f32 matmul,
                     f32 softmax, probs@v f32 matmul, output downcast.
  bf16 composition   softmax on bf16 scores, probs@v bf16 (coopmat), scores
                     bf16 matmul alpha=1 (coopmat) as the no-alpha bound;
                     the alpha-scaled scores ride the whole-sdpa residual.
  rates              per-shape projection matmuls, lm_head, rope, rms_norm,
                     silu, residual add, tiny-op host launch floor.

Every number is a wall-clock median over repeated full evaluations. Prints
NDJSON and writes it to --out.
"""

import argparse
import json
import os
import statistics
import time


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

    os.environ.pop("MLX_OMARCHY_SDPA_BF16_FAST", None)
    os.environ.setdefault("MLX_DISABLE_COMPILE", "1")

    from mlx_lm.utils import load

    import mlx.core as mx

    def nn_silu(x):
        return x / (1.0 + mx.exp(-x))

    model, tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-bf16")
    model.eval()
    out = open(args.out, "w")
    rows = []

    def emit(row):
        rows.append(row)
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

    # ---- whole-model prefill, f32 composition (gate off) ----
    evals = {}
    for name, ids in legs.items():
        x = mx.array(ids)[None]
        logits = model(x)
        mx.eval(logits)
        mx.synchronize()
        evals[name] = x
        us = timed(lambda: (mx.eval(model(x)), mx.synchronize()),
                   reps=args.reps)
        emit({"k": "whole_prefill", "leg": name, "gate": "off",
              "tokens": int(x.shape[1]), "us_median": round(us, 1),
              "tok_s": round(x.shape[1] / (us / 1e6), 1)})

    # ---- whole-model prefill, bf16 composition (gate on, in-process) ----
    os.environ["MLX_OMARCHY_SDPA_BF16_FAST"] = "1"
    for name in ("long", "ctx1024"):
        x = evals[name]
        logits = model(x)
        mx.eval(logits)
        mx.synchronize()
        us = timed(lambda: (mx.eval(model(x)), mx.synchronize()),
                   reps=args.reps)
        emit({"k": "whole_prefill", "leg": name, "gate": "on",
              "tokens": int(x.shape[1]), "us_median": round(us, 1),
              "tok_s": round(x.shape[1] / (us / 1e6), 1)})
    os.environ.pop("MLX_OMARCHY_SDPA_BF16_FAST", None)

    h = 896
    for m in (262, 1053):
        q = mx.random.normal((1, 14, m, 64)).astype(mx.bfloat16)
        kk = mx.random.normal((1, 14, m, 64)).astype(mx.bfloat16)
        v = mx.random.normal((1, 14, m, 64)).astype(mx.bfloat16)
        scale = 0.125

        # ---- sdpa whole, both gates ----
        us = timed(
            lambda: (mx.eval(mx.fast.scaled_dot_product_attention(
                q, kk, v, scale=scale)), mx.synchronize()),
            reps=args.reps)
        emit({"k": "sdpa_whole", "leg": f"m{m}", "gate": "off",
              "us_median": round(us, 1)})
        os.environ["MLX_OMARCHY_SDPA_BF16_FAST"] = "1"
        us = timed(
            lambda: (mx.eval(mx.fast.scaled_dot_product_attention(
                q, kk, v, scale=scale)), mx.synchronize()),
            reps=args.reps)
        os.environ.pop("MLX_OMARCHY_SDPA_BF16_FAST", None)
        emit({"k": "sdpa_whole", "leg": f"m{m}", "gate": "on",
              "us_median": round(us, 1)})

        # ---- f32 composition components (the shipped path) ----
        q32 = q.astype(mx.float32)
        k32 = kk.astype(mx.float32)
        v32 = v.astype(mx.float32)
        mx.eval(q32, k32, v32)
        mx.synchronize()
        us = timed(lambda: (mx.eval(q.astype(mx.float32)), mx.synchronize()),
                   reps=args.reps)
        emit({"k": "cast_q_f32", "leg": f"m{m}", "us_median": round(us, 1)})
        us = timed(
            lambda: (mx.eval(kk.astype(mx.float32),
                             v.astype(mx.float32)), mx.synchronize()),
            reps=args.reps)
        emit({"k": "cast_kv_f32", "leg": f"m{m}", "us_median": round(us, 1)})
        qs = mx.matmul(q32, scale)
        mx.eval(qs)
        mx.synchronize()
        us = timed(lambda: (mx.eval(mx.matmul(q32, scale)), mx.synchronize()),
                   reps=args.reps)
        emit({"k": "scale_mul_f32", "leg": f"m{m}", "us_median": round(us, 1)})
        scores = mx.matmul(qs, k32.swapaxes(-1, -2))
        probs = mx.softmax(scores, axis=-1)
        pv = mx.matmul(probs, v32)
        cast = pv.astype(mx.bfloat16)
        mx.eval(scores, probs, pv, cast)
        mx.synchronize()
        us = timed(lambda: (mx.eval(
            mx.matmul(qs, k32.swapaxes(-1, -2))), mx.synchronize()),
            reps=args.reps)
        emit({"k": "scores_f32", "leg": f"m{m}", "us_median": round(us, 1)})
        us = timed(lambda: (mx.eval(mx.softmax(scores, axis=-1)),
                            mx.synchronize()), reps=args.reps)
        emit({"k": "softmax_f32", "leg": f"m{m}", "us_median": round(us, 1)})
        us = timed(lambda: (mx.eval(mx.matmul(probs, v32)), mx.synchronize()),
                   reps=args.reps)
        emit({"k": "pv_f32", "leg": f"m{m}", "us_median": round(us, 1)})
        us = timed(lambda: (mx.eval(pv.astype(mx.bfloat16)),
                            mx.synchronize()), reps=args.reps)
        emit({"k": "cast_out_bf16", "leg": f"m{m}", "us_median": round(us, 1)})

        # ---- bf16 composition components ----
        scores_b = mx.matmul(q, kk.swapaxes(-1, -2))
        mx.eval(scores_b)
        mx.synchronize()
        # alpha=1 bound: the alpha-scaled scores ride the tile kernel and
        # are only observable as the sdpa_whole(gate=on) residual.
        us = timed(lambda: (mx.eval(
            mx.matmul(q, kk.swapaxes(-1, -2))), mx.synchronize()),
            reps=args.reps)
        emit({"k": "scores_bf16_alpha1", "leg": f"m{m}",
              "us_median": round(us, 1)})
        sb = mx.softmax(scores_b, axis=-1)
        mx.eval(sb)
        mx.synchronize()
        us = timed(lambda: (mx.eval(mx.softmax(scores_b, axis=-1)),
                            mx.synchronize()), reps=args.reps)
        emit({"k": "softmax_bf16", "leg": f"m{m}", "us_median": round(us, 1)})
        us = timed(lambda: (mx.eval(mx.matmul(sb, v)), mx.synchronize()),
                   reps=args.reps)
        emit({"k": "pv_bf16", "leg": f"m{m}", "us_median": round(us, 1)})

    # ---- per-layer projection + mlp rates, norms, rope ----
    for m in (262, 1053):
        for (k, n, tag) in ((896, 896, "q_o"), (896, 128, "kv"),
                            (896, 4864, "gate_up"), (4864, 896, "down")):
            xb = mx.random.normal((m, k)).astype(mx.bfloat16)
            w = mx.random.normal((n, k)).astype(mx.bfloat16)
            mx.eval(xb, w)
            mx.synchronize()
            us = timed(
                lambda: (mx.eval(xb @ w.T), mx.synchronize()),
                reps=args.reps)
            emit({"k": "matmul_bf16", "m": m, "kn": f"{k}x{n}", "tag": tag,
                  "us_median": round(us, 1),
                  "tflops": round(2 * m * k * n / (us * 1e6), 3)})
        lw = mx.random.normal((151936, 896)).astype(mx.bfloat16)
        lt = mx.random.normal((m, 896)).astype(mx.bfloat16)
        mx.eval(lw, lt)
        mx.synchronize()
        us = timed(lambda: (mx.eval(lt @ lw.T), mx.synchronize()),
                   reps=args.reps)
        emit({"k": "lm_head_bf16", "m": m, "us_median": round(us, 1),
              "tflops": round(2 * m * 896 * 151936 / (us * 1e6), 3)})
        t = mx.random.normal((1, m, h)).astype(mx.bfloat16)
        g = mx.random.normal((m, 4864)).astype(mx.bfloat16)
        r = mx.random.normal((m, 896)).astype(mx.bfloat16)
        mx.eval(t, g, r)
        mx.synchronize()
        us = timed(lambda: (mx.eval(mx.fast.rope(
            t, 64, traditional=False, base=10000.0, scale=1.0,
            offset=0)), mx.synchronize()), reps=args.reps)
        emit({"k": "rope", "m": m, "us_median": round(us, 1)})
        us = timed(lambda: (mx.eval(mx.fast.rms_norm(
            t, mx.ones((h,)).astype(mx.bfloat16), 1e-6)),
            mx.synchronize()), reps=args.reps)
        emit({"k": "rms_norm", "m": m, "us_median": round(us, 1)})
        us = timed(lambda: (mx.eval(nn_silu(g)), mx.synchronize()),
                   reps=args.reps)
        emit({"k": "silu", "m": m, "us_median": round(us, 1)})
        us = timed(lambda: (mx.eval(r + r), mx.synchronize()),
                   reps=args.reps)
        emit({"k": "residual_add", "m": m, "us_median": round(us, 1)})

    # ---- host launch floor: one tiny dispatch ----
    a = mx.array([1.0])
    mx.eval(a)
    mx.synchronize()
    us = timed(lambda: (mx.eval(a + a), mx.synchronize()), reps=51,
               discard=10)
    emit({"k": "host_launch_floor", "us_median": round(us, 1)})

    out.close()


if __name__ == "__main__":
    main()
