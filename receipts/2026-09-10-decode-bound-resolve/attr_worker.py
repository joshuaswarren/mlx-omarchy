#!/usr/bin/env python3
"""Decode-attribution discriminator: one leg, four direct instruments.

Part 1 of the decode-bound resolution. Question: is a canonical Q4
decode token GPU-bound or host-paced? The two prior instruments
disagree (receipts/2026-09-10-decode-attribution-finish reads GPU-bound
from ablation marginals; receipts/2026-09-10-hostpath reads host-paced
from wall-region occupancy). Wall-region occupancy is not CPU-busy
time, so this leg measures the missing quantities directly:

  1. thread CPU time vs wall per token (time.thread_time_ns): a host
     thread that is EXECUTING accumulates CPU time; a thread BLOCKED
     on the GPU does not. Python work and C++ eval work both land here
     because both run on this thread.
  2. process CPU time vs wall per token (time.process_time_ns): adds
     every other thread (completion/submission threads).
  3. inside mx.eval: wall vs thread CPU per call - splits "C++ working"
     from "blocked in eval waiting for the GPU".
  4. submitted-vs-completed per token: the timed eval boundaries are
     the submit side; the token yield is the completed side.

Knobs (env):
  ATTR_SPIN_US   pure-Python spin per token in the consumer loop
                 (scales host only; GPU work unchanged)
  ATTR_GUP_K     mx.eval of a fresh big f16 matmul per token (scales
                 GPU only; ~K host ops added, digest must not move)
  ATTR_WARMUP / ATTR_TOKENS / ATTR_PROMPT
  MLX_OMARCHY_ABLATE  honored by the ablation wheel (scales GPU down)

Digest is computed exactly like scripts/bench_decode.py and gated by
the driver against the canonical pins.

--self-test needs no mlx: checks digest, spin calibration accounting,
and the per-token record shape with a stub loop.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def ids_digest(ids):
    # EXACT copy of scripts/bench_decode.py:ids_digest
    h = hashlib.sha256()
    h.update(",".join(str(int(i)) for i in ids).encode("ascii"))
    return h.hexdigest()[:16]


def spin_us(us):
    end = time.perf_counter_ns() + us * 1000
    x = 0
    while time.perf_counter_ns() < end:
        x += 1
    return x


def record_shape():
    return {
        "wall_ns": 0, "thread_cpu_ns": 0, "proc_cpu_ns": 0,
        "eval_wall_ns": 0, "eval_thread_cpu_ns": 0, "eval_calls": 0,
        "gup_wall_ns": 0, "gup_calls": 0,
    }


def self_test():
    assert ids_digest([1, 2, 3]) == ids_digest([1, 2, 3])
    assert ids_digest([1, 2, 3]) != ids_digest([1, 2, 4])
    t0 = time.thread_time_ns()
    spin_us(20000)
    busy = time.thread_time_ns() - t0
    assert busy > 10_000_000, f"spin did not accumulate thread CPU: {busy}"
    r = record_shape()
    assert set(r) == {"wall_ns", "thread_cpu_ns", "proc_cpu_ns",
                      "eval_wall_ns", "eval_thread_cpu_ns", "eval_calls",
                      "gup_wall_ns", "gup_calls"}
    # a blocked wait accumulates wall but (almost) no thread CPU
    t1 = time.thread_time_ns()
    w1 = time.perf_counter_ns()
    time.sleep(0.02)
    blocked_wall = time.perf_counter_ns() - w1
    blocked_cpu = time.thread_time_ns() - t1
    assert blocked_wall > 15_000_000 and blocked_cpu < 5_000_000, (
        blocked_wall, blocked_cpu)
    print("self-test: OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.model:
        ap.error("--model is required (path to the mlx_lm model)")
    if args.tokens < 2:
        ap.error("--tokens must be >= 2")

    from mlx_lm.utils import load
    from mlx_lm.generate import stream_generate
    from mlx_lm.sample_utils import make_sampler
    import mlx.core as mx

    model, tokenizer = load(args.model)
    prompt = args.prompt
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True)
    if isinstance(prompt, str):
        add_special = (getattr(tokenizer, "bos_token", None) is None or
                       not prompt.startswith(tokenizer.bos_token))
        prompt_ids = mx.array(tokenizer.encode(
            prompt, add_special_tokens=add_special))
    else:
        prompt_ids = mx.array(prompt)
    tokenizer.eos_token_ids = set()
    mx.random.seed(0)
    sampler = make_sampler(temp=0.0)

    spin_amt = int(os.environ.get("ATTR_SPIN_US", "0"))
    gup_k = int(os.environ.get("ATTR_GUP_K", "0"))

    gup = None
    gup_ops = None
    if gup_k > 0:
        for n in (1600, 2048, 1280):
            a = mx.ones((n, n), dtype=mx.float16)
            b = mx.ones((n, n), dtype=mx.float16)
            cal = []
            for _ in range(7):
                d = a @ b
                t0 = time.perf_counter_ns()
                mx.eval(d)
                cal.append(time.perf_counter_ns() - t0)
            cal_us = sorted(cal)[len(cal) // 2] / 1000.0
            if 1000.0 <= cal_us <= 4000.0:
                break
        gup = {"n": n, "calib_us": cal_us}
        gup_ops = (a, b)

    eval_acc = {"wall": 0, "cpu": 0, "n": 0}
    gup_acc = {"wall": 0, "n": 0}
    orig_eval = mx.eval

    def timed_eval(*a, **k):
        w0 = time.perf_counter_ns()
        c0 = time.thread_time_ns()
        orig_eval(*a, **k)
        eval_acc["wall"] += time.perf_counter_ns() - w0
        eval_acc["cpu"] += time.thread_time_ns() - c0
        eval_acc["n"] += 1

    def do_gup():
        # gup_k-deep chain of big lazy products, one eval at the end:
        # GPU-only scale-up (~K host nodes, data-dependent chain so the
        # matmuls serialize on the device); the output buffers come
        # from the allocator cache warmed by the two calls below.
        if gup_ops is None:
            return
        a, b = gup_ops
        d = a @ b
        for _ in range(gup_k - 1):
            d = d @ b
        w0 = time.perf_counter_ns()
        orig_eval(d)
        gup_acc["wall"] += time.perf_counter_ns() - w0
        gup_acc["n"] += 1

    def generate(n_tok):
        return stream_generate(
            model, tokenizer, prompt_ids, max_tokens=n_tok, sampler=sampler)

    for _ in generate(args.warmup):
        pass
    if gup_k > 0:
        for _ in range(2):
            do_gup()

    eval_acc.update(wall=0, cpu=0, n=0)
    gup_acc.update(wall=0, n=0)
    mx.eval = timed_eval
    ids = []
    toks = []
    prev_w = time.perf_counter_ns()
    prev_c = time.thread_time_ns()
    prev_p = time.process_time_ns()
    for r in generate(args.tokens):
        now_w = time.perf_counter_ns()
        now_c = time.thread_time_ns()
        now_p = time.process_time_ns()
        rec = record_shape()
        rec["wall_ns"] = now_w - prev_w
        rec["thread_cpu_ns"] = now_c - prev_c
        rec["proc_cpu_ns"] = now_p - prev_p
        rec["eval_wall_ns"] = eval_acc["wall"]
        rec["eval_thread_cpu_ns"] = eval_acc["cpu"]
        rec["eval_calls"] = eval_acc["n"]
        rec["gup_wall_ns"] = gup_acc["wall"]
        rec["gup_calls"] = gup_acc["n"]
        eval_acc.update(wall=0, cpu=0, n=0)
        gup_acc.update(wall=0, n=0)
        toks.append(rec)
        prev_w, prev_c, prev_p = now_w, now_c, now_p
        ids.append(int(getattr(r, "token", r)))
        if spin_amt:
            spin_us(spin_amt)
        if gup_k > 0:
            do_gup()
    mx.eval = orig_eval

    gaps = [t["wall_ns"] for t in toks][1:]
    wall_s = sum(gaps) / 1e9
    tok_ms = 1000.0 * wall_s / len(gaps)
    thread_cpu_ms = (sum(t["thread_cpu_ns"] for t in toks[1:]) / 1e6) / len(gaps)
    proc_cpu_ms = (sum(t["proc_cpu_ns"] for t in toks[1:]) / 1e6) / len(gaps)
    eval_wall_ms = (sum(t["eval_wall_ns"] for t in toks[1:]) / 1e6) / len(gaps)
    eval_cpu_ms = (sum(t["eval_thread_cpu_ns"] for t in toks[1:]) / 1e6) / len(gaps)
    result = {
        "schema": "mlx-omarchy/decode-attr-leg/1",
        "model": os.path.basename(args.model.rstrip("/")),
        "prompt_tokens": int(prompt_ids.size),
        "n_tokens": len(ids),
        "digest": ids_digest(ids),
        "decode_tok_s": len(gaps) / wall_s if wall_s else None,
        "tok_ms": tok_ms,
        "thread_cpu_ms_per_token": thread_cpu_ms,
        "proc_cpu_ms_per_token": proc_cpu_ms,
        "eval_wall_ms_per_token": eval_wall_ms,
        "eval_thread_cpu_ms_per_token": eval_cpu_ms,
        "eval_calls_per_token": sum(t["eval_calls"] for t in toks[1:]) / len(gaps),
        "spin_us": spin_amt,
        "gup_k": gup_k,
        "gup": gup,
        "gup_wall_ms_per_token": (sum(t["gup_wall_ns"] for t in toks[1:])
                                  / 1e6) / len(gaps) if gup else None,
        "ablate": os.environ.get("MLX_OMARCHY_ABLATE"),
        "tok_cpu_vs_wall_ms": [
            [round(t["thread_cpu_ns"] / 1e6, 3),
             round(t["wall_ns"] / 1e6, 3)] for t in toks[1:]],
    }
    text = json.dumps(result)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
