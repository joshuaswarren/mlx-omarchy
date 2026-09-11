#!/usr/bin/env python3
"""Host-phase decode leg on the host-trace instrumented wheel.

Measurement wheel for receipts/2026-09-10-hostpath (branch
wave/HostPathOverhead, never merged). Mirrors scripts/bench_decode.py's
canonical flow - mlx_lm.load, chat template, greedy sampler, warmup
tokens, then a timed pinned window - and adds exactly three
instruments, none of them GPU timestamps:

1. the backend host-trace counters (MLX_OMARCHY_HOST_TRACE), reset via
   ctypes right before the timed window and dumped right after, so the
   totals cover the decode window only;
2. a monkeypatched mx.eval recording per-call host durations (C++ time
   inside eval includes scheduling + record + submit, excludes Python);
3. per-token wall gaps from the token iterator, as the canonical runner
   computes them.

Env: MLX_OMARCHY_HOST_TRACE must point somewhere (the dump also goes to
--trace-out); MLX_OMARCHY_REPLAY=1 arms the replay prototype.
"""
import argparse
import ctypes
import hashlib
import importlib.util
import json
import time
from pathlib import Path


def ids_digest(ids):
    # EXACT copy of scripts/bench_decode.py:ids_digest - identity is
    # decided by comparing these digests against the canonical pins.
    h = hashlib.sha256()
    h.update(",".join(str(int(i)) for i in ids).encode("ascii"))
    return h.hexdigest()[:16]


def trace_lib():
    spec = importlib.util.find_spec("mlx.core")
    assert spec is not None and spec.origin, "mlx.core not importable"
    lib = ctypes.CDLL(spec.origin)
    try:
        lib.mlx_omarchy_host_trace_reset.argtypes = []
        lib.mlx_omarchy_host_trace_reset.restype = None
        lib.mlx_omarchy_host_trace_dump.argtypes = [ctypes.c_char_p]
        lib.mlx_omarchy_host_trace_dump.restype = ctypes.c_int
    except AttributeError:
        return None  # release wheel: no trace symbols
    return lib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--trace-out", default="/tmp/hosttrace.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from mlx_lm.utils import load
    from mlx_lm.generate import stream_generate
    from mlx_lm.sample_utils import make_sampler
    import mlx.core as mx

    model, tokenizer = load(args.model)
    prompt = args.prompt
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            add_generation_prompt=True)
    add_special = (getattr(tokenizer, "bos_token", None) is None or
                   not prompt.startswith(tokenizer.bos_token))
    prompt_ids = mx.array(tokenizer.encode(
        prompt, add_special_tokens=add_special))
    tokenizer.eos_token_ids = set()
    mx.random.seed(0)
    sampler = make_sampler(temp=0.0)

    eval_ns = {"t": 0, "n": 0}
    orig_eval = mx.eval

    def timed_eval(*a, **k):
        t0 = time.perf_counter_ns()
        orig_eval(*a, **k)
        eval_ns["t"] += time.perf_counter_ns() - t0
        eval_ns["n"] += 1

    def generate(n):
        return stream_generate(
            model, tokenizer, prompt_ids, max_tokens=n, sampler=sampler)

    for _ in generate(args.warmup):
        pass

    lib = trace_lib()
    if lib is not None:
        lib.mlx_omarchy_host_trace_reset()
    eval_ns["t"] = 0
    eval_ns["n"] = 0
    mx.eval = timed_eval

    ids = []
    gaps = []
    prev = time.perf_counter_ns()
    for r in generate(args.tokens):
        now = time.perf_counter_ns()
        gaps.append(now - prev)
        prev = now
        ids.append(int(getattr(r, "token", r)))
    mx.eval = orig_eval

    trace = None
    rc = None
    if lib is not None:
        rc = lib.mlx_omarchy_host_trace_dump(args.trace_out.encode())
        if args.trace_out != "/dev/null":
            trace = json.loads(Path(args.trace_out).read_text())
    decode_gaps = gaps[1:]
    wall_s = sum(decode_gaps) / 1e9
    result = {
        "schema": "mlx-omarchy/hostpath-leg/1",
        "model": args.model,
        "prompt_tokens": int(prompt_ids.size),
        "n_tokens": len(ids),
        "digest": ids_digest(ids),
        "decode_tok_s": len(decode_gaps) / wall_s,
        "decode_ms_per_token": 1000.0 * wall_s / len(decode_gaps),
        "eval_call_ns_total": eval_ns["t"],
        "eval_calls": eval_ns["n"],
        "trace_dump_rc": rc,
        "trace": trace,
    }
    text = json.dumps(result, indent=1)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--trace-out", default="/tmp/hosttrace.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from mlx_lm.utils import load
    from mlx_lm.generate import stream_generate
    from mlx_lm.sample_utils import make_sampler
    import mlx.core as mx

    model, tokenizer = load(args.model)
    prompt = args.prompt
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            add_generation_prompt=True)
    add_special = (getattr(tokenizer, "bos_token", None) is None or
                   not prompt.startswith(tokenizer.bos_token))
    prompt_ids = mx.array(tokenizer.encode(
        prompt, add_special_tokens=add_special))
    tokenizer.eos_token_ids = set()
    mx.random.seed(0)
    sampler = make_sampler(temp=0.0)

    eval_ns = {"t": 0, "n": 0}
    orig_eval = mx.eval

    def timed_eval(*a, **k):
        t0 = time.perf_counter_ns()
        orig_eval(*a, **k)
        eval_ns["t"] += time.perf_counter_ns() - t0
        eval_ns["n"] += 1

    def generate(n):
        return stream_generate(
            model, tokenizer, prompt_ids, max_tokens=n, sampler=sampler)

    for _ in generate(args.warmup):
        pass

    lib = trace_lib()
    lib.mlx_omarchy_host_trace_reset()
    eval_ns["t"] = 0
    eval_ns["n"] = 0
    mx.eval = timed_eval

    ids = []
    gaps = []
    prev = time.perf_counter_ns()
    for r in generate(args.tokens):
        now = time.perf_counter_ns()
        gaps.append(now - prev)
        prev = now
        ids.append(int(getattr(r, "token", r)))
    mx.eval = orig_eval

    rc = lib.mlx_omarchy_host_trace_dump(args.trace_out.encode())
    trace = json.loads(Path(args.trace_out).read_text())
    decode_gaps = gaps[1:]
    wall_s = sum(decode_gaps) / 1e9
    result = {
        "schema": "mlx-omarchy/hostpath-leg/1",
        "model": args.model,
        "prompt_tokens": int(prompt_ids.size),
        "n_tokens": len(ids),
        "digest": ids_digest(ids),
        "decode_tok_s": len(decode_gaps) / wall_s,
        "decode_ms_per_token": 1000.0 * wall_s / len(decode_gaps),
        "eval_call_ns_total": eval_ns["t"],
        "eval_calls": eval_ns["n"],
        "trace_dump_rc": rc,
        "trace": trace,
    }
    text = json.dumps(result, indent=1)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
