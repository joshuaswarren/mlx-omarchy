#!/usr/bin/env python3
"""Profile exactly the warm bench_decode prefill-to-first-token window."""

import argparse
import cProfile
import ctypes
import json
import pathlib
import pstats
import time


def counter_reader(mx):
    so = pathlib.Path(mx.__file__).parent / "lib" / "libmlx.so"

    class Snapshot(ctypes.Structure):
        _fields_ = [
            (name, ctypes.c_uint64)
            for name in (
                "gpu_primitive_dispatches",
                "vk_submissions",
                "vk_buffer_copies",
                "vk_buffer_fills",
                "vk_compute_dispatches",
                "omarchy_finalize_calls",
                "commit_calls_with_work",
                "commit_calls_noop",
            )
        ]

    lib = ctypes.CDLL(str(so))
    fn = lib.mlx_omarchy_trace_snapshot
    fn.argtypes = [ctypes.POINTER(Snapshot)]
    fn.restype = None

    def read():
        snap = Snapshot()
        fn(ctypes.byref(snap))
        return {name: getattr(snap, name) for name, _ in snap._fields_}

    return read


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup-tokens", type=int, default=4)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    from mlx_lm.generate import stream_generate
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.utils import load
    import mlx.core as mx

    read_counters = counter_reader(mx)
    model, tokenizer = load(args.model)
    prompt = pathlib.Path(args.prompt_file).read_text()
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
        )
    sampler = make_sampler(temp=0.0)
    saved_eos = getattr(tokenizer, "eos_token_ids", None)
    tokenizer.eos_token_ids = set()

    def generate(n):
        return stream_generate(
            model, tokenizer, prompt, max_tokens=n, sampler=sampler
        )

    for _ in generate(args.warmup_tokens):
        pass

    events = []

    def wrap(obj, name):
        original = getattr(obj, name)

        def timed(*a, **kw):
            before = read_counters()
            t0 = time.monotonic_ns()
            result = original(*a, **kw)
            t1 = time.monotonic_ns()
            after = read_counters()
            events.append({
                "name": name,
                "start_ns": t0,
                "wall_ms": (t1 - t0) / 1e6,
                "counters": {k: after[k] - before[k] for k in after},
            })
            return result

        setattr(obj, name, timed)
        return original

    originals = [
        (mx, "eval", wrap(mx, "eval")),
        (mx, "async_eval", wrap(mx, "async_eval")),
        (mx, "clear_cache", wrap(mx, "clear_cache")),
        (mx.array, "item", wrap(mx.array, "item")),
    ]

    markers = out.with_suffix(".markers.jsonl")
    gpu_start = time.monotonic_ns()
    markers.write_text(json.dumps({"t": gpu_start, "p": "prefill_start"}) + "\n")
    start = read_counters()
    profile = cProfile.Profile()
    iterator = generate(2)
    profile.enable()
    t0 = time.monotonic_ns()
    first = next(iterator)
    t1 = time.monotonic_ns()
    profile.disable()
    end = read_counters()
    with markers.open("a") as f:
        f.write(json.dumps({"t": t1, "p": "prefill_done"}) + "\n")
    profile.dump_stats(str(out.with_suffix(".pstats")))
    with out.with_suffix(".cprofile.txt").open("w") as f:
        pstats.Stats(profile, stream=f).strip_dirs().sort_stats("cumulative").print_stats(120)

    for obj, name, original in originals:
        setattr(obj, name, original)
    if saved_eos is not None:
        tokenizer.eos_token_ids = saved_eos
    mx.synchronize()

    result = {
        "wall_ms": (t1 - t0) / 1e6,
        "prompt_tokens": int(first.prompt_tokens),
        "first_token": int(first.token),
        "counters": {k: end[k] - start[k] for k in end},
        "boundary_events": events,
    }
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
