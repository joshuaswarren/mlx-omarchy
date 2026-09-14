#!/usr/bin/env python3
"""Per-token backend counters for Qwen2.5-0.5B-Instruct-4bit decode.

Samples the omarchy trace counters (trace.h snapshot, ctypes on the
loaded libmlx.so) around every generated token of a short greedy run
and prints the per-token deltas of vk_compute_dispatches (the number
the fold targets), gpu_primitive_dispatches (eval_gpu calls, so a fold
that trades kernels for host-side fixups shows here) and vk_submissions.
Counts are host-side bookkeeping: valid on any device. Run with
MLX_DISABLE_COMPILE=1 like the receipt legs.
"""

import argparse
import ctypes
import json
import pathlib
import sys


def counter_reader():
    import mlx.core as mx

    class Snapshot(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in (
            "gpu_primitive_dispatches", "vk_submissions",
            "vk_buffer_copies", "vk_buffer_fills", "vk_compute_dispatches",
            "omarchy_finalize_calls", "commit_calls_with_work",
            "commit_calls_noop")]

    so = pathlib.Path(mx.__file__).parent / "lib" / "libmlx.so"
    lib = ctypes.CDLL(str(so))
    lib.mlx_omarchy_trace_snapshot.argtypes = [ctypes.POINTER(Snapshot)]
    lib.mlx_omarchy_trace_snapshot.restype = None
    snap = Snapshot()

    def read():
        lib.mlx_omarchy_trace_snapshot(ctypes.byref(snap))
        return {name: getattr(snap, name) for name, _ in Snapshot._fields_}
    return read


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="Hi")
    ap.add_argument("--tokens", type=int, default=8)
    args = ap.parse_args()
    sys.path.insert(0, "/var/tmp/DecodeEpilogueFold/cand/scripts")
    from mlx_provenance import installed_provenance, provenance_line
    print(provenance_line(installed_provenance()))
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    read = counter_reader()
    model, tokenizer = load(args.model)
    messages = [{"role": "user", "content": args.prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True)
    keys = ("vk_compute_dispatches", "gpu_primitive_dispatches",
            "vk_submissions")
    before = read()
    rows = []
    for i, resp in enumerate(stream_generate(
            model, tokenizer, prompt, max_tokens=args.tokens,
            sampler=make_sampler(0.0))):
        after = read()
        rows.append({k: after[k] - before[k] for k in keys})
        rows[-1]["token"] = i
        rows[-1]["prompt_tokens"] = resp.prompt_tokens
        before = after
    # Token 0 carries the prefill; the decode figure is the median of
    # the rest.
    decode = rows[1:]
    out = {"device": str(mx.default_device()), "prompt_tokens":
           rows[0]["prompt_tokens"], "rows": rows,
           "per_token": {k: sorted(r[k] for r in decode)[len(decode) // 2]
                         for k in keys}}
    print(json.dumps(out, sort_keys=True))


if __name__ == "__main__":
    main()
