#!/usr/bin/env python3
"""CPU-device oracle leg for the long-prompt divergence.

Same engine (bench_decode), same pinned model, same greedy/EOS-suppressed
policy, but weights and all math run on the stock upstream CPU backend
(no omarchy Vulkan code in the numeric path). One leg answers whether the
Linux long128 digest 4cc08910089477fd is stack-wide (tokenizer/chat
template/sampling) or Vulkan-kernel-specific.

Usage (cwd = repo root, env like run-baseline.py):
  MLX_PAIR_IDS=/tmp/cpu.ids.jsonl python3 cpu-oracle.py \
    --model <snapshot> --prompt-id long --tokens 128
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "scripts"))
import bench_matrix
import mlx.core as mx

mx.set_default_device(mx.cpu)

import bench_decode

original_report = bench_decode.report


def capture_report(prefill_ns, token_times, requested, ids=None,
                   prompt_tokens=None, device=None):
    if ids is None:
        raise RuntimeError("Benchmark did not provide generated token IDs")
    result = original_report(prefill_ns, token_times, requested, ids,
                             prompt_tokens=prompt_tokens,
                             device=f"{device}+cpu-backend")
    if os.environ.get("MLX_PAIR_IDS"):
        with open(os.environ["MLX_PAIR_IDS"], "a") as output:
            output.write(json.dumps({
                "requested": requested,
                "prompt_tokens": prompt_tokens,
                "ids": [int(i) for i in ids],
                "device": "cpu",
            }) + "\n")
    return result


bench_decode.report = capture_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-id", default="long")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup-tokens", type=int, default=4)
    args = ap.parse_args()
    manifest = json.loads(
        (Path.cwd() / "scripts" / "bench_matrix.json").read_text())
    prompt_text = bench_matrix.prompt_text(manifest, args.prompt_id)
    sys.argv = ["bench_decode", "--model", args.model,
                "--prompt", prompt_text, "--tokens", str(args.tokens),
                "--temp", str(args.temp), "--seed", str(args.seed),
                "--warmup-tokens", str(args.warmup_tokens)]
    raise SystemExit(bench_decode.main())


if __name__ == "__main__":
    main()
