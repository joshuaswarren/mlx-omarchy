#!/usr/bin/env python3
"""CPU-device oracle leg for the long-prompt divergence.

Same engine (bench_decode), same pinned model, same greedy/EOS-suppressed
policy, but weights and all math run on the stock upstream CPU backend
(no omarchy Vulkan code in the numeric path). One leg answers whether the
Linux long128 digest 4cc08910089477fd is stack-wide (tokenizer/chat
template/sampling) or Vulkan-kernel-specific.

Usage (cwd = repo root, env like run-baseline.py):
  MLX_PAIR_IDS=/tmp/cpu.ids.jsonl python3 cpu-oracle.py \
    --model <snapshot> --prompt "<long prompt>" --tokens 128 \
    --temp 0 --seed 0 --warmup-tokens 4
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "scripts"))
import mlx.core as mx

mx.set_default_device(mx.cpu)

import bench_decode

original_report = bench_decode.report


def capture_report(prefill_ns, token_times, requested, ids=None,
                   prompt_tokens=None, device=None):
    result = original_report(prefill_ns, token_times, requested, ids,
                             prompt_tokens=prompt_tokens,
                             device=f"{device}+cpu-backend")
    if ids is not None and os.environ.get("MLX_PAIR_IDS"):
        import json
        with open(os.environ["MLX_PAIR_IDS"], "a") as output:
            output.write(json.dumps({
                "requested": requested,
                "prompt_tokens": prompt_tokens,
                "ids": [int(i) for i in ids],
                "device": "cpu",
            }) + "\n")
    return result


bench_decode.report = capture_report
raise SystemExit(bench_decode.main())
