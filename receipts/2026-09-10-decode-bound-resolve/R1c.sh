#!/bin/bash
# Window R1c (DecodeBoundResolve): tape-engagement proof on the
# host-trace wheel - eager vs compiled decode leg, C++ phase counters.
set -euo pipefail
cd ~/src/mlx-DecodeBoundResolve
HP=~/src/mlx-HostPathOverhead
PY=$HP/.work/venv-hpo/bin/python
HPH=$HP/receipts/2026-09-10-hostpath/hostphases.py
M=~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3
mkdir -p results/tape
MLX_DISABLE_COMPILE=1 MLX_OMARCHY_HOST_TRACE=/tmp/t-eager.json \
  "$PY" "$HPH" --model "$M" --prompt "Hi" --tokens 32 --warmup 4 \
  --trace-out /tmp/t-eager.json --out results/tape/eager.json
env -u MLX_DISABLE_COMPILE MLX_OMARCHY_HOST_TRACE=/tmp/t-comp.json \
  "$PY" "$HPH" --model "$M" --prompt "Hi" --tokens 32 --warmup 4 \
  --trace-out /tmp/t-comp.json --out results/tape/compiled.json
python3 - << "PY"
import json
for tag in ("eager", "compiled"):
    d = json.load(open(f"results/tape/{tag}.json"))
    t = d["trace"]
    keep = ("backend_eval", "disp_total", "desc_setup", "vkcmd",
            "submit_total", "finalize", "synchronize", "replay_hits",
            "replay_records", "join_wait")
    print(tag, "digest", d["digest"], "tok/s %.2f" % d["decode_tok_s"])
    for k in keep:
        v = t.get(k) or {}
        print(f"  {k:14s} ns={v.get('ns',0):>12d} hits={v.get('hits',0)}")
PY
echo R1C-DONE
