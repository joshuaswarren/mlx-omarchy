#!/bin/bash
# Jw16LevelBatch W2 engagement proof: profiled d32 with the flag off/on.
# Profiling pollutes timing (never used for claims); counts + digest valid.
set -u
OUT=/var/tmp/lb1; mkdir -p "$OUT"
PY=/var/tmp/lb1-venv/bin/python3
BENCH=$HOME/bench-scripts/qwen38-mlx-bench.py
MODEL=$(echo ~/.cache/huggingface/hub/models--SiddhJagani--Qwen3.8-2B-mlx-4Bit/snapshots/0867d98b*)
run () { # label env...
  local label="$1"; shift
  env "$@" MLX_OMARCHY_GPU_PROFILE=$OUT/prof-$label.jsonl \
    "$PY" "$BENCH" --model "$MODEL"/ --prompts "$HOME/bench-scripts/qwen38-2b-prompts.jsonl" \
    --limit 1 --new-tokens 32 --warmup 1 --passes 1 --prefill-tokens 512 \
    --label "lb1-eng-$label" --out "$OUT/eng-$label.json" > "$OUT/eng-$label.log" 2> "$OUT/eng-$label.err"
  grep -h ordered_records_sha256 "$OUT/eng-$label.err" | tail -1
}
run off X=1
run on MLX_OMARCHY_LEVEL_BATCH=1
for l in off on; do
  echo "== $l"
  grep -h '"k":"end"' "$OUT/prof-$l.jsonl" | python3 -c 'import json,sys
for line in sys.stdin:
    d = json.loads(line)
    print({k: v for k, v in d.items() if "barrier" in k or k in ("dispatches", "submissions")})'
done
echo ENG-DONE
