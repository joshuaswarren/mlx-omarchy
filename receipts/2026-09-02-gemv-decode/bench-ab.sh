#!/usr/bin/env bash
# DecodeGemvPath A/B benchmark on jwm1, matching the published receipts'
# mlx-lm invocations (receipts/2026-09-01-m1-same-chip-parity.md and
# m1-qualification-2026-09-02/m1-models). Greedy, temp 0, seed 0, warm
# second run reported.
set -u
PY="$HOME/src/mlx-omarchy-gemv/.work/venv-gemv/bin/python"
Q4="$HOME/models/Qwen2.5-0.5B-Instruct-4bit-mlx"
BF16="$HOME/models/Qwen2.5-0.5B-Instruct-bf16-mlx"
run() {
  local tag="$1" model="$2" prompt="$3"
  for attempt in 1 2; do
    echo "=== ${tag} attempt${attempt} ==="
    "$PY" -m mlx_lm generate --model "$model" \
      --prompt "$prompt" --max-tokens 32 --temp 0 --seed 0 2>/dev/null
    echo "rc=$?"
  done
}
run bf16_greedy_hi "$BF16" 'Hi'
run bf16_greedy_france "$BF16" 'What is the capital of France? Answer in one word.'
run q4_greedy_hi "$Q4" 'Hi'
run q4_greedy_france "$Q4" 'What is the capital of France? Answer in one word.'
echo BENCH_DONE
