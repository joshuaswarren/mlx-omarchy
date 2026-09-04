#!/bin/sh
# Native TapeLayerIsolation tree at 7c3d6b4: 5 runs per step, 5/5 correct
# completions = switch "gone"; any Cos abort = "persists".
set -eu
V="$HOME/venv-bqm1-after/bin/python"
Q4="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3"
P="What is the capital of France?"
run_step() {
  TAG=$1
  shift
  GONE=0
  ABORT=0
  i=1
  while [ "$i" -le 5 ]; do
    LOG="$HOME/benchq/logs/tnt-$TAG.run$i.log"
    if env -u MLX_DISABLE_COMPILE -u MLX_OMARCHY_ALLOW_NON_APPLE \
        MLX_OMARCHY_ALLOW_UNSAFE_COMPILE=1 "$@" \
        "$V" -m mlx_lm generate \
        --model "$Q4" --prompt "$(cat "$HOME/benchq/prompt-france.txt")" \
        --max-tokens 32 --temp 0 --seed 0 > "$LOG" 2>&1 \
        && grep -q "Paris" "$LOG"; then
      GONE=$((GONE + 1))
    else
      ABORT=$((ABORT + 1))
      grep -E "Cos argument" "$LOG" | head -1
    fi
    i=$((i + 1))
  done
  if [ "$ABORT" = "0" ]; then
    echo "STEP $TAG: GONE (5/5 correct completions)"
  else
    echo "STEP $TAG: PERSISTS ($ABORT/5 aborted)"
  fi
}
run_step A MLX_OMARCHY_TAPE_PER_NODE_SUBMIT=1
run_step B MLX_OMARCHY_TAPE_FULL_BARRIERS=1
run_step C MLX_OMARCHY_TAPE_NO_REUSE=1
