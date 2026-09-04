#!/bin/sh
# Native mlx_lm path at 7c25feb, no env: does the Cos abort reproduce?
set -eu
V="$HOME/venv-bqm1-tapeiso/bin/python"
Q4="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3"
LOG="$HOME/benchq/logs/tapeiso-native-baseline.log"
if env -u MLX_DISABLE_COMPILE -u MLX_OMARCHY_ALLOW_NON_APPLE \
    MLX_OMARCHY_ALLOW_UNSAFE_COMPILE=1 \
    "$V" -m mlx_lm generate \
    --model "$Q4" --prompt "What is the capital of France?" \
    --max-tokens 32 --temp 0 --seed 0 > "$LOG" 2>&1; then
  echo "NATIVE rc=0"
  grep -q "Paris" "$LOG" && echo "NATIVE: correct Paris (no corruption)" || echo "NATIVE: ran but wrong text"
  sed -n "2,3p" "$LOG"
else
  RC=$?
  echo "NATIVE rc=$RC"
  grep -E "Cos argument|RuntimeError|Sigmoid" "$LOG" | head -3
fi
