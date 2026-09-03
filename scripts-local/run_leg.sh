#!/bin/sh
# One measurement leg: RUNS timed mlx_lm generate runs under an optional taskset spec.
# usage: run_leg.sh <venv-bin> <model-dir> <prompt-file> <taskset-spec-or--> <runs> <tag>
# Logs land in ~/benchq/logs/<tag>.runN.log. Env: MLX_DISABLE_COMPILE=1 always set.
set -eu
VENV_BIN=$1
MODEL=$2
PROMPT_FILE=$3
TS=$4
RUNS=$5
TAG=$6
PROMPT=$(cat "$PROMPT_FILE")
LOGDIR="$HOME/benchq/logs"
mkdir -p "$LOGDIR"
i=1
while [ "$i" -le "$RUNS" ]; do
  LOG="$LOGDIR/$TAG.run$i.log"
  if [ "$TS" = "-" ]; then
    MLX_DISABLE_COMPILE=1 "$VENV_BIN" -m mlx_lm generate \
      --model "$MODEL" --prompt "$PROMPT" \
      --max-tokens 32 --temp 0 --seed 0 >"$LOG" 2>&1
  else
    MLX_DISABLE_COMPILE=1 taskset -c "$TS" "$VENV_BIN" -m mlx_lm generate \
      --model "$MODEL" --prompt "$PROMPT" \
      --max-tokens 32 --temp 0 --seed 0 >"$LOG" 2>&1
  fi
  echo "done $TAG run$i"
  i=$((i + 1))
done
