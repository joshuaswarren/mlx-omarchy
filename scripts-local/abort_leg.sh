#!/bin/sh
# Leg 3: abort proof at ff4b05a with compile genuinely ON.
# 20 runs; require 0 aborts and "Paris" in every output.
# Stops at the first failure and prints its log tail verbatim.
set -u
VENV="$HOME/venv-bqm1-DORD"
Q4="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3"
PROMPT="$HOME/benchq/prompt-france.txt"
PASS=0
FAIL=0
i=1
while [ "$i" -le 20 ]; do
  LOG="$HOME/benchq/logs/ABORT.run$i.log"
  if env -u MLX_DISABLE_COMPILE "$VENV/bin/python" -m mlx_lm generate \
      --model "$Q4" --prompt "$(cat "$PROMPT")" \
      --max-tokens 32 --temp 0 --seed 0 > "$LOG" 2>&1 \
      && grep -q "Paris" "$LOG"; then
    PASS=$((PASS + 1))
    echo "run$i PASS ($(grep -c 'tokens-per-sec' "$LOG") rate lines)"
  else
    FAIL=$((FAIL + 1))
    echo "run$i FAIL — failing output, verbatim tail:"
    tail -25 "$LOG"
    break
  fi
  i=$((i + 1))
done
echo "ABORT_PROOF: pass=$PASS fail=$FAIL (stopped at run $i)"
