#!/bin/sh
# Decode A/B: three wheels x two precisions, 8 cores unpinned, 5 runs each.
set -eu
cd ~/benchq
BF16="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-bf16/snapshots/56d07e766edd7159fbe12ed12d9cf114bf38bf1e"
Q4="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3"
echo "START $(date '+%F %T') load: $(uptime)"
sh run_leg.sh ~/venv-bqm1-0606/bin/python "$BF16" prompt-bf16.txt - 5 D0606-BF16
sh run_leg.sh ~/venv-bqm1-0606/bin/python "$Q4" prompt-q4.txt - 5 D0606-Q4
sh run_leg.sh ~/venv-bqm1-0928/bin/python "$BF16" prompt-bf16.txt - 5 D0928-BF16
sh run_leg.sh ~/venv-bqm1-0928/bin/python "$Q4" prompt-q4.txt - 5 D0928-Q4
sh run_leg.sh ~/venv-bqm1-0512/bin/python "$BF16" prompt-bf16.txt - 5 D0512-BF16
sh run_leg.sh ~/venv-bqm1-0512/bin/python "$Q4" prompt-q4.txt - 5 D0512-Q4
echo "END $(date '+%F %T') load: $(uptime)"
python3 parse_legs.py \
  'Prompt: [0-9]+ tokens, ([0-9.]+) tokens-per-sec' \
  'Generation: [0-9]+ tokens, ([0-9.]+) tokens-per-sec' \
  D0606-BF16 D0606-Q4 D0928-BF16 D0928-Q4 D0512-BF16 D0512-Q4
