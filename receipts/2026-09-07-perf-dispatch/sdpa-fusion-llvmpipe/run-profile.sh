#!/usr/bin/env bash
# Usage: run-profile.sh <label> <wheel> [extra env assignments...]
# Installs the wheel into a private venv copy and runs the pinned 64-token
# greedy decode with GPU profiling on llvmpipe.
set -euo pipefail
LABEL="$1"; WHEEL="$2"; shift 2
OUT=/tmp/f2-work/prof-$LABEL
mkdir -p "$OUT"
VENV=/tmp/f2-work/venv-$LABEL
if [[ ! -x "$VENV/bin/python" ]]; then
  cp -a /tmp/fi-work/venv "$VENV"
fi
"$VENV/bin/python" -m pip install --quiet --no-deps --force-reinstall "$WHEEL"
"$VENV/bin/python" -c 'import mlx.core as mx; print("mlx", mx.__version__)'
export MLX_OMARCHY_ALLOW_NON_APPLE=1 MLX_DISABLE_COMPILE=1 LP_NUM_THREADS=8
export MLX_OMARCHY_HANG_NO_PROGRESS_NS=60000000000
export MLX_OMARCHY_GPU_PROFILE="$OUT/profile.jsonl"
export HF_HUB_OFFLINE=1
for kv in "$@"; do export "$kv"; done
env | grep -E '^MLX_' | sort > "$OUT/env.txt"
timeout 3600 "$VENV/bin/python" /tmp/fi-work/gen_ids.py \
  --model /home/joshuawarren/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3/ --tokens 64 \
  --markers "$OUT/markers.jsonl" --ids-out "$OUT/ids.json" > "$OUT/gen.stdout" 2> "$OUT/gen.stderr"
"$VENV/bin/python" /tmp/mlx-release-f5-final/scripts/profile_analyze.py \
  "$OUT/profile.jsonl" --markers "$OUT/markers.jsonl" \
  --compute-h /tmp/f2-tree-after/overlay/mlx/backend/omarchy/compute.h \
  > "$OUT/analysis.txt" 2>&1 || true
echo PROFILE_DONE_$LABEL
