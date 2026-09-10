#!/bin/sh
set -eu
ROOT=/home/joshuawarren/src/mlx-SubmissionGaps
PY=$ROOT/.venv-diagnostics/bin/python
OUT=$1
DRIVER=${2:-/tmp/asahi_coopmat_icd.json}
mkdir -p "$OUT/m262" "$OUT/m1053"
exec timeout 1800 flock -w 2400 /tmp/m1-gpu.lock env \
  VK_DRIVER_FILES="$DRIVER" AGX_SIMDMAT=1 HF_HUB_OFFLINE=1 \
  MLX_DISABLE_COMPILE=1 MESA_SHADER_CACHE_DISABLE=true \
  /bin/sh -c '
    for length in 262 1053; do
      MLX_OMARCHY_GPU_PROFILE="'$OUT'/m$length/profile.jsonl" \
        "'$PY'" "'$ROOT'/scripts/profile_generate.py" \
        --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
        --prompt "$(cat "'$ROOT'/receipts/2026-09-10-submission-gaps/prompt-$length.txt")" \
        --max-tokens 2 --temp 0 --seed 0 \
        --markers "'$OUT'/m$length/markers.jsonl" \
        > "'$OUT'/m$length/run.log" 2>&1
      "'$PY'" "'$ROOT'/receipts/2026-09-10-submission-gaps/analyze_submission_gaps.py" \
        "'$OUT'/m$length/profile.jsonl" "'$OUT'/m$length/markers.jsonl" \
        "'$OUT'/m$length/summary.json" > "'$OUT'/m$length/summary.log"
    done
  '
