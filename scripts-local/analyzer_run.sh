#!/bin/sh
# Analyze both profiling streams; writes results and a DONE marker.
set -eu
cd "$HOME/src/mlx-omarchy-bqm1-build"
python3 scripts/profile_analyze.py "$HOME/benchq/diag-p-q4.jsonl" \
  --compute-h .work/mlx/mlx/backend/omarchy/compute.h \
  --markers "$HOME/benchq/diag-m-q4.jsonl" \
  > "$HOME/benchq/diag-analyze-q4.txt" 2>&1
python3 scripts/profile_analyze.py "$HOME/benchq/diag-p-bf16.jsonl" \
  --compute-h .work/mlx/mlx/backend/omarchy/compute.h \
  --markers "$HOME/benchq/diag-m-bf16.jsonl" \
  > "$HOME/benchq/diag-analyze-bf16.txt" 2>&1
date "+done %F %T" > "$HOME/benchq/analyzer-DONE"
