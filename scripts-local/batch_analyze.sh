#!/bin/sh
# Detached analyzers for the batching profiles; writes DONE marker.
set -eu
cd "$HOME/src/mlx-omarchy-bqm1-build"
python3 scripts/profile_analyze.py "$HOME/benchq/batch-p-bf16.jsonl" \
  --compute-h .work/mlx/mlx/backend/omarchy/compute.h \
  --markers "$HOME/benchq/batch-m-bf16.jsonl" \
  > "$HOME/benchq/batch-analyze-bf16.txt" 2>&1
python3 scripts/profile_analyze.py "$HOME/benchq/batch-p-q4.jsonl" \
  --compute-h .work/mlx/mlx/backend/omarchy/compute.h \
  --markers "$HOME/benchq/batch-m-q4.jsonl" \
  > "$HOME/benchq/batch-analyze-q4.txt" 2>&1
date "+done %F %T" > "$HOME/benchq/batch-analyze-DONE"
