#!/bin/sh
# q4 analysis detached (wrapper so redirects live in a file, not inline).
cd "$HOME/src/mlx-omarchy-bqm1-build"
python3 scripts/profile_analyze.py "$HOME/benchq/diag-p-q4.jsonl" \
  --compute-h .work/mlx/mlx/backend/omarchy/compute.h \
  --markers "$HOME/benchq/diag-m-q4.jsonl" \
  > "$HOME/benchq/diag-analyze-q4.txt" 2>&1
