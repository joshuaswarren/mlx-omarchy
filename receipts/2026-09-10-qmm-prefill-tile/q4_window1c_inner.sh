#!/bin/sh
# Q4 1K prefill window 1c (inner): repetition pass for the two
# digest-safe arms (0 base, 1 m64). Confirms the screen verdict repeats.
set -eu
ulimit -c 0
OUT=$HOME/benchq/q4prefill-window1
cd "$OUT"
echo "[window1c] $(date) repetition pass arms 0,1"
for arm in 0 1 0 1; do
  MLX_OMARCHY_QMM_COOP_TILE=$arm AGX_SIMDMAT=1 \
    "$HOME/venv-q4tile/bin/python" -u q4_prefill_probe.py qmm \
    > "$OUT/probe-rep-arm$arm-$(date +%H%M%S).log" 2>&1
done
echo "[window1c] DONE"
