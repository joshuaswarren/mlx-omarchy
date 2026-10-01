#!/bin/bash
# Jw16LevelBatch W2 gate A: 5 interleaved control/candidate rounds, full cell
# set. Control = serving venv, flag unset. Candidate = lb1 venv + LEVEL_BATCH.
# run-linux-cells.sh owns its window (stop, flock, restore) and takes the
# gpuwin mutex itself; do NOT nest in gpuwin.sh.
set -u
OUT=/var/tmp/lb1; mkdir -p "$OUT"
CTL_PY=/var/tmp/v072-venv-fused/bin/python3
CAND_PY=/var/tmp/lb1-venv/bin/python3
for i in 1 2 3 4 5; do
  echo "=== round $i control $(date -u +%FT%TZ)"
  PY=$CTL_PY TAG=lb1-ctl-r$i bash /var/tmp/appbar/run-linux-cells.sh \
    "$OUT/cells-ctl-r$i" d64 d128 d256 d512 pf512 > "$OUT/cells-ctl-r$i.log" 2>&1
  echo "=== round $i candidate $(date -u +%FT%TZ)"
  PY=$CAND_PY MLX_OMARCHY_LEVEL_BATCH=1 TAG=lb1-lb-r$i bash /var/tmp/appbar/run-linux-cells.sh \
    "$OUT/cells-lb-r$i" d64 d128 d256 d512 pf512 > "$OUT/cells-lb-r$i.log" 2>&1
done
echo CELLS-GATE-DONE
