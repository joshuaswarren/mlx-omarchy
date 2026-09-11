#!/usr/bin/env bash
# GPU window for the Q4 pair-map receipt. ONE top-level flock, capped
# wait; everything GPU-related happens inside. Run on jwm1:
#   bash receipts-workdata/window.sh <wheel-path> <rep-count>
set -euo pipefail
WHEEL="$1"
REPS="${2:-3}"
ROOT="$HOME/src/mlx-pairmap-m1"
OUT="$ROOT/receipts-workdata"
PY="$ROOT/.work/venv-run/bin/python"

echo "== window start $(date -u +%Y-%m-%dT%H:%M:%SZ) =="
la1=$(cut -d' ' -f1 /proc/loadavg)
echo "loadavg before window: $la1"
if ! python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) < 1.0 else 1)" "$la1"; then
  echo "QUIET_GATE_FAILED loadavg=$la1"
  exit 2
fi

exec 9>/tmp/m1-gpu.lock
if ! flock -w 600 9; then
  echo "FLOCK_TIMEOUT after 600s"
  exit 3
fi
echo "flock acquired $(date -u +%H:%M:%SZ)"

pacman -Q mesa-honeykrisp-omarchy linux-asahi || true
sha256sum "$WHEEL"

echo "== engagement smoke: single-weight qmm under both maps =="
(cd "$ROOT" && "$PY" receipts-workdata/run_accuracy_arm.py --out "$OUT/accuracy" 2>&1 | tail -1)
(cd "$ROOT" && MLX_OMARCHY_QMM_VEC_Q4_PAIR=1 "$PY" receipts-workdata/run_accuracy_arm.py --out "$OUT/accuracy" 2>&1 | tail -1)

echo "== f64 comparison (CPU, inside window for convenience) =="
"$PY" receipts-workdata/compare_accuracy.py --dir "$OUT/accuracy" \
  --out "$OUT/accuracy/report.json" | tail -5

echo "== paired legs: $REPS reps x 4 arms x 3 legs =="
"$PY" receipts-workdata/run_pair_legs.py --wheel "$WHEEL" \
  --reps "$REPS" --out "$OUT/legs" 2>&1 | tail -8

echo "flock release $(date -u +%H:%M:%SZ)"
flock -u 9
echo "== window end $(date -u +%Y-%m-%dT%H:%M:%SZ) =="
