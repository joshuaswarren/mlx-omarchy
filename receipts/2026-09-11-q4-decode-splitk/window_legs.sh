#!/usr/bin/env bash
# Accuracy + legs GPU window for the Q4 split-K receipt. ONE top-level
# flock; quiet-gated; driver verified in-window.
#   bash receipts/2026-09-11-q4-decode-splitk/window_legs.sh <wheel> <reps> <splits>
set -euo pipefail
WHEEL="$1"
REPS="${2:-3}"
SPLITS="${3:-4}"
ROOT="$HOME/src/mlx-Q4DecodeSplitK"
OUT="$ROOT/receipts/2026-09-11-q4-decode-splitk/workdata"
PY="$ROOT/.work/venv-run/bin/python"
echo "== legs window start $(date -u +%Y-%m-%dT%H:%M:%SZ) =="
la1=$(cut -d' ' -f1 /proc/loadavg)
echo "loadavg before window: $la1"
if ! python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) < 1.0 else 1)" "$la1"; then
  echo "QUIET_GATE_FAILED loadavg=$la1"
  exit 2
fi
exec 9>/tmp/m1-gpu.lock
if ! flock -w 900 9; then
  echo "FLOCK_TIMEOUT after 900s"
  exit 3
fi
echo "flock acquired $(date -u +%H:%M:%SZ)"
pacman -Q mesa-honeykrisp-omarchy linux-asahi || true
sha256sum "$WHEEL"
echo "== accuracy arms: base + splitk (fused-group Add epilogue path) =="
(cd "$ROOT" && "$PY" receipts/2026-09-11-q4-decode-splitk/run_accuracy_arm.py --out "$OUT/accuracy" 2>&1 | tail -1)
(cd "$ROOT" && MLX_OMARCHY_QMM_VEC_Q4_SPLITK="$SPLITS" "$PY" receipts/2026-09-11-q4-decode-splitk/run_accuracy_arm.py --out "$OUT/accuracy" 2>&1 | tail -1)
echo "== f64 comparison (CPU, inside window for convenience) =="
"$PY" receipts/2026-09-11-q4-decode-splitk/compare_accuracy.py --dir "$OUT/accuracy" \
  --out "$OUT/accuracy/report.json" | tail -5
echo "== paired legs: $REPS reps x 4 arms x 3 legs, splitk=$SPLITS =="
"$PY" receipts/2026-09-11-q4-decode-splitk/run_splitk_legs.py --splits "$SPLITS" \
  --python "$PY" --wheel "$WHEEL" --reps "$REPS" --out "$OUT/legs" 2>&1 | tail -12
echo "flock release $(date -u +%H:%M:%SZ)"
flock -u 9
echo "== legs window end $(date -u +%Y-%m-%dT%H:%M:%SZ) =="
