#!/usr/bin/env bash
# Legs-only GPU window for the Q4 pair-map receipt (accuracy arms and the
# f64 comparison already ran). ONE top-level flock, capped wait.
set -euo pipefail
WHEEL="$1"
REPS="${2:-3}"
ROOT="$HOME/src/mlx-pairmap-m1"
OUT="$ROOT/receipts-workdata"
PY="$ROOT/.work/venv-run/bin/python"

echo "== legs window start $(date -u +%Y-%m-%dT%H:%M:%SZ) =="
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

"$PY" receipts-workdata/run_pair_legs.py --python "$PY" --wheel "$WHEEL" \
  --reps "$REPS" --out "$OUT/legs" 2>&1 | tail -12
echo "flock release $(date -u +%H:%M:%SZ)"
flock -u 9
echo "== legs window end $(date -u +%Y-%m-%dT%H:%M:%SZ) =="
