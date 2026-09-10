#!/usr/bin/env bash
# Window orchestrator on jwm1-linux. SINGLE top-level flock: all GPU phases
# run inside this one acquisition; nothing nested re-acquires (per Main).
# Usage: bash m1_window.sh   (run on the M1; logs to receipt-requal/window.log)
set -uo pipefail
cd ~/src/mlx-requal
R=receipts/2026-09-10-bf16-decode-gemv-requal

exec 9>/tmp/m1-gpu.lock
echo "$(date -Is) waiting for /tmp/m1-gpu.lock"
flock 9
echo "$(date -Is) lock acquired"

{
  echo "== phase 1: matrix =="
  bash "$R/m1_matrix.sh"
  echo "== phase 2: suites + ulp =="
  bash "$R/m1_suites.sh"
} > receipt-requal/window.log 2>&1
rc=$?
echo "$(date -Is) window complete rc=$rc"
exit $rc
