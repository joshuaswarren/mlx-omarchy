#!/usr/bin/env bash
# Window orchestrator on jwm1-linux for the BF16 decode GEMV landing.
# SINGLE top-level flock: all GPU phases run inside this one acquisition;
# nothing nested re-acquires (per Main).
# Usage: bash receipts/2026-09-10-bf16-decode-gemv-land/m1_window.sh
set -uo pipefail
cd ~/src/mlx-land
R=receipts/2026-09-10-bf16-decode-gemv-land

exec 9>/tmp/m1-gpu.lock
echo "$(date -Is) waiting for /tmp/m1-gpu.lock"
flock 9
echo "$(date -Is) lock acquired"

{
  echo "== phase 0: driver identities (fork + stock) =="
  vulkaninfo --summary 2>/dev/null | sed -n '1,40p' > "$R/m1-logs/drivers-fork.txt"
  VK_DRIVER_FILES=/home/joshuawarren/stock-mesa/stock-icd.json \
    vulkaninfo --summary 2>/dev/null | sed -n '1,40p' > "$R/m1-logs/drivers-stock.txt"
  echo "== phase 1: matrix =="
  bash "$R/m1_matrix.sh"
  echo "== phase 2: suites, both drivers =="
  bash "$R/m1_suites.sh"
} > "$R/window.log" 2>&1
rc=$?
echo "$(date -Is) window complete rc=$rc"
exit $rc
