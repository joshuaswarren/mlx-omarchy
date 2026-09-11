#!/usr/bin/env bash
# Matrix-only retry after the harness cand_wheel fix. SINGLE top-level
# flock acquisition; suites already ran green in the prior window.
set -uo pipefail
cd ~/src/mlx-land
R=receipts/2026-09-10-bf16-decode-gemv-land

exec 9>/tmp/m1-gpu.lock
echo "$(date -Is) waiting for /tmp/m1-gpu.lock"
flock 9
echo "$(date -Is) lock acquired"
bash "$R/m1_matrix.sh"
rc=$?
echo "$(date -Is) matrix window complete rc=$rc"
exit $rc
