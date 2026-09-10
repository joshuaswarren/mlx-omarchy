#!/usr/bin/env bash
# omarchy suites on the M1 (fork driver) + ULP capture/truth, all four cells.
# MUST run while holding /tmp/m1-gpu.lock (outer flock in m1_window.sh).
# --smoke-only: just one ULP capture cell, to fail fast before the matrix.
set -uo pipefail
cd ~/src/mlx-requal
R=receipts/2026-09-10-bf16-decode-gemv-requal
out=receipt-requal
mkdir -p "$out/suites" "$out/ulp"
ulimit -c 0

if [[ ${1:-} == --smoke-only ]]; then
  .venv-requal-base/bin/python "$R/m1_ulp_capture.py" "$out/ulp/base-fork" \
    || echo "SMOKE-CAPTURE-FAILED"
  exit 0
fi

echo "== omarchy_matmul_family_tests (M1 fork) =="
timeout 1800 .work/build-requal/tests/omarchy/omarchy_matmul_family_tests \
  --out="$out/suites/m1-fork-matmul-family.log"
echo "matmul_family_rc=$?"

echo "== omarchy_runtime_tests (M1 fork) =="
timeout 900 .work/build-requal/tests/omarchy/omarchy_runtime_tests \
  --out="$out/suites/m1-fork-runtime.log"
echo "runtime_rc=$?"
tail -4 "$out/suites/m1-fork-matmul-family.log" "$out/suites/m1-fork-runtime.log"

echo "== ULP capture: 4 cells =="
.venv-requal-base/bin/python "$R/m1_ulp_capture.py" "$out/ulp/base-fork"
VK_DRIVER_FILES=/home/joshuawarren/stock-mesa/stock-icd.json \
  .venv-requal-base/bin/python "$R/m1_ulp_capture.py" "$out/ulp/base-stock"
.venv-requal-cand/bin/python "$R/m1_ulp_capture.py" "$out/ulp/cand-fork"
VK_DRIVER_FILES=/home/joshuawarren/stock-mesa/stock-icd.json \
  .venv-requal-cand/bin/python "$R/m1_ulp_capture.py" "$out/ulp/cand-stock"

echo "== ULP truth (CPU f64) =="
.venv-requal-base/bin/python "$R/m1_ulp_truth.py" "$out/ulp/truth.json" \
  base_fork="$out/ulp/base-fork" base_stock="$out/ulp/base-stock" \
  cand_fork="$out/ulp/cand-fork" cand_stock="$out/ulp/cand-stock"
echo "SUITES-ULP-OK"
