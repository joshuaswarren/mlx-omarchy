#!/usr/bin/env bash
# omarchy suites on the M1 (fork driver) + ULP capture/truth, all four cells.
# MUST run while holding /tmp/m1-gpu.lock (outer flock in m1_window.sh).
set -uo pipefail
cd ~/src/mlx-requal
R=receipts/2026-09-10-bf16-decode-gemv-requal
out=receipt-requal
mkdir -p "$out/suites" "$out/ulp"
ulimit -c 0

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
base_wheel="$HOME/src/mlx-main-b6d662a8/dist/mlx_omarchy-0.32.2.dev202609101916+b6d662a-cp314-cp314-linux_aarch64.whl"
# cells run with the matching venv python (wheel provenance gate is the
# bench-side hash check; here the venv IS the wheel)
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
