#!/usr/bin/env bash
# omarchy suites on the M1, fork AND stock drivers.
# MUST run while holding /tmp/m1-gpu.lock (outer flock in m1_window.sh).
set -uo pipefail
cd ~/src/mlx-land
R=receipts/2026-09-10-bf16-decode-gemv-land
out="$R/m1-suites"
mkdir -p "$out"
ulimit -c 0

for driver in fork stock; do
  if [[ $driver == stock ]]; then
    export VK_DRIVER_FILES=/home/joshuawarren/stock-mesa/stock-icd.json
  else
    unset VK_DRIVER_FILES
  fi
  echo "== omarchy_matmul_family_tests (M1 $driver) =="
  timeout 3600 .work/build-land/tests/omarchy/omarchy_matmul_family_tests \
    --out="$out/m1-$driver-matmul-family.log"
  echo "matmul_family_${driver}_rc=$?"

  echo "== omarchy_runtime_tests (M1 $driver) =="
  timeout 1800 .work/build-land/tests/omarchy/omarchy_runtime_tests \
    --out="$out/m1-$driver-runtime.log"
  echo "runtime_${driver}_rc=$?"
  tail -4 "$out/m1-$driver-matmul-family.log" "$out/m1-$driver-runtime.log"
done
echo "SUITES-OK"
