#!/usr/bin/env bash
# Targeted omarchy test binaries on the M1 for one checkout (reuses the
# prepared tree under .work/mlx left by scripts/build-wheel.sh).
#   m1-tests.sh CHECKOUT OUT_DIR
# Run under `flock -w 900 /tmp/m1-gpu.lock timeout 7200`.
set -euo pipefail
cd "$1"
out="$2"
mkdir -p "$out"
git rev-parse --short=7 HEAD > "$out/source-commit.txt"
cmake -S .work/mlx -B .work/build-accept -DCMAKE_BUILD_TYPE=Release -DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_CPU=ON -DMLX_BUILD_TESTS=ON -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF > "$out/test-build.log" 2>&1
cmake --build .work/build-accept --target omarchy_fused_chain_tests omarchy_matmul_family_tests omarchy_fast_ops_tests omarchy_runtime_tests -j 4 >> "$out/test-build.log" 2>&1
for suite in omarchy_fused_chain_tests omarchy_matmul_family_tests omarchy_fast_ops_tests omarchy_runtime_tests; do
  if .work/build-accept/tests/omarchy/"$suite" > "$out/$suite.log" 2>&1; then
    printf 'SUITE_PASS=%s\n' "$suite"
  else
    printf 'SUITE_FAIL=%s\n' "$suite"
  fi
  tail -2 "$out/$suite.log"
done
