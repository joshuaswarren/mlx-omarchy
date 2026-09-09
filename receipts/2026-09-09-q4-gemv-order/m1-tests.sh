#!/usr/bin/env bash
# Targeted omarchy test binaries for the Q4 GEMV order change. Run under
# `flock -w 600 /tmp/m1-gpu.lock timeout 7200` from the checkout root
# after the wheel build (the prepared tree under .work/mlx is reused).
set -euo pipefail
cd "$(dirname "$0")/../.."
out=receipts/2026-09-09-q4-gemv-order
cmake -S .work/mlx -B .work/build-accept -DCMAKE_BUILD_TYPE=Release -DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_CPU=ON -DMLX_BUILD_TESTS=ON -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF > "$out/test-build.log" 2>&1
cmake --build .work/build-accept --target omarchy_matmul_family_tests omarchy_fast_ops_tests omarchy_runtime_tests -j 4 >> "$out/test-build.log" 2>&1
for suite in omarchy_matmul_family_tests omarchy_fast_ops_tests omarchy_runtime_tests; do
    if .work/build-accept/tests/omarchy/"$suite" > "$out/$suite.log" 2>&1; then
        printf 'SUITE_PASS=%s\n' "$suite"
    else
        printf 'SUITE_FAIL=%s\n' "$suite"
    fi
    tail -3 "$out/$suite.log"
done
