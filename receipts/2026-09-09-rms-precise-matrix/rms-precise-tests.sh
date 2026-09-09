#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/src/mlx-rms-round-screen"
out=receipts/2026-09-09-rms-precise-tests
mkdir "$out"
scripts/prepare-mlx.sh > "$out/prepare-tests.log" 2>&1
cmake -S .work/mlx -B .work/build-accept -DCMAKE_BUILD_TYPE=Release -DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_CPU=ON -DMLX_BUILD_TESTS=ON -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF > "$out/test-build.log" 2>&1
cmake --build .work/build-accept --target omarchy_matmul_family_tests omarchy_fast_ops_tests omarchy_runtime_tests -j 4 >> "$out/test-build.log" 2>&1
failed=0
for suite in omarchy_matmul_family_tests omarchy_fast_ops_tests omarchy_runtime_tests; do
    if .work/build-accept/tests/omarchy/"$suite" > "$out/$suite.log" 2>&1; then
        printf 'SUITE_PASS=%s\n' "$suite"
    else
        printf 'SUITE_FAIL=%s\n' "$suite"
        failed=1
    fi
done
exit "$failed"