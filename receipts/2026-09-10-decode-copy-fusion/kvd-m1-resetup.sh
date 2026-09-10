#!/usr/bin/env bash
# CPU-only re-prep of ~/src/mlx-KvDirect after the extension lands on
# origin: fetch the branch, rebuild both wheels (glslc, no GPU lock),
# rebuild the five test binaries, reinstall the release wheel into
# .venv-accept. Run BEFORE kvd-m1-window.sh.
set -euo pipefail
cand=~/src/mlx-KvDirect
log=~/src/mlx-KvDirect-prep.log
exec >> "$log" 2>&1
echo "=== re-setup start $(date -u +%FT%TZ)"
cd "$cand"
git fetch -q origin wave/DecodeCopyFusion
git checkout -q wave/DecodeCopyFusion
git reset -q --hard FETCH_HEAD
echo "checkout: $(git rev-parse --short=7 HEAD)"
unset HK_PERF HK_PERFTEST MLX_OMARCHY_SOURCE_COMMIT MLX_OMARCHY_LOCAL_VERSION MLX_OMARCHY_GPU_PROFILE PYTHONPATH VK_DRIVER_FILES
export DEV_RELEASE=1 CMAKE_BUILD_PARALLEL_LEVEL=4
rm -rf .work/mlx/build .work/build-wheel dist/*.whl wheels/diag/*.whl 2>/dev/null || true
nice -n 5 bash scripts/build-wheel.sh --diagnostics > build-diag.log 2>&1
mkdir -p wheels/diag
mv dist/mlx_omarchy-*.whl wheels/diag/
nice -n 5 bash scripts/build-wheel.sh > build-release.log 2>&1
echo "wheels: $(ls dist/*.whl wheels/diag/*.whl)"
cmake -S .work/mlx -B .work/build-accept -DCMAKE_BUILD_TYPE=Release -DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_CPU=ON -DMLX_BUILD_TESTS=ON -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF > test-build.log 2>&1
nice -n 5 cmake --build .work/build-accept --target omarchy_kv_ops_tests omarchy_runtime_tests omarchy_fast_ops_tests omarchy_fast_regression_tests omarchy_fused_chain_tests -j 4 >> test-build.log 2>&1
echo "test binaries rebuilt"
.venv-accept/bin/pip install -q --no-deps --force-reinstall dist/mlx_omarchy-*.whl
sha256sum dist/*.whl wheels/diag/*.whl
echo "=== re-setup done $(date -u +%FT%TZ)"
echo RESETUP_DONE
