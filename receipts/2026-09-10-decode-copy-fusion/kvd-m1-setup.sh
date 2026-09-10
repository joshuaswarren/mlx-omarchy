#!/usr/bin/env bash
# CPU-only prep for the DecodeCopyFusion M1 window. NO GPU lock held here:
# wheel builds (glslc) and test-binary builds are CPU work and run niced
# so concurrent windows are not starved.
#   kvd-m1-setup.sh
set -euo pipefail
cand=~/src/mlx-KvDirect
log=~/src/mlx-KvDirect-prep.log
exec >> "$log" 2>&1
echo "=== setup start $(date -u +%FT%TZ)"
if [ ! -e "$cand/.git" ]; then
  git init -q "$cand"
  git -C "$cand" remote add origin https://github.com/joshuaswarren/mlx-omarchy.git
fi
cd "$cand"
git fetch -q origin wave/DecodeCopyFusion
git checkout -q -B wave/DecodeCopyFusion FETCH_HEAD
git reset -q --hard FETCH_HEAD
echo "checkout: $(git rev-parse --short=7 HEAD)"
# engine script for bench_matrix (receipt-local tool, not in-tree)
cp /tmp/kvd-bench_decode_identity.py scripts/bench_decode_identity.py
# window scripts
cp /tmp/kvd-paired.py /tmp/kvd-window-inner.sh scripts/ 2>/dev/null || true
unset HK_PERF HK_PERFTEST MLX_OMARCHY_SOURCE_COMMIT MLX_OMARCHY_LOCAL_VERSION MLX_OMARCHY_GPU_PROFILE PYTHONPATH VK_DRIVER_FILES
export DEV_RELEASE=1 CMAKE_BUILD_PARALLEL_LEVEL=4
# wheels: diagnostics first (moves to wheels/diag), release second (stays in dist/)
rm -rf .work/mlx/build .work/build-wheel dist/*.whl wheels/diag/*.whl 2>/dev/null || true
nice -n 5 bash scripts/build-wheel.sh --diagnostics > build-diag.log 2>&1
mkdir -p wheels/diag
mv dist/mlx_omarchy-*.whl wheels/diag/
nice -n 5 bash scripts/build-wheel.sh > build-release.log 2>&1
echo "wheels: $(ls dist/*.whl wheels/diag/*.whl)"
# test binaries (CPU)
cmake -S .work/mlx -B .work/build-accept -DCMAKE_BUILD_TYPE=Release -DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_CPU=ON -DMLX_BUILD_TESTS=ON -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF > test-build.log 2>&1
nice -n 5 cmake --build .work/build-accept --target omarchy_kv_ops_tests omarchy_runtime_tests omarchy_fast_ops_tests omarchy_fast_regression_tests omarchy_fused_chain_tests -j 4 >> test-build.log 2>&1
echo "test binaries: $(ls .work/build-accept/tests/omarchy/ | grep _tests | tr '\n' ' ')"
# venv: wheel first, then mlx-lm --no-deps (provisioning rule)
rm -rf .venv-accept
python3 -m venv .venv-accept
.venv-accept/bin/pip install -q --no-deps dist/mlx_omarchy-*.whl
.venv-accept/bin/pip install -q --no-deps "mlx-lm==0.31.3" numpy
.venv-accept/bin/pip install -q huggingface_hub transformers sentencepiece jinja2 protobuf
sha256sum dist/*.whl wheels/diag/*.whl
echo "=== setup done $(date -u +%FT%TZ)"
echo SETUP_DONE
