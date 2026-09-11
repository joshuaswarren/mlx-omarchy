#!/usr/bin/env bash
# CPU-only candidate build on jwm1-linux, OUTSIDE the lock. Run this in the
# gap between GPU windows (niced; the arms windows record loadavg, so do
# not overlap it with a wall-clock-sensitive GPU run without agreement).
# Produces: cand wheel, .venv-attn-cand, and the cand suite binaries
# /tmp/fam-attn, /tmp/fast-attn, /tmp/rt-attn.
set -euo pipefail
cd ~/src/mlx-bf16-prefill-attn
R=receipts/2026-09-11-bf16-prefill-attention
mkdir -p "$R/m1-logs"

echo "== cand commit ==" | tee "$R/m1-logs/cand-build.log"
git rev-parse HEAD | tee "$R/m1-logs/cand-commit.txt"

echo "== cand wheel build (niced) ==" | tee -a "$R/m1-logs/cand-build.log"
nice -n 10 env DEV_RELEASE=1 CMAKE_BUILD_PARALLEL_LEVEL=4 \
  scripts/build-wheel.sh 2>&1 | tail -5 | tee -a "$R/m1-logs/cand-build.log"
cand_wheel=$(ls -t dist/mlx_omarchy-*.whl | head -1)
sha256sum "$cand_wheel" | tee "$R/m1-logs/candidate-wheel.sha256"
realpath "$cand_wheel" | tee "$R/m1-logs/cand-wheel-path.txt"

echo "== cand venv ==" | tee -a "$R/m1-logs/cand-build.log"
rm -rf .venv-attn-cand
python3.14 -m venv --system-site-packages .venv-attn-cand
.venv-attn-cand/bin/pip install -q "$cand_wheel" "mlx-lm==0.31.3" \
  "transformers==5.17.0"

echo "== suite binaries (niced) ==" | tee -a "$R/m1-logs/cand-build.log"
if [[ ! -f .work/build-attn/CMakeCache.txt ]]; then
  nice -n 10 cmake -S .work/mlx -B .work/build-attn \
    -DBUILD_TESTING=ON \
    -DMLX_BUILD_OMARCHY=ON \
    -DMLX_BUILD_PYTHON_BINDINGS=OFF \
    -DMLX_BUILD_CPU=OFF \
    -DMLX_BUILD_BENCHMARKS=OFF \
    -DMLX_BUILD_EXAMPLES=OFF \
    -DMLX_BUILD_METAL=OFF \
    -DMLX_BUILD_CUDA=OFF \
    > "$R/m1-logs/cmake-configure.log" 2>&1
fi
nice -n 10 cmake --build .work/build-attn -j4 --target \
  omarchy_matmul_family_tests omarchy_fast_ops_tests omarchy_runtime_tests \
  2>&1 | tail -3 | tee -a "$R/m1-logs/cand-build.log"
cp .work/build-attn/tests/omarchy/omarchy_matmul_family_tests /tmp/fam-attn
cp .work/build-attn/tests/omarchy/omarchy_fast_ops_tests /tmp/fast-attn
cp .work/build-attn/tests/omarchy/omarchy_runtime_tests /tmp/rt-attn
sha256sum /tmp/fam-attn /tmp/fast-attn /tmp/rt-attn \
  | tee "$R/m1-logs/suite-binaries-attn.sha256"
echo "BUILD-CAND-OK"
