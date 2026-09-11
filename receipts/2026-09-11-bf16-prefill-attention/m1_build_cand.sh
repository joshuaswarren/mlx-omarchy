#!/usr/bin/env bash
# CPU-only candidate build on jwm1-linux, OUTSIDE the lock. Run this in the
# gap between GPU windows. Cores 0-1 only, niced: keeps 1-min loadavg under
# the peer's 3.0 quarantine threshold while their arms window runs.
# Produces: cand wheel, .venv-attn-cand, and the cand suite binaries
# /tmp/fam-attn, /tmp/fast-attn, /tmp/rt-attn.
set -euo pipefail
cd ~/src/mlx-bf16-prefill-attn
R=receipts/2026-09-11-bf16-prefill-attention
RUN="taskset -c 0,1 nice -n 10"
mkdir -p "$R/m1-logs"

echo "== cand commit ==" | tee "$R/m1-logs/cand-build.log"
git rev-parse HEAD | tee "$R/m1-logs/cand-commit.txt"

echo "== cand wheel build (taskset 0-1, niced) ==" | tee -a "$R/m1-logs/cand-build.log"
$RUN env DEV_RELEASE=1 CMAKE_BUILD_PARALLEL_LEVEL=2 \
  scripts/build-wheel.sh 2>&1 | tail -5 | tee -a "$R/m1-logs/cand-build.log"
cand_wheel=$(ls -t dist/mlx_omarchy-*.whl | head -1)
sha256sum "$cand_wheel" | tee "$R/m1-logs/candidate-wheel.sha256"
realpath "$cand_wheel" | tee "$R/m1-logs/cand-wheel-path.txt"

echo "== cand venv ==" | tee -a "$R/m1-logs/cand-build.log"
rm -rf .venv-attn-cand
python3.14 -m venv --system-site-packages .venv-attn-cand
.venv-attn-cand/bin/pip install -q "$cand_wheel" "mlx-lm==0.31.3" \
  "transformers==5.17.0"

echo "== suite binaries (taskset 0-1, niced) ==" | tee -a "$R/m1-logs/cand-build.log"
if [[ ! -f .work/build-attn/CMakeCache.txt ]]; then
  $RUN cmake -S .work/mlx -B .work/build-attn \
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
$RUN cmake --build .work/build-attn -j2 --target \
  omarchy_matmul_family_tests omarchy_fast_ops_tests omarchy_runtime_tests \
  2>&1 | tail -3 | tee -a "$R/m1-logs/cand-build.log"
cp .work/build-attn/tests/omarchy/omarchy_matmul_family_tests /tmp/fam-attn
cp .work/build-attn/tests/omarchy/omarchy_fast_ops_tests /tmp/fast-attn
cp .work/build-attn/tests/omarchy/omarchy_runtime_tests /tmp/rt-attn
sha256sum /tmp/fam-attn /tmp/fast-attn /tmp/rt-attn \
  | tee "$R/m1-logs/suite-binaries-attn.sha256"
echo "BUILD-CAND-OK"
