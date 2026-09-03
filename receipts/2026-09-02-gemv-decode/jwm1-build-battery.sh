#!/usr/bin/env bash
# DecodeGemvPath jwm1 build + battery. Run in the isolated -gemv worktree.
# Single CPU online: -j1, nohup, poll. Mirrors the m1-qualification battery
# (each test binary twice, rc gate) and then builds + installs the wheel.
set -u
ROOT="$HOME/src/mlx-omarchy-gemv"
cd "$ROOT" || exit 1
echo "[gemv] stage $(date)"
./scripts/prepare-mlx.sh || { echo PREPARE_FAILED; exit 1; }
echo "[gemv] configure $(date)"
cmake -S "$ROOT/.work/mlx" -B "$ROOT/.work/build" \
  -DMLX_BUILD_OMARCHY=ON \
  -DMLX_BUILD_CPU=ON \
  -DMLX_BUILD_METAL=OFF \
  -DMLX_BUILD_CUDA=OFF \
  -DMLX_BUILD_TESTS=ON \
  -DMLX_BUILD_EXAMPLES=OFF \
  -DMLX_BUILD_BENCHMARKS=OFF \
  -DMLX_BUILD_PYTHON_BINDINGS=ON || { echo CONFIGURE_FAILED; exit 1; }
echo "[gemv] build $(date)"
cmake --build .work/build -j1 2>&1 | tail -5
rc=${PIPESTATUS[0]}
[ "$rc" -eq 0 ] || { echo "BUILD_FAILED rc=$rc"; exit 1; }
echo "[gemv] build ok $(date)"

echo "[gemv] battery $(date)"
BINARIES="$(ls .work/build/tests/omarchy/omarchy_*_tests | sort)"
echo "$BINARIES" | wc -l
FAIL=0
for pass in 1 2; do
  for bin in $BINARIES; do
    name="$(basename "$bin")"
    start=$(date +%s)
    if .work/build/tests/omarchy/"$name" > "/tmp/gemv_${pass}_${name}.log" 2>&1; then
      echo "pass${pass} ${name} rc=0 secs=$(($(date +%s)-start))"
    else
      echo "pass${pass} ${name} rc=FAIL secs=$(($(date +%s)-start))"
      FAIL=1
    fi
  done
done
echo "[gemv] battery done $(date) FAIL=$FAIL"

echo "[gemv] counts (pass2)"
for bin in $BINARIES; do
  name="$(basename "$bin")"
  grep -E "test cases|assertions" "/tmp/gemv_2_${name}.log" | tr '\n' ' ' | sed "s|^|${name}: |"
  echo
done

echo "[gemv] wheel $(date)"
export CMAKE_ARGS="-DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_CPU=ON -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF -DMLX_BUILD_TESTS=OFF -DMLX_BUILD_EXAMPLES=OFF -DMLX_BUILD_BENCHMARKS=OFF"
export CMAKE_BUILD_PARALLEL_LEVEL=1
mkdir -p dist
"$ROOT/.work/venv-gemv/bin/pip" wheel --no-build-isolation --no-deps \
  --wheel-dir "$ROOT/dist" "$ROOT/.work/mlx" > /tmp/gemv_wheel.log 2>&1 \
  || { echo WHEEL_FAILED; tail -20 /tmp/gemv_wheel.log; exit 1; }
ls "$ROOT"/dist/mlx_omarchy-*.whl
"$ROOT/.work/venv-gemv/bin/pip" install -q --force-reinstall --no-deps \
  "$ROOT"/dist/mlx_omarchy-*.whl || { echo INSTALL_FAILED; exit 1; }
"$ROOT/.work/venv-gemv/bin/python" -c "import mlx.core as mx; print('mlx import ok', mx.device_info)" 2>&1 | tail -1
echo GEMV_SLICE_DONE
