#!/usr/bin/env bash
# Window B on jwm1-linux: candidate wheel build + suites + paired matrix.
# ONE top-level flock held for the whole window (build included, per the
# jwm1 fleet rule that builds never overlap a neighbor's measurement
# window). Usage:
#   ssh jwm1 'cd ~/src/mlx-Bf16PrefillClose && nohup bash \
#     receipts/2026-09-10-bf16-prefill-close/m1_window_b.sh ... &'
# Env: CAND_COMMIT=<7+ char commit sha> (stamps the wheel), optional
# SKIP_BUILD=1 to reuse a built candidate wheel.
set -uo pipefail
cd ~/src/mlx-Bf16PrefillClose
R=receipts/2026-09-10-bf16-prefill-close
mkdir -p "$R/m1-logs" "$R/matrix" "$R/wheels/candidate"

base_wheel="$HOME/src/mlx-main-b6d662a8/dist/mlx_omarchy-0.32.2.dev202609101916+b6d662a-cp314-cp314-linux_aarch64.whl"
echo "98821134f4bcf306ab1d64d6885ddf3d3e8e932311283bbd11f97aecf6a8e439  $base_wheel" \
  | sha256sum -c - || { echo "FATAL: base wheel mismatch"; exit 1; }

exec 9>/tmp/m1-gpu.lock
echo "$(date -Is) waiting for /tmp/m1-gpu.lock"
flock 9
echo "$(date -Is) lock acquired"

{
  echo "== phase 0: driver capability gate =="
  g++ -std=c++17 -O2 -o /tmp/bf16-prefill-bench tools/bf16-prefill-bench/bench.cpp
  /tmp/bf16-prefill-bench --tiny 2>&1 | grep '\"k\":\"dev\"' \
    | tee "$R/m1-logs/driver-gate-b.txt"
  grep -q '\"coopmat\":true' "$R/m1-logs/driver-gate-b.txt" || {
    echo "FATAL: default driver lacks cooperative matrix; bailing."
    exit 3
  }

  if [[ "${SKIP_BUILD:-0}" != 1 ]]; then
    echo "== phase 1: candidate wheel build (niced, -j4) =="
    nice -n 10 env DEV_RELEASE=1 CMAKE_BUILD_PARALLEL_LEVEL=4 \
      MLX_OMARCHY_SOURCE_COMMIT="${CAND_COMMIT:-local}" \
      scripts/build-wheel.sh 2>&1 | tail -5 | tee "$R/m1-logs/build-wheel-cand.log"
    cand_wheel=$(ls -t dist/mlx_omarchy-*.whl | head -1)
    cp "$cand_wheel" "$R/wheels/candidate/"
    sha256sum "$cand_wheel" | tee "$R/m1-logs/candidate-wheel.sha256"
  else
    cand_wheel=$(ls -t "$R"/wheels/candidate/*.whl | head -1)
  fi

  echo "== phase 2: fresh venvs =="
  rm -rf .venv-prefill-base .venv-prefill-cand
  python3.14 -m venv .venv-prefill-base
  .venv-prefill-base/bin/pip install --quiet "$base_wheel" mlx-lm==0.31.3
  python3.14 -m venv .venv-prefill-cand
  .venv-prefill-cand/bin/pip install --quiet "$cand_wheel" mlx-lm==0.31.3

  echo "== phase 3: suite binaries (niced, -j4) =="
  cmake -S .work/mlx -B .work/build-prefill -G Ninja \
    -DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_CPU=OFF -DMLX_BUILD_METAL=OFF \
    -DMLX_BUILD_CUDA=OFF -DMLX_BUILD_TESTS=ON -DMLX_BUILD_EXAMPLES=OFF \
    -DMLX_BUILD_BENCHMARKS=OFF -DMLX_BUILD_PYTHON_BINDINGS=OFF \
    > "$R/m1-logs/cmake-suite.log" 2>&1
  nice -n 10 cmake --build .work/build-prefill -j4 --target \
    omarchy_matmul_family_tests omarchy_runtime_tests mlx-omarchy-info \
    2>&1 | tail -3 | tee "$R/m1-logs/build-suite.log"

  echo "== phase 4: cooldown before measurements =="
  sleep 180

  echo "== phase 5: family + runtime suites, fork driver (base and cand) =="
  for cell in base cand; do
    timeout 1800 ./.work/build-prefill/tests/omarchy/omarchy_matmul_family_tests \
      --out="$R/m1-logs/m1-fork-matmul-family-$cell.log"
    echo "matmul_family_$cell rc=$?"
    timeout 900 ./.work/build-prefill/tests/omarchy/omarchy_runtime_tests \
      --out="$R/m1-logs/m1-fork-runtime-$cell.log"
    echo "runtime_$cell rc=$?"
    tail -4 "$R/m1-logs/m1-fork-matmul-family-$cell.log"
  done
  # llvmpipe fallback screening runs on the local x86_64 box (this M1
  # carries no llvmpipe ICD); its tile-path legs are recorded there.
  run_matrix() {  # driver cell wheel label
    local driver=$1 cell=$2 wheel=$3 label=$4
    local run="$R/matrix/$label"
    mkdir -p "$run"
    if [[ $driver == stock ]]; then
      env VK_DRIVER_FILES=/home/joshuawarren/stock-mesa/stock-icd.json \
        MLX_DISABLE_COMPILE=1 timeout 1800 scripts/bench_matrix.py --mode run \
        --python ".venv-prefill-$cell/bin/python" --wheel "$wheel" \
        --host-label "jwm1-$label" --timeout 900 \
        --expect-pins qwen25-0.5b-4bit=a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3 \
        --expect-pins qwen25-0.5b-bf16=56d07e766edd7159fbe12ed12d9cf114bf38bf1e \
        --out "$run/matrix.json" > "$run/matrix.log" 2>&1
    else
      env MLX_DISABLE_COMPILE=1 timeout 1800 scripts/bench_matrix.py --mode run \
        --python ".venv-prefill-$cell/bin/python" --wheel "$wheel" \
        --host-label "jwm1-$label" --timeout 900 \
        --expect-pins qwen25-0.5b-4bit=a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3 \
        --expect-pins qwen25-0.5b-bf16=56d07e766edd7159fbe12ed12d9cf114bf38bf1e \
        --out "$run/matrix.json" > "$run/matrix.log" 2>&1
    fi
    local ok
    ok=$(grep -c 'verified=match' "$run/matrix.log" || true)
    echo "$label: verified=match x${ok}"
  }

  run_matrix fork base "$base_wheel" warmup   # discarded
  for rep in 1 2 3; do
    for driver in fork stock; do
      run_matrix "$driver" base "$base_wheel" "r${rep}-${driver}-base"
      run_matrix "$driver" cand "$cand_wheel" "r${rep}-${driver}-cand"
    done
  done
  echo MATRIX-OK
} > "$R/window-b.log" 2>&1
rc=$?
echo "$(date -Is) window complete rc=$rc"
exit $rc
