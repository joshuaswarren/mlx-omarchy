#!/usr/bin/env bash
# Digest/perf matrix: {fork,stock} x {base,cand} x 3 reps + 1 discarded warmup.
# MUST run while holding /tmp/m1-gpu.lock (outer flock in m1_window.sh).
set -uo pipefail
cd ~/src/mlx-requal
out=receipt-requal/matrix
mkdir -p "$out"
base_wheel="$HOME/src/mlx-main-b6d662a8/dist/mlx_omarchy-0.32.2.dev202609101916+b6d662a-cp314-cp314-linux_aarch64.whl"
cand_wheel=$(ls receipt-requal/wheels/candidate/*.whl)

run_matrix() {  # driver cell wheel label
  local driver=$1 cell=$2 wheel=$3 label=$4
  local run="$out/$label"
  mkdir -p "$run"
  if [[ $driver == stock ]]; then
    env VK_DRIVER_FILES=/home/joshuawarren/stock-mesa/stock-icd.json MLX_DISABLE_COMPILE=1 \
      timeout 1800 scripts/bench_matrix.py --mode run \
      --python ".venv-requal-$cell/bin/python" --wheel "$wheel" \
      --host-label "jwm1-$label" --timeout 900 \
      --expect-pins qwen25-0.5b-4bit=a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3 \
      --expect-pins qwen25-0.5b-bf16=56d07e766edd7159fbe12ed12d9cf114bf38bf1e \
      --out "$run/matrix.json" > "$run/matrix.log" 2>&1
  else
    env MLX_DISABLE_COMPILE=1 \
      timeout 1800 scripts/bench_matrix.py --mode run \
      --python ".venv-requal-$cell/bin/python" --wheel "$wheel" \
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
