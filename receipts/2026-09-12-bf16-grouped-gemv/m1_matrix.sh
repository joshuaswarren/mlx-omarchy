#!/usr/bin/env bash
set -euo pipefail

base=/tmp/mlx-bf16-grouped-base-b41e2b74
candidate=/tmp/mlx-bf16-grouped-candidate-6b1ac029
out=/tmp/bf16-grouped-gemv-cpu-20260912.matrix
stock_icd=/tmp/stock-mesa/stock-icd.json
lease=/tmp/bf16-grouped-gemv-cpu-20260912.lease
holder=/tmp/bf16-grouped-gemv-cpu-20260912.pid

[[ -e "$lease" && -e "$holder" ]]
kill -0 "$(cat "$holder")"
mkdir -p "$out"

run_one() {
  local cell=$1 driver=$2 phase=$3
  local root python wheel run
  if [[ $cell == base ]]; then
    root=$base
  else
    root=$candidate
  fi
  python=$root/.work/venv-run/bin/python
  wheel=$(printf '%s\n' "$root"/dist/mlx_omarchy-*.whl)
  run=$out/$phase-$driver-$cell
  mkdir -p "$run"
  local -a driver_env=()
  if [[ $driver == stock ]]; then
    driver_env=(VK_DRIVER_FILES=$stock_icd)
  fi
  (
    cd "$root"
    timeout 1200 env "${driver_env[@]}" \
      MLX_DISABLE_COMPILE=1 MLX_OMARCHY_FUSED_GEMV=1 \
      "$python" scripts/bench_matrix.py --mode run \
      --python "$python" --wheel "$wheel" \
      --host-label "M1-$phase-$driver-$cell" --timeout 900 \
      --expect-pins qwen25-0.5b-4bit=a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3 \
      --expect-pins qwen25-0.5b-bf16=56d07e766edd7159fbe12ed12d9cf114bf38bf1e \
      --out "$run/matrix.json" >"$run/matrix.log" 2>&1
  )
  echo "$phase-$driver-$cell complete"
}

for driver in fork stock; do
  for cell in base candidate; do
    run_one "$cell" "$driver" warmup
  done
done
for rep in 1 2 3; do
  for driver in fork stock; do
    run_one base "$driver" "r$rep"
    run_one candidate "$driver" "r$rep"
  done
done

echo M1_GROUPED_MATRIX_OK
