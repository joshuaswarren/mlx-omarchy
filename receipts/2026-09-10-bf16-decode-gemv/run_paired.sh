#!/usr/bin/env bash
set -euo pipefail

driver=$1
out=$2
case "$driver" in
  fork) unset VK_DRIVER_FILES ;;
  stock) export VK_DRIVER_FILES=/home/joshuawarren/stock-mesa/stock-icd.json ;;
  *) echo "driver must be fork or stock" >&2; exit 2 ;;
esac

mkdir -p "$out"
for pair in 1 2 3; do
  for variant in baseline candidate; do
    if [[ $variant == baseline ]]; then
      python=.venv-base/bin/python
      wheel=receipt-bf16/wheels/baseline/mlx_omarchy-0.32.2.dev202609101409+diag.3b33709-cp314-cp314-linux_aarch64.whl
    else
      python=.venv-candidate/bin/python
      wheel=receipt-bf16/wheels/candidate/mlx_omarchy-0.32.2.dev202609101453+diag.6f1612c-cp314-cp314-linux_aarch64.whl
    fi
    run="$out/pair-${pair}-${variant}"
    mkdir -p "$run"
    MLX_DISABLE_COMPILE=1 timeout 1800 scripts/bench_matrix.py \
      --mode run \
      --python "$python" \
      --wheel "$wheel" \
      --host-label "jwm1-${driver}-${variant}" \
      --timeout 900 \
      --expect-pins qwen25-0.5b-4bit=a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3 \
      --expect-pins qwen25-0.5b-bf16=56d07e766edd7159fbe12ed12d9cf114bf38bf1e \
      --out "$run/matrix.json" >"$run/matrix.log" 2>&1
  done
done
