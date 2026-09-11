#!/bin/bash
# Window R1 (DecodeBoundResolve): fork driver -> part1 discriminator + part2 compiled-vs-eager
set -euo pipefail
cd ~/src/mlx-DecodeBoundResolve
if ! pacman -Q mesa-honeykrisp-omarchy >/dev/null 2>&1; then
  echo "== installing fork driver =="
  sudo pacman -U --noconfirm ~/src/mesa-pkg-20260908/mesa-honeykrisp-omarchy-26.3.0.devel.hk6f6afc8-1-aarch64.pkg.tar.xz
fi
vulkaninfo --summary 2>/dev/null | grep -m1 driverInfo || true
W=~/src/mlx-main-b6d662a8/dist/mlx_omarchy-0.32.2.dev202609101916+b6d662a-cp314-cp314-linux_aarch64.whl
M=~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3
echo "== part 1: decode attribution discriminator =="
python3 attr_driver.py --worker attr_worker.py \
  --manifest ~/src/mlx-main-b6d662a8/scripts/bench_matrix.json \
  --model "$M" \
  --venv-run ~/src/mlx-DecodeBoundResolve/venv-run \
  --venv-ablate ~/src/mlx-DecodeAttribution/.work/venv-ablate \
  --out-dir results/attr
echo "== part 2 (fork driver): compiled vs eager =="
python3 compile_ab.py --bench-dir ~/src/mlx-main-b6d662a8 \
  --python ~/src/mlx-DecodeBoundResolve/venv-run/bin/python \
  --wheel "$W" --model "$M" --out-dir results/compile-fork
echo R1-DONE
