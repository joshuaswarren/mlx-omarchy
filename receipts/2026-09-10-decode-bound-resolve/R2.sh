#!/bin/bash
# Window R2 (DecodeBoundResolve): stock driver -> part2; restore fork after
set -euo pipefail
cd ~/src/mlx-DecodeBoundResolve
echo "== installing stock driver =="
sudo pacman -Rdd --noconfirm mesa-honeykrisp-omarchy
sudo pacman -U --noconfirm /var/cache/pacman/pkg/mesa-26.1.7-1-aarch64.pkg.tar.xz
vulkaninfo --summary 2>/dev/null | grep -m1 driverInfo || true
W=~/src/mlx-main-b6d662a8/dist/mlx_omarchy-0.32.2.dev202609101916+b6d662a-cp314-cp314-linux_aarch64.whl
M=~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3
echo "== part 2 (stock driver): compiled vs eager =="
python3 compile_ab.py --bench-dir ~/src/mlx-main-b6d662a8 \
  --python ~/src/mlx-DecodeBoundResolve/venv-run/bin/python \
  --wheel "$W" --model "$M" --out-dir results/compile-stock
echo "== restoring fork driver =="
sudo pacman -Rdd --noconfirm mesa
sudo pacman -U --noconfirm ~/src/mesa-pkg-20260908/mesa-honeykrisp-omarchy-26.3.0.devel.hk6f6afc8-1-aarch64.pkg.tar.xz
vulkaninfo --summary 2>/dev/null | grep -m1 driverInfo || true
echo R2-DONE
