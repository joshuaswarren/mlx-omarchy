#!/bin/bash
# Q4 native-map timing continuation (receipts/2026-09-10-q4-gemv-native-map).
# Stages D/E of window.sh with the corrected per-driver pins (commit
# 328218c), reusing the staged wheel + venv-run from the 2026-09-11 run
# (provenance verified=match, stamp dev202609110603+9a66501).
# ONE top-level flock; never merges; restores the fork driver.
set -euo pipefail
exec 9>/tmp/m1-gpu.lock
flock -w 7200 9
echo "lock held $(date -u +%H:%M:%SZ)"
TREE=~/src/mlx-Q4GemvNativeMap
R=$TREE/receipts/2026-09-10-q4-gemv-native-map/results
Q4=~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3
BF=~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-bf16/snapshots/56d07e766edd7159fbe12ed12d9cf114bf38bf1e
cd "$TREE"
until [ "$(python3 -c "import os;print(int(os.getloadavg()[0]*100))")" -lt 100 ]; do sleep 5; done
W=$(ls -t dist/mlx_omarchy-*-cp314-cp314-linux_aarch64.whl | head -1)
sha256sum "$W"
python3 receipts/2026-09-10-q4-gemv-native-map/q4nm_driver.py \
  --bench-dir "$TREE" --python "$TREE/venv-run/bin/python" --wheel "$W" \
  --model-dirs "qwen25-0.5b-4bit=$Q4,qwen25-0.5b-bf16=$BF" \
  --out-dir "$R/legs2-fork" --pins fork
echo "== installing stock driver =="
sudo pacman -Rdd --noconfirm mesa-honeykrisp-omarchy
sudo pacman -U --noconfirm /var/cache/pacman/pkg/mesa-26.1.7-1-aarch64.pkg.tar.xz
vulkaninfo --summary 2>/dev/null | grep -m1 driverInfo || true
python3 receipts/2026-09-10-q4-gemv-native-map/q4nm_driver.py \
  --bench-dir "$TREE" --python "$TREE/venv-run/bin/python" --wheel "$W" \
  --model-dirs "qwen25-0.5b-4bit=$Q4,qwen25-0.5b-bf16=$BF" \
  --out-dir "$R/legs2-stock" --pins stock
echo "== restoring fork driver =="
sudo pacman -Rdd --noconfirm mesa
sudo pacman -U --noconfirm ~/src/mesa-pkg-20260908/mesa-honeykrisp-omarchy-26.3.0.devel.hk6f6afc8-1-aarch64.pkg.tar.xz
vulkaninfo --summary 2>/dev/null | grep -m1 driverInfo || true
echo CONTINUATION-DONE
