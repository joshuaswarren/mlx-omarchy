#!/bin/bash
# Q4 native-map GEMV M1 window (receipts/2026-09-10-q4-gemv-native-map).
# ONE top-level flock for the whole window; quiet-gated; never merges.
#
# Stages inside the window:
#   A. build the release wheel from the rsynced tree (niced, CPU-heavy)
#   B. fresh venv: install wheel + pinned mlx-lm 0.31.3
#   C. q4-bw-bench (subgroup flavor): bit-eq + isolated timing + --occ
#      occupancy sweep, base vs nativemap vs nm32
#   D. digest legs on the fork driver (6 canonical pins, 3 reps)
#   E. driver swap to stock mesa, digest legs again, fork restored
set -euo pipefail
exec 9>/tmp/m1-gpu.lock
flock -w 14400 9
echo "lock held $(date -u +%H:%M:%SZ)"

TREE=~/src/mlx-Q4GemvNativeMap
VENV=~/src/mlx-Q4GemvNativeMap/venv-run
Q4=~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3
BF=~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-bf16/snapshots/56d07e766edd7159fbe12ed12d9cf114bf38bf1e
cd "$TREE"
mkdir -p receipts/2026-09-10-q4-gemv-native-map/results
R=receipts/2026-09-10-q4-gemv-native-map/results
echo "quiet gate..."
until [ "$(python3 -c 'import os;print(int(os.getloadavg()[0]*100))')" -lt 100 ]; do sleep 5; done

# A. wheel (niced: this window holds the lock, nobody else measures)
nice -n 5 env DEV_RELEASE=1 ./scripts/build-wheel.sh 2>&1 | tail -3
W=$(ls -t dist/mlx_omarchy-*-cp314-cp314-linux_aarch64.whl | head -1)
echo "wheel=$W"
sha256sum "$W" | tee "$R/wheel.sha256"

# B. fresh venv
python3 -m venv "$VENV"
nice -n 5 "$VENV/bin/pip" install --no-input "$W" mlx-lm==0.31.3 2>&1 | tail -2

# C. kernel bench: eq + timing + occupancy sweep, two legs
g++ -std=c++17 -O2 -o /tmp/q4-bw-bench tools/q4-bw-bench/bench.cpp
sha256sum tools/q4-bw-bench/shaders/qmm_vec_cand_nativemap.comp \
    overlay/mlx/backend/omarchy/shaders/qmm_vec.comp
diff -q tools/q4-bw-bench/shaders/qmm_vec_cand_nativemap.comp \
    overlay/mlx/backend/omarchy/shaders/qmm_vec.comp \
  || { echo "FATAL cand drifted"; exit 1; }
/tmp/q4-bw-bench --occ 2>&1 | tee "$R/bench-a.ndjson"
/tmp/q4-bw-bench --occ --quick 2>&1 | tee "$R/bench-b.ndjson"

# D. digest legs, fork driver
python3 receipts/2026-09-10-q4-gemv-native-map/q4nm_driver.py \
  --bench-dir "$TREE" \
  --python "$VENV/bin/python" --wheel "$W" \
  --model-dirs "qwen25-0.5b-4bit=$Q4,qwen25-0.5b-bf16=$BF" \
  --out-dir "$R/legs-fork"

# E. stock driver legs + fork restore
echo "== installing stock driver =="
sudo pacman -Rdd --noconfirm mesa-honeykrisp-omarchy
sudo pacman -U --noconfirm /var/cache/pacman/pkg/mesa-26.1.7-1-aarch64.pkg.tar.xz
vulkaninfo --summary 2>/dev/null | grep -m1 driverInfo || true
python3 receipts/2026-09-10-q4-gemv-native-map/q4nm_driver.py \
  --bench-dir "$TREE" \
  --python "$VENV/bin/python" --wheel "$W" \
  --model-dirs "qwen25-0.5b-4bit=$Q4,qwen25-0.5b-bf16=$BF" \
  --out-dir "$R/legs-stock"
echo "== restoring fork driver =="
sudo pacman -Rdd --noconfirm mesa
sudo pacman -U --noconfirm ~/src/mesa-pkg-20260908/mesa-honeykrisp-omarchy-26.3.0.devel.hk6f6afc8-1-aarch64.pkg.tar.xz
vulkaninfo --summary 2>/dev/null | grep -m1 driverInfo || true

echo WINDOW-DONE
