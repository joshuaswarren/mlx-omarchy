#!/usr/bin/env bash
# Window A on jwm1-linux: BF16 prefill attribution + shader bench.
# ONE top-level flock; the bench binary is built BEFORE acquiring the
# lock. Everything inside the window is a GPU measurement or a
# seconds-long setup step. Run from the checkout root:
#   bash receipts/2026-09-10-bf16-prefill-close/m1_window_a.sh
set -uo pipefail
cd ~/src/mlx-Bf16PrefillClose
R=receipts/2026-09-10-bf16-prefill-close
mkdir -p "$R/m1-logs"

echo "$(date -Is) building bench binary (pre-lock)"
g++ -std=c++17 -O2 -o /tmp/bf16-prefill-bench tools/bf16-prefill-bench/bench.cpp \
  || { echo "bench build failed"; exit 1; }
sha256sum tools/bf16-prefill-bench/shaders/*.comp \
  | tee "$R/m1-logs/shader-hashes.txt"
# The base copy must equal the production shader for the measurement to
# be about the shipped kernel.
diff -q tools/bf16-prefill-bench/shaders/mm_bf16_base.comp \
    overlay/mlx/backend/omarchy/shaders/matmul_coopmat_bf16.comp || {
  echo "FATAL: mm_bf16_base.comp drifted from the production shader"
  exit 1
}

exec 9>/tmp/m1-gpu.lock
echo "$(date -Is) waiting for /tmp/m1-gpu.lock"
flock 9
echo "$(date -Is) lock acquired"

{
  echo "== phase 0: driver capability gate =="
  /tmp/bf16-prefill-bench --tiny 2>&1 | grep -E '\"k\":\"dev\"' \
    | tee "$R/m1-logs/driver-gate.txt"
  grep -q '\"coopmat\":true' "$R/m1-logs/driver-gate.txt" || {
    echo "FATAL: default driver does not expose cooperative matrix -"
    echo "the fork driver build is not the default. Bailing without"
    echo "measuring; rerun after the fork package is ensured."
    exit 3
  }

  echo "== phase 1: shader bench, fork driver (default) ="
  /tmp/bf16-prefill-bench 2>&1 | tee "$R/m1-logs/bench-fork.ndjson"

  echo "== phase 2: shader bench, stock driver =="
  env VK_DRIVER_FILES=/home/joshuawarren/stock-mesa/stock-icd.json \
    /tmp/bf16-prefill-bench --quick 2>&1 \
    | tee "$R/m1-logs/bench-stock.ndjson"

  echo "== phase 3: attribution probe, fork (coopmat) =="
  MLX_DISABLE_COMPILE=1 \
    ~/src/mlx-requal/.venv-requal-base/bin/python \
    "$R/attribution_probe.py" --out "$R/m1-logs/attrib-fork.ndjson" 2>&1 \
    | tail -30

  echo "== phase 4: attribution probe, tile fallback =="
  MLX_DISABLE_COMPILE=1 MLX_OMARCHY_NO_COOPMAT=1 \
    ~/src/mlx-requal/.venv-requal-base/bin/python \
    "$R/attribution_probe.py" --out "$R/m1-logs/attrib-tile.ndjson" 2>&1 \
    | tail -30

  echo "== phase 5: power + provenance line =="
  /usr/bin/pmset -g batt 2>/dev/null || acpi 2>/dev/null || true
  ~/src/mlx-requal/.venv-requal-base/bin/python -c "
import mlx.core as mx
print('driver', mx.device_info())" 2>&1 | tail -2
} > "$R/window-a.log" 2>&1
rc=$?
echo "$(date -Is) window complete rc=$rc"
exit $rc
