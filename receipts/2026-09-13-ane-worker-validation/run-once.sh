#!/usr/bin/env bash
# Bounded ANE worker hardware validation on jwm1 (Phase 4, sections 22-27).
#
# Contract mirrors receipts/2026-09-13-parakeet-hardware-window:
# preflight (device/module/power/lock/workers/quarantine), one bounded
# run of the 64-element add-mul fixture through the new in-tree worker,
# post-verify, receipt. Prohibited: unloading ane, rebooting, 1x896.
#
# Usage (on jwm1): run-once.sh SOURCE_ROOT
set -euo pipefail

SOURCE_ROOT="${1:?usage: run-once.sh SOURCE_ROOT}"
OUT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$OUT_DIR/hardware-window.log"
LOCK=/tmp/m1-gpu.lock
LIBANE_SRC=~/src/omarchy-ane-h13-f261a6c/libane
FIXTURE="$SOURCE_ROOT/overlay/tests/omarchy/ane/fixtures/h13-explicit-chain-add-mul"
DEADLINE_MS=5000
ITERATIONS=2

log() { printf '%s\n' "$*" | tee -a "$LOG"; }

# --- preflight ---------------------------------------------------------
preflight_device=$(stat -c '%F' /dev/accel/accel0)
preflight_module=missing
[[ -d /sys/module/ane ]] && preflight_module=loaded
preflight_power=$(cat /sys/devices/genpd_provider/ane_sys/power/runtime_status \
  2>/dev/null || echo unknown)
preflight_gpu_lock=held
flock -n "$LOCK" -c true 2>/dev/null && preflight_gpu_lock=free
preflight_quarantine=$(cat /sys/kernel/ane/quarantine 2>/dev/null || echo 0)
log "host=$(hostname) started_at=$(date -Is)"
log "preflight_device=$preflight_device preflight_module=$preflight_module"
log "preflight_power=$preflight_power preflight_gpu_lock=$preflight_gpu_lock"
log "preflight_quarantine_bytes=$preflight_quarantine"
[[ "$preflight_device" == *"character special"* ]]
[[ "$preflight_module" == loaded ]]

# --- build (inside the window; keeps the tree pinned to this run) ------
exec 9>"$LOCK"
flock -n 9 || { log "gpu lock busy; refusing"; exit 1; }

BUILD="$SOURCE_ROOT/.work/mlx"
(cd "$SOURCE_ROOT" && scripts/prepare-mlx.sh >/dev/null)
cmake -S "$BUILD" -B "$BUILD/build-ane-device" -G Ninja \
  -DMLX_BUILD_TESTS=ON -DMLX_BUILD_OMARCHY=ON \
  -DMLX_OMARCHY_ANE_DEVICE=ON \
  -DOMARCHY_ANE_INCLUDE_DIR="$LIBANE_SRC" \
  -DCMAKE_BUILD_TYPE=Release >/dev/null
ninja -C "$BUILD/build-ane-device" mlx-omarchy-ane-worker \
  omarchy_ane_worker_tests >/dev/null
"$BUILD/build-ane-device/tests/omarchy/omarchy_ane_worker_tests" \
  | tee -a "$LOG"

gcc -shared -fPIC -std=gnu99 -O3 -o "$OUT_DIR/libane.so" \
  -I"$LIBANE_SRC" -I/usr/include/libdrm \
  -I"$HOME/src/omarchy-ane-h13-f261a6c/ane/src/uapi/drm" \
  "$LIBANE_SRC/ane.c" -ldrm

# --- adapt the compiler package into a strict schema-4 bundle ---------
BUNDLE_OUT="$OUT_DIR/bundle"
rm -f "$BUNDLE_OUT"/manifest.json "$BUNDLE_OUT"/program-*.anec
rmdir "$BUNDLE_OUT" 2>/dev/null || true
python3 "$SOURCE_ROOT/overlay/tools/ane-export/h13_package_to_bundle.py" \
  "$FIXTURE" \
  --out-dir "$BUNDLE_OUT" \
  --graph-source "$FIXTURE/model.mil" \
  --compiler-source "$HOME/src/mil-hwxc-h13-ea903c4" \
  --name schema4-add-mul-worker \
  --source-repo mil-hwxc \
  --source-commit aa688df66cbc2110e0df94f0d50fb72c7fa30a18 \
  --model model.mil 2>&1 | tee -a "$LOG"

# --- deterministic fixture inputs and expected output ------------------
python3 - "$OUT_DIR" <<'PY'
import numpy as np
import sys
out = sys.argv[1]
i = np.arange(64, dtype=np.float32)
a = (i * 0.25 - 8.0).astype(np.float16)
b = ((i % 7) * 1.5 - 4.5).astype(np.float16)
y = ((a + b) * b).astype(np.float16)  # fp16 arithmetic matches the device
a.tofile(f"{out}/a.bin")
b.tofile(f"{out}/b.bin")
y.tofile(f"{out}/y.bin")
PY

# --- bounded hardware run ----------------------------------------------
TOOL="$BUILD/build-ane-device/tools/mlx-omarchy-ane-worker/mlx-omarchy-ane-worker"
set +e
"$TOOL" --bundle "$BUNDLE_OUT" --libane "$OUT_DIR/libane.so" \
  --deadline-ms "$DEADLINE_MS" --iterations "$ITERATIONS" \
  --input a="$OUT_DIR/a.bin" --input b="$OUT_DIR/b.bin" \
  --expect y="$OUT_DIR/y.bin" 2>&1 | tee -a "$LOG"
hardware_exit=${PIPESTATUS[0]}
set -e
log "hardware_command_exit=$hardware_exit"

# --- post-verify --------------------------------------------------------
post_module=missing
[[ -d /sys/module/ane ]] && post_module=loaded
post_power=$(cat /sys/devices/genpd_provider/ane_sys/power/control \
  2>/dev/null || echo unknown)
post_runtime=$(cat /sys/devices/genpd_provider/ane_sys/power/runtime_status \
  2>/dev/null || echo unknown)
post_quarantine=$(cat /sys/kernel/ane/quarantine 2>/dev/null || echo 0)
log "post_device=$(stat -c '%F' /dev/accel/accel0)"
log "post_module=$post_module post_power_control=$post_power"
log "post_runtime_status=$post_runtime post_quarantine_bytes=$post_quarantine"
log "finished_at=$(date -Is)"

[[ "$hardware_exit" -eq 0 ]]
[[ "$post_module" == loaded ]]
[[ "$post_quarantine" == 0 ]]
log "result=PASS"
