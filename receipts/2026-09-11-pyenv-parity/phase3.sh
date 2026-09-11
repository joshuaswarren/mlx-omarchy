#!/usr/bin/env bash
# PythonHostParity phase 3: rebuild flag wheels via env-flag channel, verify
# flags reached the build, install, then run win4b (LTO) and win5b (mcpu).
set -u
ROOT="$HOME/src/mlx-HostPathOverhead"
S="$ROOT/receipts/2026-09-11-pyenv-parity"
DIST="$ROOT/dist"
LOG="$S/phase3.log"
echo "[phase3 $(date -u +%FT%TZ)] start pid $$" >>"$LOG"

wait_gap() {
  while true; do
    if flock -n /tmp/m1-gpu.lock true 2>/dev/null; then
      la=$(cut -d' ' -f1 /proc/loadavg)
      awk -v v="$la" 'BEGIN{exit !(v<1.5)}' && return 0
    fi
    sleep 45
  done
}

# --- LTO wheel: whole-mlx-lib LTO via CXXFLAGS/LDFLAGS
if ! ls "$S"/wheels/lto-verified/*.whl >/dev/null 2>&1; then
  wait_gap
  CXXFLAGS="-flto" CFLAGS="-flto" LDFLAGS="-flto" \
    "$S/build_wheel.sh" /usr/bin/python3 cp314lto2 >>"$LOG" 2>&1
  W=$(ls "$DIST"/mlx_omarchy-*-cp314-*.whl | head -1)
  mkdir -p "$S/wheels/lto-verified"; cp "$W" "$S/wheels/lto-verified/"
  FL=$(grep -rl "flto" "$ROOT"/.work-cp314lto2-314/mlx/build/temp.*/mlx.core/mlx/CMakeFiles/mlx.dir/flags.make 2>/dev/null | head -1)
  if [ -n "$FL" ]; then echo "[phase3] LTO verified in mlx target flags" >>"$LOG"; else echo "[phase3] FATAL LTO flag absent" >>"$LOG"; exit 2; fi
fi
WLTO=$(ls "$S"/wheels/lto-verified/*.whl | head -1)

# --- mcpu wheel
if ! ls "$S"/wheels/mcpu-verified/*.whl >/dev/null 2>&1; then
  wait_gap
  CXXFLAGS="-mcpu=native" CFLAGS="-mcpu=native" \
    "$S/build_wheel.sh" /usr/bin/python3 cp314mcpu2 >>"$LOG" 2>&1
  W=$(ls "$DIST"/mlx_omarchy-*-cp314-*.whl | head -1)
  mkdir -p "$S/wheels/mcpu-verified"; cp "$W" "$S/wheels/mcpu-verified/"
  FL=$(grep -rl "mcpu=native" "$ROOT"/.work-cp314mcpu2-314/mlx/build/temp.*/mlx.core/mlx/CMakeFiles/mlx.dir/flags.make 2>/dev/null | head -1)
  if [ -n "$FL" ]; then echo "[phase3] mcpu verified in mlx target flags" >>"$LOG"; else echo "[phase3] FATAL mcpu flag absent" >>"$LOG"; exit 2; fi
fi
WMCPU=$(ls "$S"/wheels/mcpu-verified/*.whl | head -1)

# --- install into fresh venvs
"$S/make_venv.sh" "$S/venv-lto2" /usr/bin/python3 "$WLTO" >>"$LOG" 2>&1 || { echo "[phase3] FATAL venv-lto2" >>"$LOG"; exit 2; }
"$S/make_venv.sh" "$S/venv-mcpu2" /usr/bin/python3 "$WMCPU" >>"$LOG" 2>&1 || { echo "[phase3] FATAL venv-mcpu2" >>"$LOG"; exit 2; }

# --- windows (flock FIFO inside run_window)
"$S/run_window.sh" win4b 5 "cp314hpo=$ROOT/.work/venv-hpo/bin/python:$ROOT/receipts/2026-09-11-pyenv-parity/wheels/mlx_omarchy-0.32.2.dev202609110320+12012beb-cp314-cp314-linux_aarch64.whl:1" "cp314lto=$S/venv-lto2/bin/python:$WLTO:1"
echo "[phase3] win4b rc=$?" >>"$LOG"
"$S/run_window.sh" win5b 5 "cp314hpo=$ROOT/.work/venv-hpo/bin/python:$ROOT/receipts/2026-09-11-pyenv-parity/wheels/mlx_omarchy-0.32.2.dev202609110320+12012beb-cp314-cp314-linux_aarch64.whl:1" "cp314mcpu=$S/venv-mcpu2/bin/python:$WMCPU:1"
echo "[phase3] win5b rc=$?" >>"$LOG"
echo "[phase3 $(date -u +%FT%TZ)] complete" >>"$LOG"
