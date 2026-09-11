#!/usr/bin/env bash
# Phase3b: mcpu wheel (env channel, verified) -> venvs -> win4b + win5b.
set -u
ROOT="$HOME/src/mlx-HostPathOverhead"
S="$ROOT/receipts/2026-09-11-pyenv-parity"
DIST="$ROOT/dist"
LOG="$S/phase3.log"
echo "[phase3b $(date -u +%FT%TZ)] start pid $$" >>"$LOG"

wait_gap() {
  while true; do
    if flock -n /tmp/m1-gpu.lock true 2>/dev/null; then
      la=$(cut -d' ' -f1 /proc/loadavg)
      awk -v v="$la" 'BEGIN{exit !(v<1.5)}' && return 0
    fi
    sleep 45
  done
}

if ! ls "$S"/wheels/mcpu-verified/*.whl >/dev/null 2>&1; then
  wait_gap
  CXXFLAGS="-mcpu=native" CFLAGS="-mcpu=native" \
    "$S/build_wheel.sh" /usr/bin/python3 cp314mcpu2 >>"$LOG" 2>&1
  W=$(ls "$DIST"/mlx_omarchy-*-cp314-*.whl | head -1)
  mkdir -p "$S/wheels/mcpu-verified"; cp "$W" "$S/wheels/mcpu-verified/"
fi
B=$(ls -d "$ROOT"/.work-cp314mcpu2-314/mlx/build/temp.*/mlx.core 2>/dev/null | head -1)
FL=$(grep -m1 "CXX_FLAGS" "$B/CMakeFiles/mlx.dir/flags.make" 2>/dev/null)
echo "[phase3b] mlx flags: $FL" >>"$LOG"
case "$FL" in *mcpu=native*) echo "[phase3b] mcpu verified" >>"$LOG";; *) echo "[phase3b] FATAL mcpu absent" >>"$LOG"; exit 2;; esac

WMCPU=$(ls "$S"/wheels/mcpu-verified/*.whl | head -1)
WLTO=$(ls "$S"/wheels/lto-verified/*.whl | head -1)
"$S/make_venv.sh" "$S/venv-lto2" /usr/bin/python3 "$WLTO" >>"$LOG" 2>&1 || { echo "[phase3b] FATAL venv-lto2" >>"$LOG"; exit 2; }
"$S/make_venv.sh" "$S/venv-mcpu2" /usr/bin/python3 "$WMCPU" >>"$LOG" 2>&1 || { echo "[phase3b] FATAL venv-mcpu2" >>"$LOG"; exit 2; }

W314="$S/wheels/mlx_omarchy-0.32.2.dev202609110320+12012beb-cp314-cp314-linux_aarch64.whl"
"$S/run_window.sh" win4b 5 "cp314hpo=$ROOT/.work/venv-hpo/bin/python:$W314:1" "cp314lto=$S/venv-lto2/bin/python:$WLTO:1"
echo "[phase3b] win4b rc=$?" >>"$LOG"
"$S/run_window.sh" win5b 5 "cp314hpo=$ROOT/.work/venv-hpo/bin/python:$W314:1" "cp314mcpu=$S/venv-mcpu2/bin/python:$WMCPU:1"
echo "[phase3b] win5b rc=$?" >>"$LOG"
echo "[phase3b $(date -u +%FT%TZ)] complete" >>"$LOG"
