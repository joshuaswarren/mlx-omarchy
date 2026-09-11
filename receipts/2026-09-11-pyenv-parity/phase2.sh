#!/usr/bin/env bash
# PythonHostParity phase 2: wheels -> venvs -> windows -> remaining builds.
# Builds reniced + gap-gated (lock free AND loadavg<1.5); windows flock FIFO.
set -u
ROOT="$HOME/src/mlx-HostPathOverhead"
S="$ROOT/receipts/2026-09-11-pyenv-parity"
DIST="$ROOT/dist"
LOG="$S/phase2.log"
ST="$S/phase2.state"
touch "$ST"
log() { echo "[phase2 $(date -u +%FT%TZ)] $*" >>"$LOG"; }
done_step() { grep -q "^done $1\$" "$ST"; }
mark() { echo "done $1" >>"$ST"; }

wait_gap() {
  while true; do
    if flock -n /tmp/m1-gpu.lock true 2>/dev/null; then
      la=$(cut -d' ' -f1 /proc/loadavg)
      awk -v v="$la" 'BEGIN{exit !(v<1.5)}' && return 0
    fi
    sleep 45
  done
}

HPO_PY="$ROOT/.work/venv-hpo/bin/python"
W314="$S/wheels/mlx_omarchy-0.32.2.dev202609110320+12012beb-cp314-cp314-linux_aarch64.whl"

log "phase2 start pid $$"

# 1. wait for cp312 wheel (build already running), install into both cp312 venvs
if ! done_step cp312-install; then
  W=""
  for i in $(seq 1 60); do
    W=$(ls "$DIST"/mlx_omarchy-*-cp312-*.whl 2>/dev/null | head -1)
    [ -n "$W" ] && break
    sleep 30
  done
  [ -n "$W" ] || { log "FATAL no cp312 wheel after 30min"; exit 2; }
  cp "$W" "$S/wheels/"
  sha256sum "$W" >> "$S/wheels/SHA256SUMS"
  for v in venv-cp312base venv-cp312pgo; do
    nice -n 19 "$S/$v/bin/pip" install -q --force-reinstall --no-deps "$W" >>"$LOG" 2>&1
    "$S/$v/bin/python" -c "import mlx.core, mlx_lm, sys; print('venv $v ok', sys.version.split()[0])" >>"$LOG" 2>&1 \
      || { log "FATAL $v import"; exit 2; }
  done
  mark cp312-install; log "cp312 wheel installed into cp312 venvs"
fi
W312=$(ls "$S"/wheels/mlx_omarchy-*-cp312-*.whl | head -1)

# 2. window 2: interpreter version + interpreter build levers, 5 reps
if ! done_step win2; then
  "$S/run_window.sh" win2 5 \
    "cp314hpo=$HPO_PY:$W314:1" \
    "cp312base=$S/venv-cp312base/bin/python:$W312:1" \
    "cp314pgo=$S/venv-cp314pgo/bin/python:$W314:1"
  rc=$?
  log "win2 rc=$rc"
  [ $rc -ne 0 ] && { log "FATAL win2"; exit 2; }
  mark win2
fi

# 3. remaining builds, gap-gated: py311-pgo, then wheels cp311 / lto / mcpu
if ! done_step py311-pgo; then
  wait_gap
  "$S/build_python.sh" 3.11.16-pgo >>"$LOG" 2>&1 && mark py311-pgo || { log "FATAL py311-pgo"; exit 2; }
fi
if ! done_step wheel-cp311; then
  wait_gap
  "$S/build_wheel.sh" "$HOME/opt/3.11.16-pgo/bin/python3.11" cp311 >>"$LOG" 2>&1 \
    && { W=$(ls "$DIST"/mlx_omarchy-*-cp311-*.whl | head -1); cp "$W" "$S/wheels/"; mark wheel-cp311; } \
    || { log "FATAL wheel-cp311"; exit 2; }
fi
W311=$(ls "$S"/wheels/mlx_omarchy-*-cp311-*.whl | head -1)
if ! done_step venv-cp311pgo; then
  "$S/make_venv.sh" "$S/venv-cp311pgo" "$HOME/opt/3.11.16-pgo/bin/python3.11" "$W311" >>"$LOG" 2>&1 \
    && mark venv-cp311pgo || { log "FATAL venv-cp311pgo"; exit 2; }
fi

# 4. window 3: native-parity interpreter (cp311 PGO+LTO)
if ! done_step win3; then
  "$S/run_window.sh" win3 5 \
    "cp314hpo=$HPO_PY:$W314:1" \
    "cp311pgo=$S/venv-cp311pgo/bin/python:$W311:1"
  rc=$?
  log "win3 rc=$rc"
  [ $rc -ne 0 ] && { log "FATAL win3"; exit 2; }
  mark win3
fi

# 5. wheel-flag variants on cp314
if ! done_step wheel-lto; then
  wait_gap
  "$S/build_wheel.sh" /usr/bin/python3 cp314lto "-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON" >>"$LOG" 2>&1 \
    && { W=$(ls "$DIST"/mlx_omarchy-*-cp314-*.whl | head -1); cp "$W" "$S/wheels/lto-cp314.whl"; mark wheel-lto; } \
    || { log "FATAL wheel-lto"; exit 2; }
fi
if ! done_step wheel-mcpu; then
  wait_gap
  "$S/build_wheel.sh" /usr/bin/python3 cp314mcpu "-DCMAKE_CXX_FLAGS=-mcpu=native -DCMAKE_C_FLAGS=-mcpu=native" >>"$LOG" 2>&1 \
    && { W=$(ls "$DIST"/mlx_omarchy-*-cp314-*.whl | head -1); cp "$W" "$S/wheels/mcpu-cp314.whl"; mark wheel-mcpu; } \
    || { log "FATAL wheel-mcpu"; exit 2; }
fi
if ! done_step venv-lto; then
  "$S/make_venv.sh" "$S/venv-cp314lto" /usr/bin/python3 "$S/wheels/lto-cp314.whl" >>"$LOG" 2>&1 && mark venv-lto
fi
if ! done_step venv-mcpu; then
  "$S/make_venv.sh" "$S/venv-cp314mcpu" /usr/bin/python3 "$S/wheels/mcpu-cp314.whl" >>"$LOG" 2>&1 && mark venv-mcpu
fi

# 6. windows 4 and 5: wheel-flag levers
if ! done_step win4; then
  "$S/run_window.sh" win4 5 \
    "cp314hpo=$HPO_PY:$W314:1" \
    "cp314lto=$S/venv-cp314lto/bin/python:$S/wheels/lto-cp314.whl:1"
  log "win4 rc=$?"
  mark win4
fi
if ! done_step win5; then
  "$S/run_window.sh" win5 5 \
    "cp314hpo=$HPO_PY:$W314:1" \
    "cp314mcpu=$S/venv-cp314mcpu/bin/python:$S/wheels/mcpu-cp314.whl:1"
  log "win5 rc=$?"
  mark win5
fi

# 7. cp312pgo window (venv already has wheel): version-vs-build on 3.12
if ! done_step win6; then
  "$S/run_window.sh" win6 5 \
    "cp314hpo=$HPO_PY:$W314:1" \
    "cp312pgo=$S/venv-cp312pgo/bin/python:$W312:1"
  log "win6 rc=$?"
  mark win6
fi

log "phase2 complete"
