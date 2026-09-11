#!/usr/bin/env bash
# PythonHostParity build pipeline (jwm1). Builds ONLY in gaps: waits until
# /tmp/m1-gpu.lock is free AND 1-min loadavg < 1.5, then runs the next step.
# All builds reniced (script does renice 19 internally), -j6 CPU only.
set -u
ROOT="$HOME/src/mlx-HostPathOverhead"
S="$ROOT/receipts/2026-09-11-pyenv-parity"
LOG="$S/pipeline.log"
ST="$S/pipeline.state"
touch "$ST"
log() { echo "[pipeline $(date -u +%FT%TZ)] $*" >>"$LOG"; }

wait_gap() {
  while true; do
    if flock -n /tmp/m1-gpu.lock true 2>/dev/null; then
      la=$(cut -d' ' -f1 /proc/loadavg)
      if awk -v v="$la" 'BEGIN{exit !(v<1.5)}'; then return 0; fi
    fi
    sleep 45
  done
}

step_done() { grep -q "^done $1$" "$ST"; }
run_step() {
  local id="$1"; shift
  if step_done "$id"; then log "skip $id (done)"; return 0; fi
  wait_gap
  log "start $id"
  if "$@" >>"$LOG" 2>&1; then
    echo "done $id" >>"$ST"; log "OK $id"
  else
    log "FAIL $id"; return 1
  fi
}

log "pipeline start pid $$"
run_step py312-base "$S/build_python.sh" 3.12.14-base
run_step py314-pgo  "$S/build_python.sh" 3.14.7-pgo
run_step py312-pgo  "$S/build_python.sh" 3.12.14-pgo
run_step py311-pgo  "$S/build_python.sh" 3.11.16-pgo
run_step wheel-cp312 "$S/build_wheel.sh" "$HOME/opt/3.12.14-base/bin/python3.12" cp312
run_step wheel-cp311 "$S/build_wheel.sh" "$HOME/opt/3.11.16-pgo/bin/python3.11" cp311
run_step wheel-cp314-lto "$S/build_wheel.sh" /usr/bin/python3 cp314lto \
  "-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON"
run_step wheel-cp314-mcpu "$S/build_wheel.sh" /usr/bin/python3 cp314mcpu \
  "-DCMAKE_CXX_FLAGS=-mcpu=native -DCMAKE_C_FLAGS=-mcpu=native"
log "pipeline complete"
