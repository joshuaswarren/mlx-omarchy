#!/usr/bin/env bash
# BoolAllFix jwm1 verification: probe table + reduce suite + full battery.
# Phase argument: "unfixed" runs only probe + reduce suite (regression-test
# proof); "fixed" runs probe + reduce suite + full battery, two passes.
set -u
ROOT="$HOME/src/mlx-boolall"
cd "$ROOT" || exit 1
PHASE="${1:-fixed}"
PY="$HOME/src/mlx-boolall/.work/venv-boolall/bin/python"
if [ ! -x "$PY" ]; then
  PY="$(command -v python3.12 || command -v python3)"
fi
export PYTHONPATH="$ROOT/.work/pylink"
echo "[boolall] phase=$PHASE start $(date)"
echo "[boolall] device:"
"$PY" -c "import mlx.core as mx; print(' ', mx.device_info)"
echo "[boolall] probe table"
"$PY" boolall-2026-09-03/probe_boolall.py > "/tmp/boolall-table-$PHASE.txt" 2>&1
rc=$?
echo "[boolall] probe rc=$rc -> /tmp/boolall-table-$PHASE.txt"
tail -1 "/tmp/boolall-table-$PHASE.txt"

echo "[boolall] reduce suite"
start=$(date +%s)
if .work/build/tests/omarchy/omarchy_reduce_ops_tests > "/tmp/boolall-reduce-$PHASE.log" 2>&1; then
  echo "[boolall] reduce rc=0 secs=$(($(date +%s)-start))"
else
  echo "[boolall] reduce rc=FAIL secs=$(($(date +%s)-start))"
fi
grep -E "test cases|assertions" "/tmp/boolall-reduce-$PHASE.log" | tr '\n' ' '
echo
grep -B1 -A8 "FAILED" "/tmp/boolall-reduce-$PHASE.log" | head -40

if [ "$PHASE" = "unfixed" ]; then
  echo "[boolall] unfixed phase done $(date)"
  exit 0
fi

echo "[boolall] full battery $(date)"
BINARIES="$(ls .work/build/tests/omarchy/omarchy_*_tests | sort)"
echo "[boolall] binaries: $(echo "$BINARIES" | wc -l)"
for pass in 1 2; do
  FAIL=0
  for bin in $BINARIES; do
    name="$(basename "$bin")"
    start=$(date +%s)
    if .work/build/tests/omarchy/"$name" > "/tmp/boolall-b${pass}-${name}.log" 2>&1; then
      echo "pass${pass} ${name} rc=0 secs=$(($(date +%s)-start))"
    else
      echo "pass${pass} ${name} rc=FAIL secs=$(($(date +%s)-start))"
      FAIL=1
    fi
  done
  echo "[boolall] pass${pass} done FAIL=$FAIL $(date)"
done
echo "[boolall] counts (pass2)"
TOTAL_CASES=0
TOTAL_ASSERTS=0
for bin in $BINARIES; do
  name="$(basename "$bin")"
  line="$(grep -E "test cases|assertions" "/tmp/boolall-b2-${name}.log" | tr '\n' ' ')"
  echo "${name}: ${line}"
done
echo "[boolall] fixed phase done $(date)"
