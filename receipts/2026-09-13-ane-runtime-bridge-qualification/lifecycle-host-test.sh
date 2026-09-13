#!/usr/bin/env bash
set -euo pipefail

RUNNER=/tmp/AneRuntimeBridge-57cc36a2-installed-combined.sh
PREFIX=$(mktemp)
ROOT_TMP=$(mktemp -d)
cleanup() {
  rm -f "$PREFIX"
  rm -rf "$ROOT_TMP"
}
trap cleanup EXIT
python3 - "$RUNNER" "$PREFIX" <<'PY'
from pathlib import Path
import sys
source, output = map(Path, sys.argv[1:])
text = source.read_text()
marker = '\ncapture_guardian\n'
if text.count(marker) != 1:
    raise SystemExit('main marker is not unique')
prefix = text.split(marker, 1)[0]
output.write_text(prefix + '\n')
PY

ACQUIRE_CUTOFF_EPOCH=1 OUTER_DEADLINE_EPOCH=1 PREFIX="$PREFIX" timeout --signal=TERM --kill-after=1s 5s setsid bash -c '
set -euo pipefail
source "$PREFIX"
PGID=$(ps -o pgid= -p $$ | tr -d " ")
GUARDIAN_PID=$$
sleep 2 &
child=$!
observed=$(process_group_members)
[[ "$observed" == *"pid=$child "* ]]
kill "$child"
wait "$child" 2>/dev/null || true
sleep 0.05
clean=$(process_group_members)
test -z "$clean"
printf "residual_child_detection=pass child=%s clean_case=pass pgid=%s\n" "$child" "$PGID"
'

run_finish_case() {
  local mode=$1 expected_rc=$2 expected_scan=$3 output rc
  local lease="$ROOT_TMP/$mode.lease" quarantine="$ROOT_TMP/$mode.quarantine"
  : >"$lease"
  : >"$quarantine"
  set +e
  output=$(ACQUIRE_CUTOFF_EPOCH=1 OUTER_DEADLINE_EPOCH=1 PREFIX="$PREFIX" MODE="$mode" LEASE_PATH="$lease" QUARANTINE_PATH="$quarantine" bash -c '
set -euo pipefail
source "$PREFIX"
sudo() {
  [[ ${1:-} == -n ]] && shift
  command "$@"
}
LEASE=$LEASE_PATH
QUARANTINE=$QUARANTINE_PATH
SESSION_DEADLINE_EPOCH=$(( $(date +%s) + 30 ))
LEASE_OWNED=1
PGID=999999
GUARDIAN_PID=999998
exec 9>/tmp/m1-gpu.lock
flock -n 9
if [[ "$MODE" == scanner-failure ]]; then
  process_group_members() { return 41; }
  workers() { return 42; }
else
  process_group_members() { return 0; }
  workers() { return 0; }
fi
finish
' 2>&1)
  rc=$?
  set -e
  test "$rc" -eq "$expected_rc"
  [[ "$output" == *"$expected_scan"* ]]
  if [[ "$mode" == scanner-failure ]]; then
    test -e "$lease"
    grep -F 'group_scan=failed' "$lease" >/dev/null
    grep -F 'worker_scan=failed' "$lease" >/dev/null
  else
    test ! -e "$lease"
  fi
  printf 'finish_case=%s pass rc=%s output=%s\n' "$mode" "$rc" "$output"
}

run_finish_case scanner-failure 98 'process_group_scan=failed'
run_finish_case clean 0 'clearance=true'
