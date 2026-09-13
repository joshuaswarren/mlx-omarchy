#!/usr/bin/env bash
set -uo pipefail

PAYLOAD=${PAYLOAD_DIR:-$HOME/LongContextCostAttribution-gpu-profile-payload}
DEADLINE_EPOCH=${DEADLINE_EPOCH:?set the parent-granted total deadline epoch}
TOTAL="$HOME/longctx-gpu-profile-total-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$TOTAL"
exec > >(tee "$TOTAL/master.log") 2>&1
exec 9>/tmp/m1-gpu.lock
if ! flock -n 9; then
  echo "total_lock_acquired=false"
  exit 75
fi
printf 'total_lock_acquired=true utc=%s pid=%s deadline_epoch=%s result_dir=%s\n' \
  "$(date -u +%FT%TZ)" "$$" "$DEADLINE_EPOCH" "$TOTAL"
printf 'lease=Main-grant-LongContextCostAttribution phase=diagnostic-build-and-profile pid=%s acquired_utc=%s deadline_epoch=%s\n' \
  "$$" "$(date -u +%FT%TZ)" "$DEADLINE_EPOCH" > "$TOTAL/lease.txt"

cleanup() {
  rc=$?
  trap - EXIT INT TERM
  remaining=$(jobs -pr)
  if [ -n "$remaining" ]; then
    kill $remaining 2>/dev/null || true
    wait $remaining 2>/dev/null || true
  fi
  if [ -n "$(jobs -pr)" ]; then
    echo "TOTAL_CHILD_PIDS_NOT_CLEAR"
    rc=70
  else
    echo "TOTAL_CHILD_PIDS_CLEAR"
  fi
  printf 'released_utc=%s rc=%s\n' "$(date -u +%FT%TZ)" "$rc" >> "$TOTAL/lease.txt"
  flock -u 9
  echo "total_lock_released=true utc=$(date -u +%FT%TZ) rc=$rc result_dir=$TOTAL"
  exit "$rc"
}
trap cleanup EXIT INT TERM

[ -x "$PAYLOAD/build_diag.sh" ] || { echo "FATAL missing build payload"; exit 2; }
[ -x "$PAYLOAD/window.sh" ] || { echo "FATAL missing measurement payload"; exit 2; }

echo "build_phase_start utc=$(date -u +%FT%TZ)"
DEADLINE_EPOCH="$DEADLINE_EPOCH" \
  BUILD_LOG="$TOTAL/build.log" "$PAYLOAD/build_diag.sh"
rc=$?
echo "build_phase_end utc=$(date -u +%FT%TZ) rc=$rc"
[ "$rc" -eq 0 ] || exit "$rc"
remaining=$((DEADLINE_EPOCH - $(date +%s)))
[ "$remaining" -gt 0 ] || { echo "FATAL total deadline consumed by build; measurement not started"; exit 124; }

echo "measurement_phase_start utc=$(date -u +%FT%TZ) remaining_seconds=$remaining"
LOCK_ALREADY_HELD=1 PAYLOAD_DIR="$PAYLOAD" DEADLINE_EPOCH="$DEADLINE_EPOCH" \
  "$PAYLOAD/window.sh"
rc=$?
echo "measurement_phase_end utc=$(date -u +%FT%TZ) rc=$rc"
[ "$rc" -eq 0 ] || exit "$rc"
echo "TOTAL_WINDOW_COMPLETE result_dir=$TOTAL"
