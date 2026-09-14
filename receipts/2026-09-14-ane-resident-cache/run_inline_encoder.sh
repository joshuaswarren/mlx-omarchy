#!/bin/bash
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
#
# One GPU-lock window: launch-per-submit baseline, then resident+inline.
# flock -w, never stolen. No retries.
set -u

BASE=/var/tmp/AneResidentCache
WORKER=$BASE/mlx-omarchy-ane-worker-inline
LIBANE=/var/tmp/AneWorkerValidation-c05ba1df/receipts/2026-09-13-ane-worker-validation/libane.so
SRC=/var/tmp/EncoderParityAne/encoder-source
CAP=/var/tmp/EncoderParityAne/capture
REF=/var/tmp/EncoderParityAne/out-ane
RUNNER=$BASE/vulkan_encoder_resident.py
PY=$HOME/venv-agxgen/bin/python
LOCK=/tmp/m1-gpu.lock
OUT=${OUT:-$BASE/out-inline}
BUNDLES=$BASE/bundles

export PYTHONPATH="$BASE/tools:$BASE/tools/coreml:$BASE/tools/ane-export"

mkdir -p "$OUT"

snapshot() {
  local tag=$1
  {
    echo "tag=$tag"
    echo "at=$(date -Is)"
    echo "boot_id=$(cat /proc/sys/kernel/random/boot_id)"
    echo "uptime_start=$(uptime -s)"
    echo "device=$(stat -c '%F %a' /dev/accel/accel0 2>&1)"
    echo "module=$(lsmod | awk '$1=="ane"{print $1" refs="$3}')"
    echo "worker=$WORKER"
    echo "worker_sha256=$(sha256sum "$WORKER" | awk '{print $1}')"
    echo "liveness=$($PY "$BASE/tools/ane_worker_liveness.py" 2>&1)"
  } | tee "$OUT/state-$tag.txt"
}

encoder_arm() {
  local tag=$1; shift
  local out="$OUT/$tag"
  mkdir -p "$out" "$BASE/scratch-inline/$tag"
  date -Is > "$out/started_at"
  timeout --signal=KILL 900 \
    "$PY" "$RUNNER" \
      --source "$SRC" \
      --capture "$CAP" \
      --bundles "$BUNDLES" \
      --worker "$WORKER" \
      --libane "$LIBANE" \
      --scratch "$BASE/scratch-inline/$tag" \
      --out "$out" \
      --deadline-ms 20000 \
      "$@" \
      > "$out/stdout.txt" 2> "$out/stderr.txt"
  local rc=$?
  date -Is > "$out/finished_at"
  echo "$rc" > "$out/exit"
  echo "arm $tag exit=$rc"
  if [ "$rc" = 0 ]; then
    "$PY" "$BASE/compare_encoder.py" "$out" "$CAP" "$out/compare.json" \
      > "$out/compare.txt" 2>&1
    sha256sum "$out/encoder_hidden.npy" "$REF/encoder_hidden.npy" \
      > "$out/identity.txt"
  else
    tail -50 "$out/stderr.txt"
  fi
}

exec 9>"$LOCK"
if ! flock -w 1800 9; then
  echo "gpu lock busy; refusing"
  exit 1
fi

snapshot pre
encoder_arm launch
encoder_arm resident --resident
snapshot post

echo "=== done"
cat "$OUT"/launch/identity.txt "$OUT"/resident/identity.txt 2>/dev/null
