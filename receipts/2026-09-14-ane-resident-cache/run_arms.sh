#!/bin/bash
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
#
# Phase 8 measurement on jwm1-linux: resident worker plus compiled-model
# cache against the 48-launch baseline, all arms in one boot with one
# worker binary.
#
# Arms, in order:
#   cache-miss    resolve both island packages through the compiled cache
#   resident-1    encoder pass, one resident worker for all 48 submits
#   cache-hit     resolve again; must be a hit and must not re-adapt
#   resident-2    second encoder pass on the cache-hit bundles
#   launch        the 48-launch baseline, same binary, same boot
#
# No retries, no loops, no reboot, no module load or unload, no SET
# write, no 1x896. The GPU lock is taken with flock -w and never stolen
# or unlinked. A failing arm does not stop the ones after it, but it is
# never re-run.
set -u

BASE=/var/tmp/AneResidentCache
WORKER=${WORKER:-$BASE/mlx-omarchy-ane-worker}
LIBANE=/var/tmp/AneWorkerValidation-c05ba1df/receipts/2026-09-13-ane-worker-validation/libane.so
SRC=/var/tmp/EncoderParityAne/encoder-source
CAP=/var/tmp/EncoderParityAne/capture
QUALIFIED=/var/tmp/jwm1-encoder-islands/bundles
RUNNER=$BASE/vulkan_encoder_resident.py
PY=$HOME/venv-agxgen/bin/python
LOCK=/tmp/m1-gpu.lock
OUT=${OUT:-$BASE/out}
CACHE=$BASE/cache
BUNDLES=$BASE/bundles

export PYTHONPATH="$BASE/tools:$BASE/tools/coreml:$BASE/tools/ane-export"

mkdir -p "$OUT" "$BUNDLES"

snapshot() {
  local tag=$1
  {
    echo "tag=$tag"
    echo "at=$(date -Is)"
    echo "boot_id=$(cat /proc/sys/kernel/random/boot_id)"
    echo "uptime_start=$(uptime -s)"
    echo "device=$(stat -c '%F %a' /dev/accel/accel0 2>&1)"
    echo "module=$(lsmod | awk '$1=="ane"{print $1" refs="$3}')"
    echo "quarantine_bytes=$(stat -c %s /run/lock/mlx-omarchy-ane/quarantine 2>/dev/null || echo 0)"
    echo "liveness=$($PY "$BASE/tools/ane_worker_liveness.py" 2>&1)"
    echo "dmesg_tm_failed=$(dmesg 2>/dev/null | grep -c 'tm execution failed')"
    echo "dmesg_errno110=$(dmesg 2>/dev/null | grep -cE 'errno[ =-]*110')"
  } | tee "$OUT/state-$tag.txt"
}

resolve_bundles() {
  # jwm1 has no GNU time, so the wall clock comes from the shell; the
  # per-package seconds the resolver prints itself are the finer number.
  local tag=$1
  local started ended
  started=$(date +%s%N)
  "$PY" -m coreml.bundle_cache \
      --cache-root "$CACHE" \
      --source-repo joshuaswarren/mlx-omarchy \
      --source-commit 030bab6ff6c279a5d76770b9dfc4ab03aa7975d8 \
      --model parakeet-tdt-0.6b-v3-encoder \
      --package "island-attn-a-kt=$BASE/packages/island-attn-a-kt" \
      --package "island-pv=$BASE/packages/island-pv" \
      > "$OUT/cache-$tag.txt" 2>&1
  local rc=$?
  ended=$(date +%s%N)
  echo "$(( (ended - started) / 1000000 ))" > "$OUT/cache-$tag.ms"
  echo "cache $tag exit=$rc wall_ms=$(cat "$OUT/cache-$tag.ms")"
  cat "$OUT/cache-$tag.txt"
  return $rc
}

publish_bundles() {
  # The cache stores entries by key digest; the runner wants one
  # directory per island name.
  while read -r _ name _ _ path; do
    rm -rf "$BUNDLES/$name"
    cp -r "$path" "$BUNDLES/$name"
  done < <(grep '^bundle ' "$OUT/cache-miss.txt")
  ls -R "$BUNDLES" > "$OUT/bundles-published.txt"
  for name in island-attn-a-kt island-pv; do
    {
      echo "== $name"
      diff -r "$QUALIFIED/$name" "$BUNDLES/$name" && echo "identical to the qualified bundle"
      sha256sum "$BUNDLES/$name"/* "$QUALIFIED/$name"/*
    } >> "$OUT/bundle-equivalence.txt" 2>&1
  done
  for name in island-attn-a-kt island-pv; do
    if [ ! -f "$BUNDLES/$name/manifest.json" ]; then
      echo "no published bundle for $name; refusing to spend the device window"
      exit 1
    fi
  done
}

encoder_arm() {
  local tag=$1; shift
  local out="$OUT/$tag"
  mkdir -p "$out" "$BASE/scratch/$tag"
  date -Is > "$out/started_at"
  flock -w 1800 "$LOCK" timeout --signal=KILL 900 \
    "$PY" "$RUNNER" \
      --source "$SRC" \
      --capture "$CAP" \
      --bundles "$BUNDLES" \
      --worker "$WORKER" \
      --libane "$LIBANE" \
      --scratch "$BASE/scratch/$tag" \
      --out "$out" \
      --deadline-ms 20000 \
      "$@" \
      > "$out/stdout.txt" 2> "$out/stderr.txt"
  local rc=$?
  date -Is > "$out/finished_at"
  echo "$rc" > "$out/exit"
  echo "arm $tag exit=$rc"
  tail -3 "$out/stderr.txt" 2>/dev/null
  if [ "$rc" = 0 ]; then
    "$PY" "$BASE/compare_encoder.py" "$out" "$CAP" "$out/compare.json" \
      > "$out/compare.txt" 2>&1
    echo "arm $tag compare exit=$? $(grep -E '\"(rel_l2_err|encoder_hidden_sha256|all_bounds_pass)\"' "$out/compare.json" | tr -d ' ' | tr '\n' ' ')"
  fi
}

snapshot pre
rm -rf "$CACHE"
resolve_bundles miss
publish_bundles
encoder_arm resident-1 --resident
resolve_bundles hit
encoder_arm resident-2 --resident
encoder_arm launch
snapshot post

echo "=== done"
