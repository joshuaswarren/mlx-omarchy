#!/bin/bash
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
#
# Isolates the per-submit cost of the two worker forms: 24 one-shot
# launches against 24 submits on one resident worker, same bundle, same
# input files, same binary, no MLX, no GPU lock, no encoder. Arms are
# interleaved over ROUNDS so machine load cannot land in one arm only.
#
# usage: submit_microbench.sh WORKER BUNDLE OUT_DIR [ROUNDS] -- \
#          NAME=FILE... : OUTPUT_NAME
set -u

WORKER=$1
BUNDLE=$2
OUT=$3
ROUNDS=${4:-3}
shift 4
[ "${1:-}" = "--" ] && shift

INPUTS=()
OUTPUT=""
collecting=inputs
for token in "$@"; do
  if [ "$token" = ":" ]; then collecting=output; continue; fi
  if [ "$collecting" = inputs ]; then INPUTS+=("$token"); else OUTPUT=$token; fi
done

LIBANE=/var/tmp/AneWorkerValidation-c05ba1df/receipts/2026-09-13-ane-worker-validation/libane.so
SUBMITS=24
mkdir -p "$OUT"

input_flags() {
  local flags=()
  for assignment in "${INPUTS[@]}"; do flags+=(--input "$assignment"); done
  printf '%s\n' "${flags[@]}"
}

launch_arm() {
  local round=$1
  local started ended
  started=$(date +%s%N)
  for index in $(seq 1 $SUBMITS); do
    timeout --signal=KILL 120 "$WORKER" \
      --bundle "$BUNDLE" --libane "$LIBANE" --deadline-ms 20000 \
      --iterations 1 \
      $(input_flags | tr '\n' ' ') \
      --save "$OUTPUT=$OUT/launch-$round-$index.bin" \
      >> "$OUT/launch-$round.log" 2>&1 || { echo "launch submit $index failed"; return 1; }
  done
  ended=$(date +%s%N)
  echo $(( (ended - started) / 1000000 ))
}

resident_arm() {
  local round=$1
  local started ended
  {
    for index in $(seq 1 $SUBMITS); do
      printf 'submit B'
      for assignment in "${INPUTS[@]}"; do printf ' --input %s' "$assignment"; done
      printf ' --save %s=%s/resident-%s-%s.bin\n' "$OUTPUT" "$OUT" "$round" "$index"
    done
    printf 'quit\n'
  } > "$OUT/jobs-$round.txt"
  started=$(date +%s%N)
  timeout --signal=KILL 300 "$WORKER" --serve \
    --bundle "B=$BUNDLE" --libane "$LIBANE" --deadline-ms 20000 \
    --iterations 1 \
    < "$OUT/jobs-$round.txt" > "$OUT/resident-$round.log" 2>&1 \
    || { echo "resident round $round failed"; return 1; }
  ended=$(date +%s%N)
  echo $(( (ended - started) / 1000000 ))
}

for round in $(seq 1 "$ROUNDS"); do
  launched=$(launch_arm "$round") || exit 1
  resident=$(resident_arm "$round") || exit 1
  echo "round=$round submits=$SUBMITS launch_ms=$launched resident_ms=$resident"
done

# Every arm must have produced identical bytes, or the timings mean nothing.
digests=$(sha256sum "$OUT"/launch-*.bin "$OUT"/resident-*.bin | awk '{print $1}' | sort -u)
echo "distinct_output_digests=$(echo "$digests" | wc -l)"
echo "$digests"
grep -h 'elapsed_ms' "$OUT"/resident-1.log | head -3
