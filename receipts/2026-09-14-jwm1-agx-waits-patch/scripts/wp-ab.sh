#!/usr/bin/env bash
# wp-ab.sh <rounds> <outdir>
# Interleaved 8-cell kernel A/B of the two ICD arms inside ONE lock hold,
# so neither arm gets a quieter machine than the other. Same venv, same
# kernel source, same shapes: the driver is the only variable.
set -euo pipefail

ROUNDS=${1:?rounds}
OUT=${2:?outdir}
D=/home/joshuawarren/benchq/waitspatch
PROBE=/home/joshuawarren/benchq/qmmpad/probe.py

mkdir -p "$OUT"

timeout -k 10s 1800s flock -w 60 /tmp/m1-gpu.lock bash -c '
set -euo pipefail
ROUNDS=$1; OUT=$2; D=$3; PROBE=$4
# Prove the lock is actually held while the screen runs.
flock -n /tmp/m1-gpu.lock -c true && { echo "ERROR: lock not held"; exit 1; }
echo "lock_inode=$(stat -c %i /tmp/m1-gpu.lock) nested_flock_n=1 (held)"
for r in $(seq 1 "$ROUNDS"); do
  for arm in base patched; do
    "$D/py-$arm" "$PROBE" \
      | sed "s/^{/{\"round\": $r, \"arm_name\": \"$arm\", /"
  done
done
' _ "$ROUNDS" "$OUT" "$D" "$PROBE" | tee "$OUT/ab.jsonl"

echo "== lock after: $(fuser /tmp/m1-gpu.lock >/dev/null 2>&1 && echo held || echo free)"
