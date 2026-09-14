#!/usr/bin/env bash
# Interleaved arm-by-arm confirmation of the per-shape result, inside one
# lock hold. The sweep in shape-ab.jsonl runs each arm once in sequence,
# which cannot separate an arm effect from drift across the sequence;
# this alternates arms round by round so drift hits both equally.
set -euo pipefail
OUT=${1:?usage: qmm-interleave.sh <outdir>}
ROUNDS=${2:-5}
PY="$OUT/venv/bin/python"
LOCK=/tmp/m1-gpu.lock
exec 9<>"$LOCK"
flock -w 60 9 || { echo "could not take $LOCK" >&2; exit 1; }
echo "lock_inode=$(stat -c %i "$LOCK") nested_flock_n=$(flock -n "$LOCK" -c true; echo $?)"
: > "$OUT/interleave.jsonl"
for r in $(seq 1 "$ROUNDS"); do
  for pc in 0 8 16; do
    env -i HOME=/home/joshuawarren PATH=/usr/bin:/bin MLX_DISABLE_COMPILE=1 \
      HF_HUB_OFFLINE=1 MLX_OMARCHY_QMM_COOPMAT_WG_PER_CORE="$pc" \
      "$PY" "$OUT/qmm_shape_ab.py" --reps 40 2>/dev/null |
      "$PY" -c "
import json, sys
d = json.loads(sys.stdin.read())
d['round'] = $r
print(json.dumps(d, sort_keys=True))" >> "$OUT/interleave.jsonl"
  done
done
exec 9>&-
echo "nested_flock_n_after_release=$(flock -n "$LOCK" -c true; echo $?) inode=$(stat -c %i "$LOCK")"
wc -l "$OUT/interleave.jsonl"
