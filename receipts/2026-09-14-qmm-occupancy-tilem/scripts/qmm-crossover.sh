#!/usr/bin/env bash
# Where does halving the coopmat row tile stop paying?
#
# The four real prefill shapes only sample 4.1 and 28.9 workgroups per
# core, and 16 rows wins at the first and loses at the second, so the
# crossover sits somewhere between and the occupancy floor cannot be
# picked without finding it. These n values are not in Qwen2.5-0.5B;
# they exist to place the floor on measurement instead of on a guess.
#
# Per n, the wg-per-core value is chosen so the target lands strictly
# between the 32-row grid and the 16-row grid FOR THAT n, which makes
# the arm pair exactly 32 rows vs 16 rows. Arms alternate per round.
set -euo pipefail
OUT=${1:?usage: qmm-crossover.sh <outdir>}
ROUNDS=${2:-3}
PY="$OUT/venv/bin/python"
LOCK=/tmp/m1-gpu.lock
# n:forced-16-rows floor (32 cores): floor*32 in (33*ceil(n/32), 66*ceil(n/32)]
CASES="128:5 256:9 384:13 512:17 640:21 896:29"
exec 9<>"$LOCK"
flock -w 60 9 || { echo "could not take $LOCK" >&2; exit 1; }
echo "lock_inode=$(stat -c %i "$LOCK") nested_flock_n=$(flock -n "$LOCK" -c true; echo $?)"
: > "$OUT/crossover.jsonl"
for r in $(seq 1 "$ROUNDS"); do
  for case in $CASES; do
    n=${case%%:*}; pc16=${case##*:}
    for pc in 0 "$pc16"; do
      env -i HOME=/home/joshuawarren PATH=/usr/bin:/bin MLX_DISABLE_COMPILE=1 \
        HF_HUB_OFFLINE=1 MLX_OMARCHY_QMM_COOPMAT_WG_PER_CORE="$pc" \
        "$PY" "$OUT/qmm_shape_ab.py" --reps 40 --shapes "$n:896" 2>/dev/null |
        "$PY" -c "
import json, sys
d = json.loads(sys.stdin.read())
d['round'] = $r
print(json.dumps(d, sort_keys=True))" >> "$OUT/crossover.jsonl"
    done
  done
done
exec 9>&-
echo "nested_flock_n_after_release=$(flock -n "$LOCK" -c true; echo $?) inode=$(stat -c %i "$LOCK")"
wc -l "$OUT/crossover.jsonl"
