#!/usr/bin/env bash
# ab_all.sh <rounds> <out.jsonl>
# Receipt-grade interleaved screen of every arm in one GPU lock hold.
set -euo pipefail
ROUNDS=${1:?rounds}
OUT=${2:?out}
W=/var/tmp/qmm-coop-arms
BASEPY=/var/tmp/mlx-omarchy-profile-enabled-b41e2b74/venv/bin/python
ARMS=(
  "base=$BASEPY"
  "pad2=$W/venv-pad2/bin/python"
  "pad4=$W/venv-pad4/bin/python"
  "mat16=$W/venv-mat16/bin/python"
  "swz8=$W/venv-swz8/bin/python"
  "clamp=$W/venv-clamp/bin/python"
  "uvec4=$W/venv-uvec4/bin/python"
)
cd /tmp
timeout -k 10s 1800s flock -w 60 /tmp/m1-gpu.lock \
  bash /home/joshuawarren/benchq/qmmpad/ab.sh "$ROUNDS" "${ARMS[@]}" \
  > "$OUT" 2>&1
echo "rc=$? rows=$(grep -c median_ms "$OUT")"
