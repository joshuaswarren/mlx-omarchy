#!/usr/bin/env bash
# Watch phase2 state; log transitions. Runs until phase2 completes or fails.
S="$HOME/src/mlx-HostPathOverhead/receipts/2026-09-11-pyenv-parity"
last=""
while true; do
  cur="$(cat $S/phase2.state 2>/dev/null | tr '\n' ' ')fail=$(grep -c FATAL $S/phase2.log 2>/dev/null)"
  if [ "$cur" != "$last" ]; then
    echo "[monitor $(date -u +%FT%TZ)] $cur"
    last="$cur"
  fi
  grep -q "phase2 complete" "$S/phase2.log" 2>/dev/null && { echo "[monitor] COMPLETE"; break; }
  grep -q "FATAL" "$S/phase2.log" 2>/dev/null && { echo "[monitor] FATAL - needs intervention"; break; }
  sleep 120
done
