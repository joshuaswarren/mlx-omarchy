#!/bin/bash
# usage: run-suite.sh <tag> <icd.json> <suite>...  (runs on jwm1, logs to ~/log/cops-<tag>-<suite>.log)
set -u
tag=$1; icd=$2; shift 2
cd ~/src/mlx-complex-ops-20260908/.work/build/tests/omarchy || exit 1
for s in "$@"; do
  log=~/log/cops-${tag}-${s}.log
  VK_ICD_FILENAMES=$icd AGX_SIMDMAT=1 timeout 1800 flock /tmp/m1-gpu.lock ./omarchy_${s}_tests > "$log" 2>&1
  echo "== $s exit=$? $(tail -3 "$log" | tr '\n' ' ')"
done
