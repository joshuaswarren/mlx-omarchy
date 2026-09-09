#!/usr/bin/env bash
# Paired six-leg run, baseline release wheel against candidate release
# wheel, alternating order every repetition (paired-legs.py).
#   m1-legs.sh CANDIDATE_CHECKOUT BASE_CHECKOUT OUT_DIR REPS
# Run under `flock -w 900 /tmp/m1-gpu.lock timeout 7200`.
set -euo pipefail
cand="$1"; base="$2"; out="$3"; reps="$4"
cd "$cand"
unset HK_PERF HK_PERFTEST MLX_OMARCHY_SOURCE_COMMIT MLX_OMARCHY_LOCAL_VERSION MLX_OMARCHY_GPU_PROFILE PYTHONPATH VK_DRIVER_FILES
export MESA_SHADER_CACHE_DISABLE=true
cand_commit="$(git rev-parse --short=7 HEAD)"
base_commit="$(git -C "$base" rev-parse --short=7 HEAD)"
test ! -e "$out"
.venv-accept/bin/python /tmp/gb-paired-legs.py . "$out" "$reps" \
  "base=$base=$base_commit" "cand=$cand=$cand_commit" 2>&1 | tee "$out.log"
