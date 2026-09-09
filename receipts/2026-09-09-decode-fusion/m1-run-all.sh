#!/usr/bin/env bash
# DecodeFusion M1 window. Steps, each under its own flock on
# /tmp/m1-gpu.lock (never nested, never unlinked):
#   1. candidate checkout at the bundled wave/DecodeFusion head:
#      diagnostics wheel (moved to wheels/diag), then release wheel
#      (stays in dist/, the only wheel there)
#   2. base checkout (origin/main 6ca33e0): release wheel into dist/
#   3. targeted suites on the candidate (m1-tests.sh)
#   4. decode profile on the candidate diagnostics wheel (m1-model-profile.sh)
#   5. fixed-projection exactness on the candidate release wheel
#   6. paired six-leg run, base vs candidate, 3 repetitions (m1-legs.sh)
set -uo pipefail
cand=~/src/mlx-DecodeFusion
base=~/src/mlx-DecodeFusion-base
out=$cand/receipts-df
mkdir -p "$out"
echo "hostname: $(hostname)  start: $(date -u +%FT%TZ)"
flock -w 900 /tmp/m1-gpu.lock timeout 3600 bash -c '
  set -euo pipefail
  cd '"$cand"'
  git fetch -q /tmp/df.bundle wave/DecodeFusion
  git checkout -q wave/DecodeFusion && git reset -q --hard FETCH_HEAD
  git log --oneline -1
  rm -f dist/*.whl
  bash /tmp/df-m1-build.sh . diagnostics
  bash /tmp/df-m1-build.sh . release
  cd '"$base"'
  rm -f dist/*.whl
  bash /tmp/df-m1-build.sh . release
' 2>&1 | tee "$out/build.log"
flock -w 900 /tmp/m1-gpu.lock timeout 7200 bash /tmp/df-m1-tests.sh "$cand" "$out/m1-tests" 2>&1 | tee "$out/m1-tests.log"
flock -w 900 /tmp/m1-gpu.lock timeout 3600 bash /tmp/df-m1-model-profile.sh "$cand" "$out/profile-cand" 2>&1 | tail -3
flock -w 900 /tmp/m1-gpu.lock timeout 1800 bash -c '
  set -euo pipefail
  cd '"$cand"'
  unset HK_PERF HK_PERFTEST MLX_OMARCHY_SOURCE_COMMIT MLX_OMARCHY_LOCAL_VERSION MLX_OMARCHY_GPU_PROFILE PYTHONPATH VK_DRIVER_FILES
  export MESA_SHADER_CACHE_DISABLE=true HF_HUB_OFFLINE=1 MLX_DISABLE_COMPILE=1
  wheel=(dist/mlx_omarchy-*+$(git rev-parse --short=7 HEAD)-*.whl)
  test "${#wheel[@]}" -eq 1
  .venv-accept/bin/python -m pip install -q --no-deps --force-reinstall "${wheel[0]}"
  scratch="$(mktemp -d /tmp/df-fixed-projection.XXXX)"
  .venv-accept/bin/python /tmp/df-fixed-projection.py /tmp/mlx-native-q4-long-operations "$scratch" scripts 2>&1 | tee '"$out"'/fixed-projection.log
  cp "$scratch/results.json" '"$out"'/fixed-projection-results.json
  cp "$scratch/provenance.json" '"$out"'/fixed-projection-provenance.json
' 2>&1 | tail -8
if [ -e "$out/legs" ]; then mv "$out/legs" "$out/legs.stale-$(date +%s)"; fi
flock -w 900 /tmp/m1-gpu.lock timeout 10800 bash /tmp/df-m1-legs.sh "$cand" "$base" "$out/legs" 3 2>&1 | tail -12
echo "end: $(date -u +%FT%TZ)"
echo ALL_DONE
