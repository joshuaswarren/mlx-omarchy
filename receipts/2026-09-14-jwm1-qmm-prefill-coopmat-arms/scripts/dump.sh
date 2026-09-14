#!/usr/bin/env bash
# dump.sh <venv-python> <out.log>
# Capture the installed driver's AGX shader dump while the single-shape
# dominant-cell probe runs. No ICD override: the system fork driver.
set -euo pipefail
PY=${1:?python}
OUT=${2:?out log}
cd /home/joshuawarren/benchq/qmm-coop-bench
timeout -k 10s 900s flock -w 60 /tmp/m1-gpu.lock env \
  AGX_SIMDMAT=1 MESA_SHADER_CACHE_DISABLE=true AGX_MESA_DEBUG=shaders \
  "$PY" qmm_dump_probe.py > "$OUT" 2>&1
echo "rc=$? lines=$(wc -l < "$OUT")"
grep -c simd_matrix_fmadd32 "$OUT" || true
tail -2 "$OUT"
