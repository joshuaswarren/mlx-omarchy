#!/usr/bin/env bash
# Full M1 acceptance sequence for one wheel: fixed-input projections, f32/f16
# stage probes, then the six legs. Waits for /tmp/m1-gpu.lock politely.
#   m1-run-all.sh SOURCE_COMMIT
set -euo pipefail
cd "$(dirname "$0")/../.."
here=receipts/2026-09-09-q4-gemv-order
commit="$1"
while ! flock -n /tmp/m1-gpu.lock true; do sleep 20; done
flock -w 600 /tmp/m1-gpu.lock timeout 7200 bash -euo pipefail -c '
  here="$1"; commit="$2"
  export MESA_SHADER_CACHE_DISABLE=true
  unset VK_DRIVER_FILES PYTHONPATH
  "$here/m1-window.sh" "$commit" fixed
  py=.venv-accept/bin/python
  $py "$here/probe-stages.py" "$here/probe-896-f32.json" 896 8 float32 "" reduce,inputsum,dotonly,scaleonly,dot,full
  $py "$here/probe-stages.py" "$here/probe-1024-f32.json" 1024 8 float32 "" full
  $py "$here/probe-stages.py" "$here/probe-4864-f32.json" 4864 4 float32 "" full
  $py "$here/probe-stages.py" "$here/probe-896-f16.json" 896 16 float16 "" full
  "$here/m1-window.sh" "$commit" legs
' _ "$here" "$commit"
