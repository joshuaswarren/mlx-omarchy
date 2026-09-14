#!/bin/bash
# The three arms, run under the GPU lock held by ep3-run.sh.
#
#   ane3     islands A, B, C on the ANE   (the run this receipt is about)
#   ane2     islands A, C on the ANE      (same script, so the delta against
#                                          the 2026-09-14 0.024916 is
#                                          attributable to placement rather
#                                          than to the island-B code landing)
#   vulkan   no ANE at all                (attribution floor)
#
# No retry: a non-zero arm aborts with the device left alone.
set -euo pipefail

ROOT=/var/tmp/EncoderParity3Islands
SRC=/var/tmp/EncoderParityAne/encoder-source
CAPTURE=/var/tmp/EncoderParityAne/capture
BUNDLES=/var/tmp/jwm1-encoder-islands/bundles
WORKER=/var/tmp/jwm1-select-island-exec2/mlx-omarchy-ane-worker
LIBANE=/var/tmp/AneWorkerValidation-c05ba1df/receipts/2026-09-13-ane-worker-validation/libane.so
PY=$HOME/venv-agxgen/bin/python

arm() {
  local label="$1" out="$2"; shift 2
  echo "=== arm $label ($*)"
  "$PY" "$ROOT/derivation/vulkan_encoder.py" \
    --source "$SRC" --capture "$CAPTURE" --bundles "$BUNDLES" \
    --worker "$WORKER" --libane "$LIBANE" \
    --scratch "$ROOT/scratch/$label" --out "$ROOT/$out" \
    --deadline-ms 20000 "$@"
  cp "$ROOT/$out/run-report.json" "$ROOT/run-report-$label.json"
}

arm ane3   out-ane3   --islands ABC
arm ane2   out-ane2   --islands AC
arm vulkan out-vulkan --no-ane
