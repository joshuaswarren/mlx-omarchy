#!/bin/bash
set -e
PY=~/venv-agxgen/bin/python
PROBE=/var/tmp/EncoderResidualE99/encoder_precision_probe.py
SRC=/var/tmp/EncoderParityAne/encoder-source
CAP=/var/tmp/EncoderParityAne/capture
NAT=/var/tmp/EncoderParityAne/capture/encoder_hidden.npy
BASE=/var/tmp/EncoderResidualE99
LIBANE=/var/tmp/AneWorkerValidation-c05ba1df/receipts/2026-09-13-ane-worker-validation/libane.so

run_arm () {
  local tag=$1; shift
  echo "=== arm $tag"
  flock -w 900 /tmp/m1-gpu.lock "$PY" "$PROBE" \
    --source "$SRC" --capture "$CAP" \
    --bundles "$BASE" --worker /bin/true --libane "$LIBANE" \
    --scratch "$BASE/scratch" --out "$BASE/out-$tag" \
    --no-ane --native "$NAT" --tag "$tag" "$@" 2>&1 | grep -E 'rel_l2_vs_native|max_abs_vs_native|Error|error' || true
}

run_arm lnrqrt16 --precision '{"layer_norm":"rsqrt16"}'
run_arm lngamma16 --precision '{"layer_norm":"gamma16"}'
echo "=== round 3 done"
