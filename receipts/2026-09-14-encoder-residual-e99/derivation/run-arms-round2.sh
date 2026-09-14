#!/bin/bash
# Attribution arms round 2: refined LN emulation + LN-combos.
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

run_arm ln16v2 --precision '{"layer_norm":"fp16"}'
run_arm softmax16v2 --precision '{"softmax":"fp16"}'
run_arm ln16v2_softmax16 --precision '{"layer_norm":"fp16","softmax":"fp16"}'
run_arm ln16v2_linear16 --precision '{"layer_norm":"fp16","linear":"fp16"}'
run_arm full16v2 --precision '{"layer_norm":"fp16","softmax":"fp16","silu":"fp16","sigmoid":"fp16"}'
echo "=== round 2 done"
