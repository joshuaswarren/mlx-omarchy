#!/bin/bash
# Encoder residual attribution arms, jwm1-linux, GPU-only under the flock.
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

run_arm fp32chain --fp32-chain
run_arm softmax16 --precision '{"softmax":"fp16"}'
run_arm ln16 --precision '{"layer_norm":"fp16"}'
run_arm silu16 --precision '{"silu":"fp16","sigmoid":"fp16"}'
run_arm linear16 --precision '{"linear":"fp16"}'
run_arm matmul16 --precision '{"matmul":"fp16"}'
run_arm conv16 --precision '{"conv":"fp16"}'
run_arm full16 --precision '{"softmax":"fp16","layer_norm":"fp16","silu":"fp16","sigmoid":"fp16","linear":"fp16","matmul":"fp16","conv":"fp16"}'
echo "=== all arms done"
