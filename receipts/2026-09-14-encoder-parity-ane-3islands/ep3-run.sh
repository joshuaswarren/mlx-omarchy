#!/bin/bash
# Prerequisite gate, then the three arms under one GPU lock hold so nothing
# interleaves between them. The lock is waited for, never stolen.
set -euo pipefail

ROOT=/var/tmp/EncoderParity3Islands

for path in /var/tmp/EncoderParityAne/encoder-source/model.mil \
            /var/tmp/EncoderParityAne/capture/encoder_input_features.npy \
            /var/tmp/jwm1-select-island-exec2/mlx-omarchy-ane-worker \
            /var/tmp/AneWorkerValidation-c05ba1df/receipts/2026-09-13-ane-worker-validation/libane.so \
            /var/tmp/jwm1-encoder-islands/bundles/island-attn-a-kt/manifest.json \
            /var/tmp/jwm1-encoder-islands/bundles/island-pv/manifest.json \
            /var/tmp/jwm1-encoder-islands/bundles/island-select-8head-scratch417/manifest.json; do
  [ -e "$path" ] || { echo "missing prerequisite: $path" >&2; exit 2; }
done

flock -w 1800 /tmp/m1-gpu.lock bash "$ROOT/ep3-arms.sh"
echo "=== all three arms complete"
