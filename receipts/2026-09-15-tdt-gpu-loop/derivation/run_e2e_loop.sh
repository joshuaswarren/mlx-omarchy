#!/bin/bash
# ANE-arm e2e gate for the GPU-resident TDT loop (--tdt-loop).
set -e
exec 9>/tmp/m1-gpu.lock
flock -x -w 3600 9
export PYTHONPATH=/var/tmp/MelFrontendPerf/venv-cache/lib/python3.14/site-packages
OUT=/var/tmp/TdtGpuLoop/e2e-after
SCR=/var/tmp/TdtGpuLoop/e2e-scratch
rm -rf "$OUT" "$SCR"
mkdir -p "$OUT" "$SCR"
~/venv-agxgen/bin/python /var/tmp/TdtGpuLoop/fused_e2e.py \
  --audio /var/tmp/ParakeetE2E/audio/fixture.flac \
  --golden /var/tmp/EncoderParityAne/capture \
  --model $HOME/.cache/mlx-omarchy/parakeet-reference/mweinbach1/parakeet-tdt-0.6b-v3-coreml/b650695c2322ee5281dff48d7345b2f3a58ff018 \
  --pkg /var/tmp/TdtGpuLoop/pkg \
  --encoder-runner /var/tmp/ParakeetE2EFusedLeftover-stage/vulkan_encoder.py \
  --source /var/tmp/EncoderParityAne/encoder-source \
  --bundles /var/tmp/jwm1-encoder-islands/bundles \
  --worker /var/tmp/jwm1-select-island-exec2/mlx-omarchy-ane-worker \
  --libane /var/tmp/AneWorkerValidation-c05ba1df/receipts/2026-09-13-ane-worker-validation/libane.so \
  --scratch "$SCR" --out "$OUT" --deadline-ms 20000 --tdt-loop
