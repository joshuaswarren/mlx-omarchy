#!/bin/bash
# Standalone clean E2E rerun (documented ParakeetE2E procedure, coopmat wheel).
# Lock /tmp/m1-gpu.lock: flock -w 900, never steal, never unlink.
set -e
RUN=/var/tmp/ParakeetE2ECoopmat
SITE=$RUN/site
CACHE=/var/tmp/MelFrontendPerf/venv-cache/lib/python3.14/site-packages
export MLX_OMARCHY_SPIRV_CACHE=/var/tmp/MelFrontendPerf/spirv-ab.3KDGHZ
export PYTHONPATH=$SITE:$CACHE
flock -w 900 /tmp/m1-gpu.lock \
  /home/joshuawarren/venv-agxgen/bin/python /var/tmp/ParakeetE2E/parakeet_e2e.py \
  --audio /var/tmp/ParakeetE2E/audio/fixture.flac \
  --golden /var/tmp/EncoderParityAne/capture \
  --model ~/.cache/mlx-omarchy/parakeet-reference/mweinbach1/parakeet-tdt-0.6b-v3-coreml/b650695c2322ee5281dff48d7345b2f3a58ff018 \
  --pkg /var/tmp/ParakeetE2EAneBnns/pkg \
  --encoder-runner $RUN/vulkan_encoder.py \
  --source /var/tmp/EncoderParityAne/encoder-source \
  --bundles /var/tmp/jwm1-encoder-islands/bundles \
  --worker /var/tmp/jwm1-select-island-exec2/mlx-omarchy-ane-worker \
  --libane /var/tmp/AneWorkerValidation-c05ba1df/receipts/2026-09-13-ane-worker-validation/libane.so \
  --scratch $RUN/scratch-r4b \
  --ane-reference /var/tmp/EncoderParityAne/out-ane/encoder_hidden.npy \
  --out $RUN/out-r4b \
  --deadline-ms 20000
echo "E2E_R4B_DONE"
