#!/bin/bash
# Parakeet E2E baseline on a non-diag wheel from origin/main 7d82ec94.
# No decode flags: GPU-resident TDT loop is the default path.
# Island batch: ANE_ISLAND_MODE=resident-batch (one bounded batch submit).
# Strict libane: wheel-pinned omarchy-ane 6fa243a; worker built from the
# same tree (f171a61e) dlopening libane-strict.so (56b46234).
# Lock /tmp/m1-gpu.lock: flock -w 900, never steal, never unlink.
set -e
RUN_IDX="$1"
RUN=/var/tmp/ParakeetE2EBaseline7d82
SITE=$RUN/site
CACHE=/var/tmp/MelFrontendPerf/venv-cache/lib/python3.14/site-packages
export MLX_OMARCHY_SPIRV_CACHE=/var/tmp/MelFrontendPerf/spirv-ab.3KDGHZ
export PYTHONPATH=$SITE:$CACHE
export ANE_ISLAND_MODE=resident-batch
OUT=$RUN/out-r$RUN_IDX
SCR=$RUN/scratch-r$RUN_IDX
rm -rf "$OUT" "$SCR"
mkdir -p "$OUT" "$SCR"
flock -w 900 /tmp/m1-gpu.lock \
  /home/joshuawarren/venv-agxgen/bin/python /var/tmp/ParakeetE2EBaseline7d82/fused_e2e.py \
  --audio /var/tmp/ParakeetE2E/audio/fixture.flac \
  --golden /var/tmp/EncoderParityAne/capture \
  --model ~/.cache/mlx-omarchy/parakeet-reference/mweinbach1/parakeet-tdt-0.6b-v3-coreml/b650695c2322ee5281dff48d7345b2f3a58ff018 \
  --pkg /var/tmp/TdtLoopDefault/pkg \
  --encoder-runner /var/tmp/TdtLoopDefault/pkg/coreml/vulkan_encoder.py \
  --source /var/tmp/EncoderParityAne/encoder-source \
  --bundles /var/tmp/island-reexport/bundles \
  --worker /var/tmp/mlx-main-strict/.work/mlx/build-ane-device/tools/mlx-omarchy-ane-worker/mlx-omarchy-ane-worker \
  --libane /var/tmp/island-reexport/libane-strict.so \
  --scratch "$SCR" \
  --ane-reference /var/tmp/EncoderParityAne/out-ane/encoder_hidden.npy \
  --out "$OUT" \
  --deadline-ms 20000
