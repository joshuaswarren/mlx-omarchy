#!/bin/bash
# Gate 2: standalone encoder leg, fused runner, real libane-strict,
# flock -w 900 never steal. Expected encoder_hidden sha 38c73261f29230276ed76f1fc017b76b024156d79218bd5f1347fdc7e7d43ec7
set -euo pipefail
P=/var/tmp/enc-fold-reland
PY=/home/joshuawarren/venv-agxgen/bin/python
export PYTHONPATH=/var/tmp/E2EREV/site:/var/tmp/MelFrontendPerf/venv-cache/lib/python3.14/site-packages
unset HK_PERF HK_PERFTEST MLX_OMARCHY_GATED_BARRIERS MLX_OMARCHY_GPU_PROFILE || true
export MLX_OMARCHY_SPIRV_CACHE=$P/spirv-enc
export ANE_ISLAND_MODE=resident-batch
RUNNER=$P/vulkan_encoder.py
RUNARGS=(--source /var/tmp/EncoderParityAne/encoder-source
  --bundles /var/tmp/island-reexport/bundles
  --worker /var/tmp/mlx-main-strict/.work/mlx/build-ane-device/tools/mlx-omarchy-ane-worker/mlx-omarchy-ane-worker
  --libane /var/tmp/island-reexport/libane-strict.so
  --deadline-ms 20000
  --capture /var/tmp/EncoderParityAne/capture
  --islands ABC)
rm -rf $P/enc-out-warm $P/enc-scratch-warm
mkdir -p $P/enc-out-warm $P/enc-scratch-warm
flock -w 900 /tmp/m1-gpu.lock \
  $PY $RUNNER "${RUNARGS[@]}" --scratch $P/enc-scratch-warm --out $P/enc-out-warm
sha256sum $P/enc-out-warm/encoder_hidden.npy
