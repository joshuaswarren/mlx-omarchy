#!/bin/bash
# EncoderGpuBusy: baseline E2E battery r1..r6 with the BASE runner (release wheel).
set -euo pipefail
P=/var/tmp/enc-gpubusy
PY=/home/joshuawarren/venv-agxgen/bin/python
export PYTHONPATH=/var/tmp/ParakeetE2EBaseline7d82/site:/var/tmp/MelFrontendPerf/venv-cache/lib/python3.14/site-packages
unset HK_PERF HK_PERFTEST MLX_OMARCHY_GATED_BARRIERS MLX_OMARCHY_GPU_PROFILE || true
export MLX_OMARCHY_SPIRV_CACHE=$P/spirv
export ANE_ISLAND_MODE=resident-batch
E2E=/var/tmp/ParakeetE2EBaseline7d82/fused_e2e.py
RUNNER=$P/vulkan_encoder_base.py
E2EARGS=(--audio /var/tmp/ParakeetE2E/audio/fixture.flac
  --golden /var/tmp/EncoderParityAne/capture
  --model ~/.cache/mlx-omarchy/parakeet-reference/mweinbach1/parakeet-tdt-0.6b-v3-coreml/b650695c2322ee5281dff48d7345b2f3a58ff018
  --pkg /var/tmp/TdtLoopDefault/pkg
  --encoder-runner $RUNNER
  --source /var/tmp/EncoderParityAne/encoder-source
  --bundles /var/tmp/island-reexport/bundles
  --worker /var/tmp/mlx-main-strict/.work/mlx/build-ane-device/tools/mlx-omarchy-ane-worker/mlx-omarchy-ane-worker
  --libane /var/tmp/island-reexport/libane-strict.so
  --deadline-ms 20000)
stat -c "lock inode=%i held $(date -Iseconds)" /tmp/m1-gpu.lock
for r in 1 2 3 4 5 6; do
  echo "--- base E2E r$r $(date -Iseconds) ---"
  rm -rf $P/base-e2e-out-r$r $P/base-e2e-scratch-r$r
  mkdir -p $P/base-e2e-out-r$r $P/base-e2e-scratch-r$r
  flock -w 900 /tmp/m1-gpu.lock \
    $PY $E2E "${E2EARGS[@]}" --scratch $P/base-e2e-scratch-r$r --out $P/base-e2e-out-r$r
done
stat -c "lock inode=%i freed $(date -Iseconds)" /tmp/m1-gpu.lock
flock -n /tmp/m1-gpu.lock true && echo "LOCK-FREE-OK"
echo '--- identity ---'
sha256sum $P/base-e2e-out-r*/encoder_hidden.npy $P/base-e2e-out-r1/transcript.txt
