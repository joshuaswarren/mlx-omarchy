#!/bin/bash
# Gate 3: Parakeet E2E battery r1..r6, fused-fold-relan runner, no flags,
# non-diag wheel site, omarchy-ane libane-strict, standalone fd worker,
# GPU TDT loop default, island resident-batch. Lock /tmp/m1-gpu.lock:
# flock -w 900, never steal, never unlink.
set -euo pipefail
P=/var/tmp/enc-fold-reland/battery
PY=/home/joshuawarren/venv-agxgen/bin/python
export PYTHONPATH=/var/tmp/E2EREV/site:/var/tmp/MelFrontendPerf/venv-cache/lib/python3.14/site-packages
unset HK_PERF HK_PERFTEST MLX_OMARCHY_GATED_BARRIERS MLX_OMARCHY_GPU_PROFILE || true
export MLX_OMARCHY_SPIRV_CACHE=$P/spirv
export ANE_ISLAND_MODE=resident-batch
E2E=/var/tmp/ParakeetE2EBaseline7d82/fused_e2e.py
RUNNER=/var/tmp/enc-fold-reland/vulkan_encoder.py
E2EARGS=(--audio /var/tmp/ParakeetE2E/audio/fixture.flac
  --golden /var/tmp/EncoderParityAne/capture
  --model ~/.cache/mlx-omarchy/parakeet-reference/mweinbach1/parakeet-tdt-0.6b-v3-coreml/b650695c2322ee5281dff48d7345b2f3a58ff018
  --pkg /var/tmp/TdtLoopDefault/pkg
  --encoder-runner $RUNNER
  --source /var/tmp/EncoderParityAne/encoder-source
  --bundles /var/tmp/island-reexport/bundles
  --worker /var/tmp/mlx-main-strict/.work/mlx/build-ane-device/tools/mlx-omarchy-ane-worker/mlx-omarchy-ane-worker
  --libane /var/tmp/island-reexport/libane-strict.so
  --ane-reference /var/tmp/EncoderParityAne/out-ane/encoder_hidden.npy
  --deadline-ms 20000)
stat -c "lock inode=%i held $(date -Iseconds)" /tmp/m1-gpu.lock
for r in 1 2 3 4 5 6; do
  echo "--- E2E r$r $(date -Iseconds) ---"
  rm -rf $P/out-$r $P/scratch-$r
  mkdir -p $P/out-$r $P/scratch-$r
  flock -w 900 /tmp/m1-gpu.lock \
    $PY $E2E "${E2EARGS[@]}" --scratch $P/scratch-$r --out $P/out-$r
done
stat -c "lock inode=%i freed $(date -Iseconds)" /tmp/m1-gpu.lock
flock -n /tmp/m1-gpu.lock true && echo "LOCK-FREE-OK"
echo '--- identity ---'
sha256sum $P/out-*/encoder_hidden.npy $P/out-1/transcript.txt
