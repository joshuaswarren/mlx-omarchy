#!/bin/bash
# Parakeet E2E battery r1..r6 on jw16 (M1 Max, T6001) at the post-revert
# mlx-omarchy main tip f43ab71c (fold dropped; no flags), non-diag wheel,
# omarchy-ane libane-strict 6fa243a, standalone fd-protocol worker,
# GPU TDT loop default, island resident-batch, T6001 island bundles.
# Lock /tmp/m1-gpu.lock: flock -w 900, never steal, never unlink.
# llama-server is stopped for this window (Main-approved, <= 30 min).
set -euo pipefail
P=/var/tmp/E2EREV-battery
PY=/home/joshuawarren/venv-agxgen/bin/python
export PYTHONPATH=/var/tmp/E2EREV/site:/var/tmp/MelFrontendPerf/venv-cache/lib/python3.14/site-packages
unset HK_PERF HK_PERFTEST MLX_OMARCHY_GATED_BARRIERS MLX_OMARCHY_GPU_PROFILE || true
export MLX_OMARCHY_SPIRV_CACHE=$P/spirv
export ANE_ISLAND_MODE=resident-batch
E2E=/var/tmp/ParakeetE2EJw16/fused_e2e.py
RUNNER=/var/tmp/E2EREV/overlay/tools/coreml/vulkan_encoder.py
E2EARGS=(--audio /var/tmp/ParakeetE2E/audio/fixture.flac
  --golden /var/tmp/EncoderParityAne/capture
  --model ~/.cache/mlx-omarchy/parakeet-reference/mweinbach1/parakeet-tdt-0.6b-v3-coreml/b650695c2322ee5281dff48d7345b2f3a58ff018
  --pkg /var/tmp/TdtLoopDefault/pkg
  --encoder-runner $RUNNER
  --source /var/tmp/EncoderParityAne/encoder-source
  --worker /var/tmp/E2E296-battery/worker/mlx-omarchy-ane-worker
  --libane /var/tmp/island-reexport/libane-strict.so
  --bundles /var/tmp/island-reexport/bundles
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
