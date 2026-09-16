#!/bin/bash
# EncoderHostResidual: verification battery with the CLEAN cut runner.
# 1) standalone encoder warm+record (clean timing, pin + dispatch counts)
# 2) full fused_e2e E2E r1..r6 (104/104, transcript, stage walls)
set -euo pipefail
P=/var/tmp/enc-hostres
PY=/home/joshuawarren/venv-agxgen/bin/python
export PYTHONPATH=/var/tmp/ParakeetE2EBaseline7d82/site:/var/tmp/MelFrontendPerf/venv-cache/lib/python3.14/site-packages
unset HK_PERF HK_PERFTEST MLX_OMARCHY_GATED_BARRIERS || true
export MLX_OMARCHY_SPIRV_CACHE=/var/tmp/enc-attr2/spirv-pw3
export ANE_ISLAND_MODE=resident-batch
RUNNER=$P/vulkan_encoder_after.py
RUNARGS=(--source /var/tmp/EncoderParityAne/encoder-source
  --bundles /var/tmp/island-reexport/bundles
  --worker /var/tmp/mlx-main-strict/.work/mlx/build-ane-device/tools/mlx-omarchy-ane-worker/mlx-omarchy-ane-worker
  --libane /var/tmp/island-reexport/libane-strict.so
  --deadline-ms 20000
  --capture /var/tmp/EncoderParityAne/capture
  --islands ABC)
E2E=/var/tmp/ParakeetE2EBaseline7d82/fused_e2e.py
E2EARGS=(--audio /var/tmp/ParakeetE2E/audio/fixture.flac
  --golden /var/tmp/EncoderParityAne/capture
  --model ~/.cache/mlx-omarchy/parakeet-reference/mweinbach1/parakeet-tdt-0.6b-v3-coreml/b650695c2322ee5281dff48d7345b24e04c90f742ef6e35a251d3728611e46b/data
  --pkg /var/tmp/TdtLoopDefault/pkg
  --encoder-runner $RUNNER
  --source /var/tmp/EncoderParityAne/encoder-source
  --bundles /var/tmp/island-reexport/bundles
  --worker /var/tmp/mlx-main-strict/.work/mlx/build-ane-device/tools/mlx-omarchy-ane-worker/mlx-omarchy-ane-worker
  --libane /var/tmp/island-reexport/libane-strict.so
  --model ~/.cache/mlx-omarchy/parakeet-reference/mweinbach1/parakeet-tdt-0.6b-v3-coreml/b650695c2322ee5281dff48d7345b2f3a58ff018
  --deadline-ms 20000)
stat -c "lock inode=%i held $(date -Iseconds)" /tmp/m1-gpu.lock
echo '--- standalone warm ---'
mkdir -p $P/ver-enc-out-warm $P/ver-enc-scratch-warm $P/ver-enc-out-record $P/ver-enc-scratch-record
flock -w 900 /tmp/m1-gpu.lock \
  $PY $RUNNER "${RUNARGS[@]}" --scratch $P/ver-enc-scratch-warm --out $P/ver-enc-out-warm
echo '--- standalone record ---'
flock -w 900 /tmp/m1-gpu.lock \
  env MLX_OMARCHY_GPU_PROFILE=$P/profile-ver-record.jsonl \
  MLX_OMARCHY_GPU_PROFILE_LABEL=encoder-host-residual-ver-record \
  $PY $RUNNER "${RUNARGS[@]}" --scratch $P/ver-enc-scratch-record --out $P/ver-enc-out-record
for r in 1 2 3 4 5 6; do
  echo "--- E2E r$r ---"
  rm -rf $P/ver-out-r$r $P/ver-scratch-r$r
  mkdir -p $P/ver-out-r$r $P/ver-scratch-r$r
  flock -w 900 /tmp/m1-gpu.lock \
    $PY $E2E "${E2EARGS[@]}" --scratch $P/ver-scratch-r$r --out $P/ver-out-r$r
done
stat -c "lock inode=%i freed $(date -Iseconds)" /tmp/m1-gpu.lock
flock -n /tmp/m1-gpu.lock true && echo "LOCK-FREE-OK"
echo '--- identity ---'
sha256sum $P/ver-enc-out-warm/encoder_hidden.npy $P/ver-enc-out-record/encoder_hidden.npy \
  $P/ver-out-r*/encoder_hidden.npy $P/ver-out-r1/transcript.txt
