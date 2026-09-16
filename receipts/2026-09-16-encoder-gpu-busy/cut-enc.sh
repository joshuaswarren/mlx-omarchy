#!/bin/bash
# EncoderGpuBusy: cut runner standalone legs — warm (release wheel) + record (diag wheel).
set -euo pipefail
P=/var/tmp/enc-gpubusy
PY=/home/joshuawarren/venv-agxgen/bin/python
ARGS=(--source /var/tmp/EncoderParityAne/encoder-source
  --bundles /var/tmp/island-reexport/bundles
  --worker /var/tmp/mlx-main-strict/.work/mlx/build-ane-device/tools/mlx-omarchy-ane-worker/mlx-omarchy-ane-worker
  --libane /var/tmp/island-reexport/libane-strict.so
  --deadline-ms 20000
  --capture /var/tmp/EncoderParityAne/capture
  --islands ABC)
stat -c "lock inode=%i held $(date -Iseconds)" /tmp/m1-gpu.lock

echo "=== cut warm (release wheel) r1 ==="
export PYTHONPATH=/var/tmp/ParakeetE2EBaseline7d82/site:/var/tmp/MelFrontendPerf/venv-cache/lib/python3.14/site-packages
unset HK_PERF HK_PERFTEST MLX_OMARCHY_GATED_BARRIERS MLX_OMARCHY_GPU_PROFILE || true
export MLX_OMARCHY_SPIRV_CACHE=$P/spirv
export ANE_ISLAND_MODE=resident-batch
mkdir -p $P/cut-warm-out1 $P/cut-warm-scratch1
flock -w 900 /tmp/m1-gpu.lock \
  $PY $P/vulkan_encoder_cut.py "${ARGS[@]}" --scratch $P/cut-warm-scratch1 --out $P/cut-warm-out1
sha256sum $P/cut-warm-out1/encoder_hidden.npy

echo "=== cut warm (release wheel) r2 ==="
mkdir -p $P/cut-warm-out2 $P/cut-warm-scratch2
flock -w 900 /tmp/m1-gpu.lock \
  $PY $P/vulkan_encoder_cut.py "${ARGS[@]}" --scratch $P/cut-warm-scratch2 --out $P/cut-warm-out2
sha256sum $P/cut-warm-out2/encoder_hidden.npy

echo "=== cut record (diag wheel) ==="
export PYTHONPATH=$P/site-diag:/var/tmp/MelFrontendPerf/venv-cache/lib/python3.14/site-packages
export MLX_OMARCHY_SPIRV_CACHE=$P/spirv-diag
mkdir -p $P/cut-rec-out $P/cut-rec-scratch
flock -w 900 /tmp/m1-gpu.lock \
  env MLX_OMARCHY_GPU_PROFILE=$P/cut-record.jsonl MLX_OMARCHY_GPU_PROFILE_LABEL=encgpubusy-cut-record \
  $PY $P/vulkan_encoder_cut.py "${ARGS[@]}" --scratch $P/cut-rec-scratch --out $P/cut-rec-out
sha256sum $P/cut-rec-out/encoder_hidden.npy

echo "=== baseline record (diag wheel, base runner) ==="
mkdir -p $P/base-rec-out $P/base-rec-scratch
flock -w 900 /tmp/m1-gpu.lock \
  env MLX_OMARCHY_GPU_PROFILE=$P/base-record.jsonl MLX_OMARCHY_GPU_PROFILE_LABEL=encgpubusy-base-record \
  $PY $P/vulkan_encoder_base.py "${ARGS[@]}" --scratch $P/base-rec-scratch --out $P/base-rec-out
sha256sum $P/base-rec-out/encoder_hidden.npy

unset MLX_OMARCHY_GPU_PROFILE MLX_OMARCHY_GPU_PROFILE_LABEL || true
stat -c "lock inode=%i freed $(date -Iseconds)" /tmp/m1-gpu.lock
flock -n /tmp/m1-gpu.lock true && echo "LOCK-FREE-OK"
