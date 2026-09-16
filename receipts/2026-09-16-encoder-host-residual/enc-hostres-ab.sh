#!/bin/bash
# EncoderHostResidual: A/B the host-graph residual, instrumented arms only.
# before = fused baseline runner (83de2255), after = precomputed releases.
# Alternating order to cancel warmth drift; every run pin-checked.
set -euo pipefail
P=/var/tmp/enc-hostres
PY=/home/joshuawarren/venv-agxgen/bin/python
export PYTHONPATH=/var/tmp/ParakeetE2EBaseline7d82/site:/var/tmp/MelFrontendPerf/venv-cache/lib/python3.14/site-packages
unset HK_PERF HK_PERFTEST MLX_OMARCHY_GATED_BARRIERS || true
export MLX_OMARCHY_SPIRV_CACHE=/var/tmp/enc-attr2/spirv-pw3
export ANE_ISLAND_MODE=resident-batch
ARGS=(--source /var/tmp/EncoderParityAne/encoder-source
  --bundles /var/tmp/island-reexport/bundles
  --worker /var/tmp/mlx-main-strict/.work/mlx/build-ane-device/tools/mlx-omarchy-ane-worker/mlx-omarchy-ane-worker
  --libane /var/tmp/island-reexport/libane-strict.so
  --deadline-ms 20000
  --capture /var/tmp/EncoderParityAne/capture
  --islands ABC)
stat -c "lock inode=%i held $(date -Iseconds)" /tmp/m1-gpu.lock
for arm in before after before after; do
  mkdir -p $P/ab-$arm-out $P/ab-$arm-scratch
  echo "=== arm=$arm $(date -Iseconds) ==="
  flock -w 900 /tmp/m1-gpu.lock \
    $PY $P/${arm}_instr.py "${ARGS[@]}" --scratch $P/ab-$arm-scratch --out $P/ab-$arm-out
  sha256sum $P/ab-$arm-out/encoder_hidden.npy
done
stat -c "lock inode=%i freed $(date -Iseconds)" /tmp/m1-gpu.lock
flock -n /tmp/m1-gpu.lock true && echo "LOCK-FREE-OK"
