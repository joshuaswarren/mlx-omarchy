#!/bin/bash
# Gate: jw16 both-host identity battery for the fold reland.
# lincheck first (service up), then pause llm-inference.service (NOT a
# router leg; Main-cleared window), standalone encoder leg + 6-run
# no-flags E2E under flock -w 900 (never steal), then ALWAYS restart
# llm-inference and confirm active. Window target: well under 30 min.
set -uo pipefail
P=/var/tmp/enc-fold-reland
PY=/home/joshuawarren/venv-agxgen/bin/python
export PYTHONPATH=/var/tmp/E2EREV/site:/var/tmp/MelFrontendPerf/venv-cache/lib/python3.14/site-packages
unset HK_PERF HK_PERFTEST MLX_OMARCHY_GATED_BARRIERS MLX_OMARCHY_GPU_PROFILE || true
export MLX_OMARCHY_SPIRV_CACHE=$P/spirv
export ANE_ISLAND_MODE=resident-batch
RUNNER=$P/vulkan_encoder.py
E2E=/var/tmp/ParakeetE2EJw16/fused_e2e.py

echo "=== pausing llm-inference $(date -Iseconds) ==="
sudo -n systemctl stop llm-inference.service || { echo SUDO-STOP-FAIL; exit 9; }
for i in $(seq 1 30); do
  flock -n /tmp/m1-gpu.lock true && break
  sleep 2
done
flock -n /tmp/m1-gpu.lock true && echo LOCK-FREE || echo LOCK-STILL-BUSY
START=$(date -Iseconds)
fail=0

echo "=== lincheck (window open) ==="
flock -w 900 /tmp/m1-gpu.lock \
  $PY $P/lincheck_fold.py $RUNNER 2>/dev/null | tail -1

echo "=== standalone encoder leg ==="
mkdir -p $P/enc-out $P/enc-scratch
flock -w 900 /tmp/m1-gpu.lock \
  $PY $RUNNER --source /var/tmp/EncoderParityAne/encoder-source \
    --bundles /var/tmp/island-reexport/bundles \
    --worker /var/tmp/E2E296-battery/worker/mlx-omarchy-ane-worker \
    --libane /var/tmp/island-reexport/libane-strict.so \
    --deadline-ms 20000 --capture /var/tmp/EncoderParityAne/capture \
    --islands ABC --scratch $P/enc-scratch --out $P/enc-out \
    2>&1 | grep -E "ane mode" || fail=1
sha256sum $P/enc-out/encoder_hidden.npy

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
for r in 1 2 3 4 5 6; do
  echo "--- E2E r$r $(date -Iseconds) ---"
  mkdir -p $P/battery/out-$r $P/battery/scratch-$r
  flock -w 900 /tmp/m1-gpu.lock \
    $PY $E2E "${E2EARGS[@]}" --scratch $P/battery/scratch-$r \
      --out $P/battery/out-$r > $P/battery/e2e-r$r.log 2>&1 || fail=1
  grep -c . $P/battery/e2e-r$r.log >/dev/null
done
END=$(date -Iseconds)
echo "=== window $START -> $END fail=$fail ==="

echo "=== restarting llm-inference ==="
sudo -n systemctl start llm-inference.service
sleep 3
systemctl is-active llm-inference.service

echo "=== identity ==="
sha256sum $P/battery/out-*/encoder_hidden.npy $P/battery/out-1/transcript.txt
exit $fail
