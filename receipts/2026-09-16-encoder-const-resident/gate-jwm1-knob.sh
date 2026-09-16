#!/bin/bash
# EncoderConstResident gates, part 4: knob-off E2E legs (after runner, cache
# disabled -> must match before-arm walls) + a literal --repeat 6 leak leg.
set -euo pipefail
P=/var/tmp/enc-const-resident
PY=/home/joshuawarren/venv-agxgen/bin/python
export PYTHONPATH=/var/tmp/E2EREV/site:/var/tmp/MelFrontendPerf/venv-cache/lib/python3.14/site-packages
unset HK_PERF HK_PERFTEST MLX_OMARCHY_GATED_BARRIERS MLX_OMARCHY_GPU_PROFILE || true
export MLX_OMARCHY_SPIRV_CACHE=$P/spirv
export ANE_ISLAND_MODE=resident-batch
AFTER=$P/vulkan_encoder_after.py
E2E=/var/tmp/ParakeetE2EBaseline7d82/fused_e2e.py
E2EARGS=(--audio /var/tmp/ParakeetE2E/audio/fixture.flac
  --golden /var/tmp/EncoderParityAne/capture
  --model ~/.cache/mlx-omarchy/parakeet-reference/mweinbach1/parakeet-tdt-0.6b-v3-coreml/b650695c2322ee5281dff48d7345b2f3a58ff018
  --pkg /var/tmp/TdtLoopDefault/pkg
  --encoder-runner $AFTER
  --source /var/tmp/EncoderParityAne/encoder-source
  --bundles /var/tmp/island-reexport/bundles
  --worker /var/tmp/mlx-main-strict/.work/mlx/build-ane-device/tools/mlx-omarchy-ane-worker/mlx-omarchy-ane-worker
  --libane /var/tmp/island-reexport/libane-strict.so
  --ane-reference /var/tmp/EncoderParityAne/out-ane/encoder_hidden.npy
  --deadline-ms 20000)
RUNARGS=(--source /var/tmp/EncoderParityAne/encoder-source
  --capture /var/tmp/EncoderParityAne/capture
  --bundles /var/tmp/island-reexport/bundles
  --worker /var/tmp/mlx-main-strict/.work/mlx/build-ane-device/tools/mlx-omarchy-ane-worker/mlx-omarchy-ane-worker
  --libane /var/tmp/island-reexport/libane-strict.so
  --deadline-ms 20000
  --islands ABC)
stat -c "lock inode=%i held $(date -Iseconds)" /tmp/m1-gpu.lock

echo "== knob-off E2E legs x3 $(date -Iseconds)"
for r in 1 2 3; do
  echo "--- E2E k$r $(date -Iseconds)"
  mkdir -p $P/out-kE2E-$r $P/scratch-kE2E-$r
  flock -w 900 /tmp/m1-gpu.lock env MLX_OMARCHY_ENCODER_CONST_CACHE=0 \
    $PY $E2E "${E2EARGS[@]}" --scratch $P/scratch-kE2E-$r --out $P/out-kE2E-$r
done

echo "== repeat-6 leak leg (cache on) $(date -Iseconds)"
mkdir -p $P/out-rep6 $P/sc-rep6
flock -w 900 /tmp/m1-gpu.lock \
  $PY $AFTER "${RUNARGS[@]}" --repeat 6 --scratch $P/sc-rep6 --out $P/out-rep6

stat -c "lock inode=%i freed $(date -Iseconds)" /tmp/m1-gpu.lock
flock -n /tmp/m1-gpu.lock true && echo "LOCK-FREE-OK"

echo "== knob-off pin check"
PIN=38c73261f29230276ed76f1fc017b76b024156d79218bd5f1347fdc7e7d43ec7
FAIL=0
for f in $P/out-kE2E-*/encoder_hidden.npy $P/out-rep6/pass-*/encoder_hidden.npy; do
  h=$(sha256sum $f | cut -d' ' -f1)
  if [ "$h" != "$PIN" ]; then echo "PIN-MISMATCH $f $h"; FAIL=1; fi
done
[ $FAIL -eq 0 ] && echo ALL-PINS-EXACT

echo "== summary"
$PY - <<'EOF'
import json, glob, statistics
print("knob-off E2E legs:")
encs = []
for d in sorted(glob.glob("/var/tmp/enc-const-resident/out-kE2E-?")):
    rep = json.load(open(d + "/e2e-report.json"))
    st = {s["stage"]: s["wall_ms"] for s in rep["stages"]}
    ex = rep["execution"]
    print(d.split("/")[-1], "status=", rep["status"], "enc=%.1f" % st["encoder_ane"],
          "prefix=", rep["layers"]["layer_6_decoder_sequence"]["matching_prefix_length"],
          "bounds=", rep["layers"]["layer_5_encoder"]["all_bounds_pass"],
          "cte=", ex["cpu_tensor_events"])
    encs.append(st["encoder_ane"])
print("knob-off median:", round(statistics.median(encs), 1))
rep = json.load(open("/var/tmp/enc-const-resident/out-rep6/run-report.json"))
print("repeat-6 passes:")
for p in rep["passes"]:
    print("pass", p["pass"], "wall_ms=%.1f" % (p["wall_ns"]/1e6),
          "active=%.1fMB peak=%.1fMB rss=%.1fMB" % (p["active_memory_mb"], p["peak_memory_mb"], p["rss_mb"]),
          "hidden=", p["encoder_hidden_sha256"][:16])
EOF
echo BATTERY4-DONE