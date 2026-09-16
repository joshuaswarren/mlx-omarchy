#!/bin/bash
# EncoderConstResident gates on jwm1: standalone identity+timing arms,
# knob-off contrast, E2E 6 runs no flags. Lock /tmp/m1-gpu.lock:
# flock -w 900, never steal, never unlink.
set -euo pipefail
P=/var/tmp/enc-const-resident
PY=/home/joshuawarren/venv-agxgen/bin/python
export PYTHONPATH=/var/tmp/E2EREV/site:/var/tmp/MelFrontendPerf/venv-cache/lib/python3.14/site-packages
unset HK_PERF HK_PERFTEST MLX_OMARCHY_GATED_BARRIERS MLX_OMARCHY_GPU_PROFILE || true
export MLX_OMARCHY_SPIRV_CACHE=$P/spirv
export ANE_ISLAND_MODE=resident-batch
PIN=38c73261f29230276ed76f1fc017b76b024156d79218bd5f1347fdc7e7d43ec7
RUNARGS=(--source /var/tmp/EncoderParityAne/encoder-source
  --capture /var/tmp/EncoderParityAne/capture
  --bundles /var/tmp/island-reexport/bundles
  --worker /var/tmp/mlx-main-strict/.work/mlx/build-ane-device/tools/mlx-omarchy-ane-worker/mlx-omarchy-ane-worker
  --libane /var/tmp/island-reexport/libane-strict.so
  --deadline-ms 20000
  --islands ABC)
BEFORE=$P/vulkan_encoder_before.py
AFTER=$P/vulkan_encoder_after.py
mkdir -p $P
cp -f /var/tmp/enc-fold-reland/vulkan_encoder.py $BEFORE
echo "== runner identity $(date -Iseconds)"
sha256sum $BEFORE $AFTER

stat -c "lock inode=%i held $(date -Iseconds)" /tmp/m1-gpu.lock

echo "== after: cold process legs x2 $(date -Iseconds)"
for i in 1 2; do
  rm -rf $P/out-cold$i $P/sc-cold$i; mkdir -p $P/out-cold$i $P/sc-cold$i
  flock -w 900 /tmp/m1-gpu.lock \
    $PY $AFTER "${RUNARGS[@]}" --scratch $P/sc-cold$i --out $P/out-cold$i
  sha256sum $P/out-cold$i/encoder_hidden.npy
done

echo "== after: 3x warm same process (--repeat 4) $(date -Iseconds)"
rm -rf $P/out-warm $P/sc-warm; mkdir -p $P/out-warm $P/sc-warm
flock -w 900 /tmp/m1-gpu.lock \
  $PY $AFTER "${RUNARGS[@]}" --repeat 4 --scratch $P/sc-warm --out $P/out-warm

echo "== before: separate-process warm legs x3 $(date -Iseconds)"
for i in 1 2 3; do
  rm -rf $P/out-b$i $P/sc-b$i; mkdir -p $P/out-b$i $P/sc-b$i
  flock -w 900 /tmp/m1-gpu.lock \
    $PY $BEFORE "${RUNARGS[@]}" --scratch $P/sc-b$i --out $P/out-b$i
  sha256sum $P/out-b$i/encoder_hidden.npy
done

echo "== after: knob off, --repeat 2 (both passes full cost) $(date -Iseconds)"
rm -rf $P/out-knob $P/sc-knob; mkdir -p $P/out-knob $P/sc-knob
flock -w 900 /tmp/m1-gpu.lock env MLX_OMARCHY_ENCODER_CONST_CACHE=0 \
  $PY $AFTER "${RUNARGS[@]}" --repeat 2 --scratch $P/sc-knob --out $P/out-knob

echo "== before: cProfile leg (residual attribution) $(date -Iseconds)"
rm -rf $P/out-prof $P/sc-prof; mkdir -p $P/out-prof $P/sc-prof
flock -w 900 /tmp/m1-gpu.lock \
  $PY -m cProfile -s cumtime $BEFORE "${RUNARGS[@]}" \
  --scratch $P/sc-prof --out $P/out-prof 2>&1 | grep -E "ncalls|eval_const|_eval_const|def run|execute|apply|parse|read_bytes|seconds|function calls" | head -25
sha256sum $P/out-prof/encoder_hidden.npy

echo "== E2E 6 runs, no flags, after runner $(date -Iseconds)"
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
for r in 1 2 3 4 5 6; do
  echo "--- E2E r$r $(date -Iseconds)"
  rm -rf $P/out-r$r $P/scratch-r$r
  mkdir -p $P/out-r$r $P/scratch-r$r
  flock -w 900 /tmp/m1-gpu.lock \
    $PY $E2E "${E2EARGS[@]}" --scratch $P/scratch-r$r --out $P/out-r$r
done

stat -c "lock inode=%i freed $(date -Iseconds)" /tmp/m1-gpu.lock
flock -n /tmp/m1-gpu.lock true && echo "LOCK-FREE-OK"

echo "== identity summary"
sha256sum $P/out-cold*/encoder_hidden.npy $P/out-warm/pass-*/encoder_hidden.npy \
  $P/out-b*/encoder_hidden.npy $P/out-knob/pass-*/encoder_hidden.npy \
  $P/out-prof/encoder_hidden.npy $P/out-r*/encoder_hidden.npy \
  $P/out-r1/transcript.txt
echo "== pin check"
for f in $P/out-cold*/encoder_hidden.npy $P/out-warm/pass-*/encoder_hidden.npy \
         $P/out-b*/encoder_hidden.npy $P/out-knob/pass-*/encoder_hidden.npy \
         $P/out-prof/encoder_hidden.npy $P/out-r*/encoder_hidden.npy; do
  h=$(sha256sum $f | cut -d' ' -f1)
  [ "$h" = "$PIN" ] || echo "PIN-MISMATCH $f $h"
done
echo PIN-CHECK-DONE
echo "== E2E stage walls"
$PY - <<'EOF'
import json, glob, statistics
walls = {}
for d in sorted(glob.glob("/var/tmp/enc-const-resident/out-r?")):
    rep = json.load(open(d + "/e2e-report.json"))
    st = {s["stage"]: s["wall_ms"] for s in rep["stages"]}
    L = rep["layers"]
    ex = rep["execution"]
    seq = L["layer_6_decoder_sequence"]
    enc = L["layer_5_encoder"]
    walls[d] = st
    print(d, "status=", rep.get("status"),
          "emissions=", seq["actual_emissions"], "/", seq["native_emissions"],
          "prefix=", seq["matching_prefix_length"],
          "mel_bit_exact=", L["layer_2_preprocessing"]["mel"]["bit_exact"],
          "bounds=", enc["all_bounds_pass"],
          "control=", ex["control"], "fallback=", ex["tdt_fallback_reason"],
          "cte=", ex["cpu_tensor_events"],
          "ane_subm=", rep["ane"]["submissions"], "ane_to=", rep["ane"]["timeouts"],
          "enc=", st.get("encoder_ane"), "total=", rep["timing"]["total_pipeline_ms"])
enc = [w["encoder_ane"] for w in walls.values()]
if len(enc) == 6:
    print("encoder_ane median r2-r6:", round(statistics.median(enc[1:]), 3),
          "r1-r6:", round(statistics.median(enc), 3))
EOF
echo BATTERY-DONE