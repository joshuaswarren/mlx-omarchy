#!/bin/bash
# EncoderConstResident, part 5: default-path (cache opt-in, default OFF)
# E2E re-verify, 3 runs no flags, after the default flip. Pin + transcript
# + 104/104 per run.
set -euo pipefail
P=/var/tmp/enc-const-resident
PY=/home/joshuawarren/venv-agxgen/bin/python
export PYTHONPATH=/var/tmp/E2EREV/site:/var/tmp/MelFrontendPerf/venv-cache/lib/python3.14/site-packages
unset HK_PERF HK_PERFTEST MLX_OMARCHY_GATED_BARRIERS MLX_OMARCHY_GPU_PROFILE \
  MLX_OMARCHY_ENCODER_CONST_CACHE || true
export MLX_OMARCHY_SPIRV_CACHE=$P/spirv
export ANE_ISLAND_MODE=resident-batch
PIN=38c73261f29230276ed76f1fc017b76b024156d79218bd5f1347fdc7e7d43ec7
TRANSCRIPT=db501a8c080380ea027ffa50a4b4956c39df77cb692c4fb78e556311a11a0790
DEFAULT=$P/vulkan_encoder_default.py
echo "== default runner $(date -Iseconds)"
sha256sum $DEFAULT

echo "== E2E 3 runs, no flags, default runner (cache off by default) $(date -Iseconds)"
E2E=/var/tmp/ParakeetE2EBaseline7d82/fused_e2e.py
E2EARGS=(--audio /var/tmp/ParakeetE2E/audio/fixture.flac
  --golden /var/tmp/EncoderParityAne/capture
  --model ~/.cache/mlx-omarchy/parakeet-reference/mweinbach1/parakeet-tdt-0.6b-v3-coreml/b650695c2322ee5281dff48d7345b2f3a58ff018
  --pkg /var/tmp/TdtLoopDefault/pkg
  --encoder-runner $DEFAULT
  --source /var/tmp/EncoderParityAne/encoder-source
  --bundles /var/tmp/island-reexport/bundles
  --worker /var/tmp/mlx-main-strict/.work/mlx/build-ane-device/tools/mlx-omarchy-ane-worker/mlx-omarchy-ane-worker
  --libane /var/tmp/island-reexport/libane-strict.so
  --ane-reference /var/tmp/EncoderParityAne/out-ane/encoder_hidden.npy
  --deadline-ms 20000)
stat -c "lock inode=%i held $(date -Iseconds)" /tmp/m1-gpu.lock
for r in 1 2 3; do
  echo "--- E2E d$r $(date -Iseconds)"
  mkdir -p $P/out-dE2E-$r $P/scratch-dE2E-$r
  flock -w 900 /tmp/m1-gpu.lock \
    $PY $E2E "${E2EARGS[@]}" --scratch $P/scratch-dE2E-$r --out $P/out-dE2E-$r
done
stat -c "lock inode=%i freed $(date -Iseconds)" /tmp/m1-gpu.lock
flock -n /tmp/m1-gpu.lock true && echo "LOCK-FREE-OK"

echo "== identity: pin + transcript"
FAIL=0
for f in $P/out-dE2E-*/encoder_hidden.npy; do
  h=$(sha256sum $f | cut -d' ' -f1)
  if [ "$h" != "$PIN" ]; then echo "PIN-MISMATCH $f $h"; FAIL=1; fi
done
[ $FAIL -eq 0 ] && echo ALL-PINS-EXACT
for f in $P/out-dE2E-*/transcript.txt; do
  h=$(sha256sum $f | cut -d' ' -f1)
  if [ "$h" != "$TRANSCRIPT" ]; then echo "TRANSCRIPT-MISMATCH $f $h"; FAIL=1; fi
done
[ $FAIL -eq 0 ] && echo ALL-TRANSCRIPTS-EXACT

echo "== summary"
$PY - <<'EOF'
import json, glob, statistics
encs = []
for d in sorted(glob.glob("/var/tmp/enc-const-resident/out-dE2E-?")):
    rep = json.load(open(d + "/e2e-report.json"))
    st = {s["stage"]: s["wall_ms"] for s in rep["stages"]}
    ex = rep["execution"]
    seq = rep["layers"]["layer_6_decoder_sequence"]
    print(d.split("/")[-1], "status=", rep["status"],
          "emissions=", seq["actual_emissions"], "/", seq["native_emissions"],
          "prefix=", seq["matching_prefix_length"],
          "mel=", rep["layers"]["layer_2_preprocessing"]["mel"]["bit_exact"],
          "bounds=", rep["layers"]["layer_5_encoder"]["all_bounds_pass"],
          "cte=", ex["cpu_tensor_events"], "fb=", ex["tdt_fallback_reason"],
          "enc=%.1f" % st["encoder_ane"],
          "ane_to=", rep["ane"]["timeouts"])
    encs.append(st["encoder_ane"])
print("default-path encoder_ane median:", round(statistics.median(encs), 1))
EOF
echo BATTERY5-DONE