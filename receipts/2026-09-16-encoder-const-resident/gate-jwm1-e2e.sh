#!/bin/bash
# EncoderConstResident gates, part 2: E2E 6 runs no flags (part 1 died at
# the profiler leg's head-pipe SIGPIPE; all part-1 arms completed).
set -euo pipefail
P=/var/tmp/enc-const-resident
PY=/home/joshuawarren/venv-agxgen/bin/python
export PYTHONPATH=/var/tmp/E2EREV/site:/var/tmp/MelFrontendPerf/venv-cache/lib/python3.14/site-packages
unset HK_PERF HK_PERFTEST MLX_OMARCHY_GATED_BARRIERS MLX_OMARCHY_GPU_PROFILE || true
export MLX_OMARCHY_SPIRV_CACHE=$P/spirv
export ANE_ISLAND_MODE=resident-batch
PIN=38c73261f29230276ed76f1fc017b76b024156d79218bd5f1347fdc7e7d43ec7
AFTER=$P/vulkan_encoder_after.py

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
stat -c "lock inode=%i held $(date -Iseconds)" /tmp/m1-gpu.lock
for r in 1 2 3 4 5 6; do
  echo "--- E2E r$r $(date -Iseconds)"
  mkdir -p $P/out-r$r $P/scratch-r$r
  flock -w 900 /tmp/m1-gpu.lock \
    $PY $E2E "${E2EARGS[@]}" --scratch $P/scratch-r$r --out $P/out-r$r
done
stat -c "lock inode=%i freed $(date -Iseconds)" /tmp/m1-gpu.lock
flock -n /tmp/m1-gpu.lock true && echo "LOCK-FREE-OK"

echo "== before-arm standalone walls (out-b1..3)"
$PY - <<'EOF'
import json
for i in (1, 2, 3):
    rep = json.load(open(f"/var/tmp/enc-const-resident/out-b{i}/run-report.json"))
    print(f"before leg b{i}: wall_ms=", round(rep["wall_ns"]/1e6, 1),
          "hidden_sha=", rep.get("passes", [{}])[0].get("encoder_hidden_sha256", "n/a"))
EOF

echo "== identity summary"
sha256sum $P/out-cold1/encoder_hidden.npy $P/out-cold2/encoder_hidden.npy \
  $P/out-warm/pass-1/encoder_hidden.npy $P/out-warm/pass-2/encoder_hidden.npy \
  $P/out-warm/pass-3/encoder_hidden.npy $P/out-warm/pass-4/encoder_hidden.npy \
  $P/out-b1/encoder_hidden.npy $P/out-b2/encoder_hidden.npy $P/out-b3/encoder_hidden.npy \
  $P/out-knob/pass-1/encoder_hidden.npy $P/out-knob/pass-2/encoder_hidden.npy \
  $P/out-prof/encoder_hidden.npy $P/out-r1/encoder_hidden.npy $P/out-r2/encoder_hidden.npy \
  $P/out-r3/encoder_hidden.npy $P/out-r4/encoder_hidden.npy $P/out-r5/encoder_hidden.npy \
  $P/out-r6/encoder_hidden.npy $P/out-r1/transcript.txt
echo "== pin check"
FAIL=0
for f in $P/out-cold1/encoder_hidden.npy $P/out-cold2/encoder_hidden.npy \
         $P/out-warm/pass-1/encoder_hidden.npy $P/out-warm/pass-2/encoder_hidden.npy \
         $P/out-warm/pass-3/encoder_hidden.npy $P/out-warm/pass-4/encoder_hidden.npy \
         $P/out-b1/encoder_hidden.npy $P/out-b2/encoder_hidden.npy $P/out-b3/encoder_hidden.npy \
         $P/out-knob/pass-1/encoder_hidden.npy $P/out-knob/pass-2/encoder_hidden.npy \
         $P/out-prof/encoder_hidden.npy $P/out-r1/encoder_hidden.npy $P/out-r2/encoder_hidden.npy \
         $P/out-r3/encoder_hidden.npy $P/out-r4/encoder_hidden.npy $P/out-r5/encoder_hidden.npy \
         $P/out-r6/encoder_hidden.npy; do
  h=$(sha256sum $f | cut -d' ' -f1)
  if [ "$h" != "$PIN" ]; then echo "PIN-MISMATCH $f $h"; FAIL=1; fi
done
[ $FAIL -eq 0 ] && echo ALL-PINS-EXACT

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
echo BATTERY2-DONE