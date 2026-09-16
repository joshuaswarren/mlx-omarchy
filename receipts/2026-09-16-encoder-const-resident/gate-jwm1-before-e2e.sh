#!/bin/bash
# EncoderConstResident gates, part 3: BEFORE-arm E2E 6 runs, immediately
# after the after-arm battery, same boot era, for a defensible median A/B.
set -euo pipefail
P=/var/tmp/enc-const-resident
PY=/home/joshuawarren/venv-agxgen/bin/python
export PYTHONPATH=/var/tmp/E2EREV/site:/var/tmp/MelFrontendPerf/venv-cache/lib/python3.14/site-packages
unset HK_PERF HK_PERFTEST MLX_OMARCHY_GATED_BARRIERS MLX_OMARCHY_GPU_PROFILE || true
export MLX_OMARCHY_SPIRV_CACHE=$P/spirv
export ANE_ISLAND_MODE=resident-batch
BEFORE=$P/vulkan_encoder_before.py

echo "== E2E 6 runs, no flags, BEFORE runner $(date -Iseconds)"
E2E=/var/tmp/ParakeetE2EBaseline7d82/fused_e2e.py
E2EARGS=(--audio /var/tmp/ParakeetE2E/audio/fixture.flac
  --golden /var/tmp/EncoderParityAne/capture
  --model ~/.cache/mlx-omarchy/parakeet-reference/mweinbach1/parakeet-tdt-0.6b-v3-coreml/b650695c2322ee5281dff48d7345b2f3a58ff018
  --pkg /var/tmp/TdtLoopDefault/pkg
  --encoder-runner $BEFORE
  --source /var/tmp/EncoderParityAne/encoder-source
  --bundles /var/tmp/island-reexport/bundles
  --worker /var/tmp/mlx-main-strict/.work/mlx/build-ane-device/tools/mlx-omarchy-ane-worker/mlx-omarchy-ane-worker
  --libane /var/tmp/island-reexport/libane-strict.so
  --ane-reference /var/tmp/EncoderParityAne/out-ane/encoder_hidden.npy
  --deadline-ms 20000)
stat -c "lock inode=%i held $(date -Iseconds)" /tmp/m1-gpu.lock
for r in 1 2 3 4 5 6; do
  echo "--- E2E b$r $(date -Iseconds)"
  mkdir -p $P/out-bE2E-$r $P/scratch-bE2E-$r
  flock -w 900 /tmp/m1-gpu.lock \
    $PY $E2E "${E2EARGS[@]}" --scratch $P/scratch-bE2E-$r --out $P/out-bE2E-$r
done
stat -c "lock inode=%i freed $(date -Iseconds)" /tmp/m1-gpu.lock
flock -n /tmp/m1-gpu.lock true && echo "LOCK-FREE-OK"
echo "== pin check before-arm E2E"
PIN=38c73261f29230276ed76f1fc017b76b024156d79218bd5f1347fdc7e7d43ec7
FAIL=0
for f in $P/out-bE2E-*/encoder_hidden.npy; do
  h=$(sha256sum $f | cut -d' ' -f1)
  if [ "$h" != "$PIN" ]; then echo "PIN-MISMATCH $f $h"; FAIL=1; fi
done
[ $FAIL -eq 0 ] && echo ALL-PINS-EXACT
echo "== A/B summary"
$PY - <<'EOF'
import json, glob, statistics
def walls(pattern):
    out = {}
    for d in sorted(glob.glob(pattern)):
        rep = json.load(open(d + "/e2e-report.json"))
        st = {s["stage"]: s["wall_ms"] for s in rep["stages"]}
        out[d] = (st["encoder_ane"], rep["timing"]["total_pipeline_ms"])
    return out
b = walls("/var/tmp/enc-const-resident/out-bE2E-?")
a = walls("/var/tmp/enc-const-resident/out-r?")
for d, (e, t) in b.items(): print("before", d.split("-")[-1], "enc=%.1f total=%.1f" % (e, t))
for d, (e, t) in a.items(): print("after ", d.split("-")[-1], "enc=%.1f total=%.1f" % (e, t))
be = [e for e, _ in b.values()]; ae = [e for e, _ in a.values()]
print("BEFORE median r2-r6:", round(statistics.median(be[1:]), 1),
      "AFTER median r2-r6:", round(statistics.median(ae[1:]), 1))
print("BEFORE median r1-r6:", round(statistics.median(be), 1),
      "AFTER median r1-r6:", round(statistics.median(ae), 1))
bt = [t for _, t in b.values()]; at = [t for _, t in a.values()]
print("total BEFORE r2-r6:", round(statistics.median(bt[1:]), 1),
      "AFTER r2-r6:", round(statistics.median(at[1:]), 1))
EOF
echo BATTERY3-DONE