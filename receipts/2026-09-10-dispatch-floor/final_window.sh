#!/usr/bin/env bash
# Dispatch-floor final window: rebased-driver bench, tests, digest legs.
# Run: flock -w 60 /tmp/m1-gpu.lock bash receipts/2026-09-10-dispatch-floor/final_window.sh
set -uo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
out=receipts/2026-09-10-dispatch-floor

# Keep the tip-drift evidence from the first (failed) leg run.
if [ -f "$out/legs/r1-wt-default.json" ] && [ ! -f "$out/legs/tip-r1-wt-default.json" ]; then
  for f in "$out/legs"/r1-*; do
    [ -e "$f" ] && mv "$f" "$out/legs/tip-$(basename "$f")"
  done
fi

export VK_DRIVER_FILES=/home/joshuawarren/src/mesa-wt-dispatchfloor/dispatchfloor-icd.json

# 1. Microbenchmark on the rebased (installed-arithmetic) driver.
timeout 180 /tmp/dfb > /tmp/dfb-6F-DEFAULT.out 2>&1
echo "bench DEFAULT exit=$?"
HK_PERFTEST=nocdmbarrier timeout 180 /tmp/dfb > /tmp/dfb-6F-NOCDMBARRIER.out 2>&1
echo "bench NOCDMBARRIER exit=$?"
HK_PERFTEST=usccdmbarrier timeout 180 /tmp/dfb > /tmp/dfb-6F-USCCDMBARRIER.out 2>&1
echo "bench USCCDMBARRIER exit=$?"
VK_DRIVER_FILES=/home/joshuawarren/stock-mesa/stock-icd.json timeout 180 /tmp/dfb > /tmp/dfb-6F-STOCK.out 2>&1
echo "bench STOCK exit=$?"

for f in DEFAULT NOCDMBARRIER USCCDMBARRIER STOCK; do
  cp "/tmp/dfb-6F-$f.out" "$out/bench/dfb-6f6afc8-$f.ndjson"
done
cp /tmp/dfb-NOP.out "$out/bench/dfb-tip-crashed-NOP.ndjson" 2>/dev/null || true

# 2. Runtime hazard suite under the worktree driver, all three modes.
T=/home/joshuawarren/src/mlx-DecodeBound3/.work/build-decode-bound-3/tests/omarchy/omarchy_runtime_tests
unset HK_PERFTEST
timeout 300 "$T" > /tmp/hk-tests-6F-DEFAULT.log 2>&1
echo "tests DEFAULT exit=$?"
HK_PERFTEST=nocdmbarrier timeout 300 "$T" > /tmp/hk-tests-6F-NOCDMBARRIER.log 2>&1
echo "tests NOCDMBARRIER exit=$?"
HK_PERFTEST=usccdmbarrier timeout 300 "$T" > /tmp/hk-tests-6F-USCCDMBARRIER.log 2>&1
echo "tests USCCDMBARRIER exit=$?"
for m in DEFAULT NOCDMBARRIER USCCDMBARRIER; do
  cp "/tmp/hk-tests-6F-$m.log" "$out/tests-6f6afc8-$m.log"
done

# 3. Digest legs: six canonical digests + driver A/B decode timing.
unset HK_PERFTEST
python3 "$out/run_driver_legs.py" \
  --python /home/joshuawarren/src/mlx-main-b6d662a8/.work/venv-run/bin/python \
  --wheel /home/joshuawarren/src/mlx-main-b6d662a8/dist/mlx_omarchy-0.32.2.dev202609101916+b6d662a-cp314-cp314-linux_aarch64.whl \
  --out "$out/legs"
rc=$?
echo "legs exit=$rc"
echo "FINAL_WINDOW_DONE rc=$rc"
