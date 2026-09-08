#!/usr/bin/env bash
# ANE timed run: load lifecycle6 module, benchmark candidates, unload.
# Failure-safe: rmmod + absence verify run even if the benchmark fails.
# Designed to run under shared /tmp/m1-gpu.lock AFTER ParityBaseline release.
set -uo pipefail
echo "hostname: $(hostname)"
BASE=~/src/ane-eightcore-20260906
COMPILER=$BASE/compiler
MODPATH=$BASE/runtime-lifecycle5/ane/ane.ko
WORK=/tmp/parity-ane-candidates
LIBANE_LIB=$HOME/src/ane-eightcore-20260906/runtime-abi1/bindings/python/dylib/libane_python.so
DEST=$WORK/timed-results
mkdir -p "$DEST"

cleanup() {
  echo "== cleanup: rmmod + absence verify (runs on success AND failure)"
  if [ -e /sys/module/ane/version ]; then
    sudo rmmod ane || true
    sleep 0.5
  fi
  if [ -e /sys/module/ane/version ]; then
    echo "CLEANUP_FAIL: ane still loaded"
  else
    echo "CLEANUP_OK: ane verified absent $(date -u +%Y-%m-%dT%H:%M:%S.%6NZ)"
  fi
  ls -l /dev/accel/ 2>/dev/null || echo "no /dev/accel (expected after unload)"
}
trap cleanup EXIT

# Baseline: module absent
echo "== baseline: module absent check"
if [ -e /sys/module/ane/version ]; then echo "ABORT: ane already loaded"; exit 2; fi
ls -l /dev/accel/ 2>/dev/null || echo "no /dev/accel (expected)"

# Record module hash + version BEFORE load
echo "== module hash/vermagic/version (pre-load)"
sha256sum "$MODPATH"
modinfo "$MODPATH" | grep -E "^version|^vermagic|^depends|^filename"

# Load
echo "== insmod"
sudo insmod "$MODPATH"
sleep 0.5
echo "== post-load"
cat /sys/module/ane/version
ls -l /dev/accel/accel0

# Benchmark candidates (lifecycle6 methodology: 3 warmups, 30 iterations,
# every program prepared via pyane_init, timing = first transfer through
# last readback per iteration, exact fp16 output match every iteration)
echo "== benchmark candidates"
export LD_LIBRARY_PATH="$HOME/.local/mil-hwx-gnustep/lib:${LD_LIBRARY_PATH:-}"
cd "$COMPILER"
python3 $BASE/benchmark-packages.py "$COMPILER" "$WORK" "$LIBANE_LIB" "$DEST" b1-gemv-k896-n4864 b2-gemv-k4864-n896
BENCH_RC=$?
echo "benchmark rc=$BENCH_RC"

# Unload happens in the EXIT trap; verify line prints there.
exit $BENCH_RC
