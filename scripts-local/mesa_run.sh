#!/bin/sh
# Mesa Honeykrisp repro batch on jwm1 (MesaForkStaging protocol).
set -u
ulimit -c 0
cd "$HOME/benchq/hk" || exit 1
sha256sum hk-repro-bf2.tar.gz
tar xzf hk-repro-bf2.tar.gz
cd repro || exit 1
rm -f repro
make repro 2>&1 | tail -2
echo "== GATE: device line =="
./repro 2>&1 | head -1 | tee "$HOME/benchq/logs/mesa-device.txt"
grep -q "Apple M1" "$HOME/benchq/logs/mesa-device.txt" || { echo "DEVICE MISMATCH - inconclusive"; exit 2; }
for PASS in 1 2 3; do
  ./repro > "$HOME/benchq/logs/mesa-pass$PASS.log" 2>&1
  echo "== PASS $PASS rc=$? =="
  cat "$HOME/benchq/logs/mesa-pass$PASS.log"
done
echo "== DUMP PASSES =="
AGX_MESA_DEBUG=shaders ./repro 3 > "$HOME/benchq/logs/mesa-dump-case3.log" 2>&1
AGX_MESA_DEBUG=shaders ./repro 5 > "$HOME/benchq/logs/mesa-dump-case5.log" 2>&1
AGX_MESA_DEBUG=shaders ./repro 2 > "$HOME/benchq/logs/mesa-dump-case2.log" 2>&1
AGX_MESA_DEBUG=shaders ./repro 50 > "$HOME/benchq/logs/mesa-dump-case50.log" 2>&1
AGX_MESA_DEBUG=shaders ./repro 1 > "$HOME/benchq/logs/mesa-dump-case1.log" 2>&1
AGX_MESA_DEBUG=shaders ./repro 4 > "$HOME/benchq/logs/mesa-dump-case4.log" 2>&1
echo "dump sizes:"
wc -c "$HOME/benchq/logs"/mesa-dump-*.log
vulkaninfo --summary 2>/dev/null | grep -i -E "driverName|apiVersion|deviceName" | head -4
