#!/bin/sh
# C++ batteries on real hardware against the ff4b05a prepared tree.
# Targets: compiled_tape, runtime, eq_math.
set -eu
cd ~/src/mlx-omarchy-bqm1-build/.work/mlx
grep -n "add_executable" tests/omarchy/CMakeLists.txt | grep -E "compiled_tape|runtime|eq_math"
cmake -S . -B build -DMLX_BUILD_TESTS=ON -DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_CPU=ON -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF > ~/benchq/logs/battery-configure.log 2>&1
tail -3 ~/benchq/logs/battery-configure.log
cmake --build build --target omarchy_compiled_tape_tests omarchy_runtime_tests omarchy_eq_math_tests -j"$(nproc)" 2>&1 | tail -5
echo "BUILD_DONE $(date '+%F %T')"
for T in omarchy_compiled_tape_tests omarchy_runtime_tests omarchy_eq_math_tests; do
  LOG=~/benchq/logs/battery-$T.log
  if ./build/tests/omarchy/$T > "$LOG" 2>&1; then RC=0; else RC=$?; fi
  echo "== $T rc=$RC =="
  tail -6 "$LOG"
done
