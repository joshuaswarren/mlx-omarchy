#!/bin/sh
# Tape battery rerun at 045ef20: with override expect 10/10 (347),
# without override expect 2 passed + 8 skipped.
set -eu
cd ~/src/mlx-omarchy-bqm1-build
git fetch --quiet origin 045ef20 2>/dev/null || git fetch --quiet origin main
git checkout --quiet --detach origin/main
echo "testing commit $(git rev-parse --short HEAD)"
sh scripts/prepare-mlx.sh > /dev/null 2>&1
cmake -S .work/mlx -B .work/mlx/build -DMLX_BUILD_TESTS=ON -DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_CPU=ON -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF > /dev/null 2>&1
cmake --build .work/mlx/build --target omarchy_compiled_tape_tests -j"$(nproc)" 2>&1 | tail -1
if MLX_OMARCHY_ALLOW_UNSAFE_COMPILE=1 .work/mlx/build/tests/omarchy/omarchy_compiled_tape_tests \
    > "$HOME/benchq/logs/tapetests-045ef20-override.log" 2>&1; then
  echo "WITH override rc=0"
else
  echo "WITH override rc=$?"
fi
tail -4 "$HOME/benchq/logs/tapetests-045ef20-override.log"
if .work/mlx/build/tests/omarchy/omarchy_compiled_tape_tests \
    > "$HOME/benchq/logs/tapetests-045ef20-nooverride.log" 2>&1; then
  echo "WITHOUT override rc=0"
else
  echo "WITHOUT override rc=$?"
fi
tail -4 "$HOME/benchq/logs/tapetests-045ef20-nooverride.log"
