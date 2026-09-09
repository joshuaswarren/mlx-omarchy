#!/usr/bin/env bash
set -euo pipefail
commit="$1"
root="$HOME/src/mlx-tile-load-${commit:0:8}"
base="$HOME/src/mlx-rope-drain-770ae465"
hostname
date -u +%FT%TZ
test ! -e "$root"
git clone --quiet --shared "$base" "$root"
git -C "$root" fetch --quiet /tmp/tile-load.bundle HEAD
git -C "$root" checkout --quiet --detach FETCH_HEAD
cd "$root"
test "$(git rev-parse HEAD)" = "$commit"
test -z "$(git status --porcelain)"
unset MLX_OMARCHY_SOURCE_COMMIT MLX_OMARCHY_LOCAL_VERSION HK_PERF MLX_OMARCHY_GATED_BARRIERS
out="$root/receipts/2026-09-08-tile-load-current"
mkdir -p "$out"
DEV_RELEASE=1 CMAKE_BUILD_PARALLEL_LEVEL=4 scripts/build-wheel.sh > "$out/build.log" 2>&1
wheels=(dist/mlx_omarchy-*.whl)
test "${#wheels[@]}" -eq 1
wheel="$root/${wheels[0]}"
sha256sum "$wheel"
python3.14 -m venv .venv-accept
.venv-accept/bin/python -m pip install --no-deps -r "$base/receipts/2026-09-08-dense-final/requirements-baseline.txt" "$wheel" mlx-lm==0.31.3 > "$out/install.log" 2>&1
cmake -S .work/mlx -B .work/build-accept -DCMAKE_BUILD_TYPE=Release -DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_CPU=ON -DMLX_BUILD_TESTS=ON -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF > "$out/test-build.log" 2>&1
cmake --build .work/build-accept --target omarchy_matmul_family_tests omarchy_fast_ops_tests omarchy_runtime_tests -j 4 >> "$out/test-build.log" 2>&1
for suite in omarchy_matmul_family_tests omarchy_fast_ops_tests omarchy_runtime_tests; do
    .work/build-accept/tests/omarchy/"$suite" > "$out/$suite.log" 2>&1
    printf 'SUITE_PASS=%s\n' "$suite"
done
python3 /tmp/tile-load-pairs.py "$root"
date -u +%FT%TZ
printf 'ROPE_DRAIN_WINDOW_DONE\n'
