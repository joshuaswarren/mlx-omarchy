#!/usr/bin/env bash
set -euo pipefail
hostname
date -u +%FT%TZ
root="$HOME/src/mlx-dense-prefill-c19e1ecd"
base="$HOME/src/mlx-parity-baseline-20260908"
test ! -e "$root"
git clone --quiet --shared "$HOME/src/mlx-coalesced-gemv-01b28eb6" "$root"
git -C "$root" fetch --quiet /tmp/dense-prefill.bundle refs/heads/wave/integrate
git -C "$root" checkout --quiet --detach FETCH_HEAD
cd "$root"
test "$(git rev-parse --short=8 HEAD)" = c19e1ecd
test -z "$(git status --porcelain)"
unset MLX_OMARCHY_SOURCE_COMMIT MLX_OMARCHY_LOCAL_VERSION HK_PERF
DEV_RELEASE=1 CMAKE_BUILD_PARALLEL_LEVEL=4 scripts/build-wheel.sh > /tmp/dense-prefill-final-build.log 2>&1
wheels=(dist/mlx_omarchy-*.whl)
test "${#wheels[@]}" -eq 1
wheel="$root/${wheels[0]}"
out="$root/receipts/2026-09-08-dense-final"
mkdir -p "$out"
printf 'SOURCE=%s\n' "$(git rev-parse HEAD)"
sha256sum "$wheel"
"$base/.venv-benchmark/bin/python" -m pip freeze --exclude mlx --exclude mlx-omarchy --exclude mlx-lm > "$out/requirements-baseline.txt"
python3.14 -m venv .venv-accept
.venv-accept/bin/python -m pip install --no-deps -r "$out/requirements-baseline.txt" "$wheel" mlx-lm==0.31.3 > "$out/install.log" 2>&1
cmake -S .work/mlx -B .work/build-accept -DCMAKE_BUILD_TYPE=Release -DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_CPU=ON -DMLX_BUILD_TESTS=ON -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF > "$out/test-build.log" 2>&1
cmake --build .work/build-accept --target omarchy_matmul_family_tests omarchy_runtime_tests omarchy_primitive_tests -j 4 >> "$out/test-build.log" 2>&1
for suite in omarchy_matmul_family_tests omarchy_runtime_tests omarchy_primitive_tests; do
    .work/build-accept/tests/omarchy/"$suite" > "$out/$suite.log" 2>&1
    printf 'SUITE_PASS=%s\n' "$suite"
done
python3 /tmp/final_dense_matrix.py "$root" "$base" "$root/.venv-accept/bin/python" "$wheel"
date -u +%FT%TZ
printf 'DENSE_PREFILL_FINAL_DONE\n'
