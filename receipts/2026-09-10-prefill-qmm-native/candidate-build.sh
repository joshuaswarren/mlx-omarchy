#!/usr/bin/env bash
set -euo pipefail

root="$HOME/src/mlx-prefill-qmm-native"
out="$root/receipts/2026-09-10-prefill-qmm-native/candidate-v3"
commit="ae299304960ece75d1cd84d4eefd722fc4d2a525"
cd "$root"
git fetch --quiet https://github.com/joshuaswarren/mlx-omarchy.git wave/prefill-qmm-native
git checkout --quiet --detach FETCH_HEAD
test "$(git rev-parse HEAD)" = "$commit"
mkdir -p "$out"
CMAKE_BUILD_PARALLEL_LEVEL=4 scripts/build-wheel.sh --diagnostics 2>&1 | tee "$out/build.log"
wheels=(dist/mlx_omarchy-*.whl)
test "${#wheels[@]}" -eq 1
wheel="${wheels[0]}"
sha256sum "$wheel" | tee "$out/wheel.sha256"
rm -rf .venv-accept
python3.14 -m venv .venv-accept
.venv-accept/bin/python -m pip install --no-deps \
  -r receipts/2026-09-10-prefill-qmm-native/baseline/requirements.txt "$wheel" \
  > "$out/install.log" 2>&1
.venv-accept/bin/python - <<'PY' | tee "$out/identity.log"
import mlx.core as mx
print(f"mlx_version={mx.__version__}")
print(f"cooperative_matrix_f32_8={mx.device_info()['cooperative_matrix_f32_8']}")
print(mx.device_info())
PY
