#!/usr/bin/env bash
# M1 acceptance window for the Q4 GEMV order change. Run under
# `flock -w 600 /tmp/m1-gpu.lock timeout 7200` from the checkout root
# after scripts/build-wheel.sh has produced exactly one wheel in dist/.
set -euo pipefail
cd "$(dirname "$0")/../.."
expected="$1"
test "$(git rev-parse --short=7 HEAD)" = "$expected"
here=receipts/2026-09-09-q4-gemv-order
wheels=(dist/mlx_omarchy-*.whl)
test "${#wheels[@]}" -eq 1
case "${wheels[0]}" in *"+$expected"*) ;; *) echo "wheel ${wheels[0]} not stamped +$expected" >&2; exit 1;; esac
sha256sum "${wheels[0]}" | tee "$here/wheel.sha256"
unset HK_PERF HK_PERFTEST MLX_OMARCHY_SOURCE_COMMIT MLX_OMARCHY_LOCAL_VERSION MLX_OMARCHY_GPU_PROFILE PYTHONPATH VK_DRIVER_FILES
export MESA_SHADER_CACHE_DISABLE=true
.venv-accept/bin/python -m pip install --no-deps --force-reinstall "${wheels[0]}" > "$here/install.log" 2>&1
rm -rf /tmp/pgp-fixed-projection
.venv-accept/bin/python "$here/fixed-projection.py" /tmp/mlx-native-q4-long-operations /tmp/pgp-fixed-projection scripts 2>&1 | tee "$here/fixed-projection.log"
cp /tmp/pgp-fixed-projection/results.json "$here/fixed-projection-results.json"
cp /tmp/pgp-fixed-projection/provenance.json "$here/fixed-projection-provenance.json"
.venv-accept/bin/python "$here/run-legs.py" . .venv-accept/bin/python "${wheels[0]}" "$here/legs" 2>&1 | tee "$here/run-legs.log"
