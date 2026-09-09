#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/src/mlx-rms-round-screen"
git add overlay/mlx/backend/omarchy/shaders/fast_norm.comp overlay/tests/omarchy/test_fast_ops.cpp
git -c user.name='Numerical experiment' -c user.email='experiment@localhost' commit -m 'fix: preserve RMSNorm intermediate rounding on Honeykrisp'
rm dist/mlx_omarchy-*.whl
DEV_RELEASE=1 CMAKE_BUILD_PARALLEL_LEVEL=4 scripts/build-wheel.sh > /tmp/mlx-rms-final-build.log 2>&1
wheels=(dist/mlx_omarchy-*.whl)
test "${#wheels[@]}" -eq 1
sha256sum "${wheels[0]}"
.venv-accept/bin/python -m pip install --no-deps --force-reinstall "${wheels[0]}" > /tmp/mlx-rms-final-install.log 2>&1
.venv-accept/bin/python /tmp/rms-final-matrix.py
