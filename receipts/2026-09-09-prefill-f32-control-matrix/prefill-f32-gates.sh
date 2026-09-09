#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/src/mlx-rms-round-screen"
restore() {
    env -u PYTHONPATH .venv-accept/bin/python -m pip install --no-deps --force-reinstall /tmp/mlx-fma-diagnostic-wheel/mlx_omarchy-0.32.2.dev202609090935+8fd73d0-cp314-cp314-linux_aarch64.whl
}
trap restore EXIT
.venv-accept/bin/python -m pip install --no-deps --force-reinstall dist/mlx_omarchy-0.32.2.dev202609091045+53697db-cp314-cp314-linux_aarch64.whl
export VK_DRIVER_FILES=/tmp/mesa-native-trig-icd.json
export MESA_SHADER_CACHE_DISABLE=true
export PYTHONPATH=/tmp/mlx-prefill-f32-control
.venv-accept/bin/python /tmp/mlx-q4-long-operation-oracle.py /tmp/mlx-f32-prefill-q4-operations
.venv-accept/bin/python /tmp/rms-gemv-matrix.py 2026-09-09-prefill-f32-control-matrix
