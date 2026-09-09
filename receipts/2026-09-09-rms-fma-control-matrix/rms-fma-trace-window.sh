#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/src/mlx-rms-round-screen"
restore() {
    .venv-accept/bin/python -m pip install --no-deps --force-reinstall /tmp/mlx-fma-diagnostic-wheel/mlx_omarchy-0.32.2.dev202609090935+8fd73d0-cp314-cp314-linux_aarch64.whl
}
trap restore EXIT
.venv-accept/bin/python -m pip install --no-deps --force-reinstall dist/mlx_omarchy-0.32.2.dev202609091024+fea521a-cp314-cp314-linux_aarch64.whl
.venv-accept/bin/python /tmp/mlx-q4-long-operation-oracle.py /tmp/mlx-fma-q4-long-operations
