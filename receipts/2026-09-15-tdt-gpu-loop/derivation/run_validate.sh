#!/bin/bash
set -e
exec 9>/tmp/m1-gpu.lock
flock -x -w 3600 9
cd /var/tmp/TdtGpuLoop
PYTHONPATH=/var/tmp/TdtGpuLoop/pkg:/var/tmp/MelFrontendPerf/venv-cache/lib/python3.14/site-packages ~/venv-agxgen/bin/python validate_loop.py /var/tmp/TdtGpuLoop/pkg $HOME/.cache/mlx-omarchy/parakeet-reference/mweinbach1/parakeet-tdt-0.6b-v3-coreml/b650695c2322ee5281dff48d7345b2f3a58ff018 24 5
