#!/usr/bin/env bash
# usage: make_venv.sh <venv-path> <python-bin> <wheel-or-none>
set -euo pipefail
VENV="$1"; PYBIN="$2"; WHEEL="${3:-none}"
if [ ! -x "$VENV/bin/python" ]; then
  "$PYBIN" -m venv "$VENV"
fi
"$VENV/bin/pip" install -q "mlx-lm==0.31.3"
if [ "$WHEEL" != none ]; then
  "$VENV/bin/pip" install -q --force-reinstall --no-deps "$WHEEL"
fi
"$VENV/bin/python" -c "import mlx.core, mlx_lm, sys; print('venv-ok', sys.version.split()[0], mlx_lm.__version__)"
