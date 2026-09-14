#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/joshuawarren/src/mlx-omarchy-encoder-conv-f32
WHEEL=$(ls -1 "$ROOT"/dist/mlx_omarchy-*.whl)
SITE=/tmp/conv-f32-site
rm -rf "$SITE"
mkdir -p "$SITE"
/home/joshuawarren/venv-agxgen/bin/python -m pip install --no-deps --force-reinstall --target "$SITE" "$WHEEL"
export PYTHONPATH="$SITE"
PY=/home/joshuawarren/venv-agxgen/bin/python
"$PY" - <<'PY'
import mlx.core as mx
print("mlx_file", mx.__file__)
print("mlx_ver", getattr(mx, "__version__", "?"))
PY
echo "--- microbench ---"
timeout -k 10s 180s flock -w 120 /tmp/m1-gpu.lock \
  "$PY" "$ROOT/receipts/2026-09-14-encoder-conv-f32/bench_conv.py"
echo "flock_after=$(flock -n /tmp/m1-gpu.lock -c true; echo $?)"
