#!/usr/bin/env bash
cd ~/src/mlx-omarchy-wvs
mkdir -p receipts/2026-09-11-wrong-value-sweep/py
exec 9>/tmp/m1-gpu.lock
flock 9 || exit 4
for f in test_conv test_nn; do
  MLX_ENABLE_TF32=0 ./.work/venv-branch/bin/python -m pytest \
    ~/src/mlx-omarchy-requal-20260911/.work/mlx/python/tests/$f.py \
    -q --no-header -p no:cacheprovider \
    --junitxml=receipts/2026-09-11-wrong-value-sweep/py/$f-narrow.xml \
    > receipts/2026-09-11-wrong-value-sweep/py/$f-narrow.log 2>&1
  tail -1 receipts/2026-09-11-wrong-value-sweep/py/$f-narrow.log
done
