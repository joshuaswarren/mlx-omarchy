#!/bin/sh
# Byte-compare the installed mlx trees of the three benchmark venvs.
set -eu
cd ~/venv-bqm1-0606/lib/python3.14/site-packages
find mlx -type f -exec sha256sum {} \; | sort -k2 > ~/benchq/pkg-0606.sha
cd ~/venv-bqm1-0928/lib/python3.14/site-packages
find mlx -type f -exec sha256sum {} \; | sort -k2 > ~/benchq/pkg-0928.sha
cd ~/venv-bqm1-0512/lib/python3.14/site-packages
find mlx -type f -exec sha256sum {} \; | sort -k2 > ~/benchq/pkg-0512.sha
echo "--- 0606 vs 0928 ---"
if diff ~/benchq/pkg-0606.sha ~/benchq/pkg-0928.sha > /dev/null; then
  echo IDENTICAL
else
  diff ~/benchq/pkg-0606.sha ~/benchq/pkg-0928.sha | head -12
fi
echo "--- 0606 vs 0512 ---"
if diff ~/benchq/pkg-0606.sha ~/benchq/pkg-0512.sha > /dev/null; then
  echo IDENTICAL
else
  diff ~/benchq/pkg-0606.sha ~/benchq/pkg-0512.sha | head -12
fi
echo "--- file counts ---"
wc -l ~/benchq/pkg-0606.sha ~/benchq/pkg-0928.sha ~/benchq/pkg-0512.sha
