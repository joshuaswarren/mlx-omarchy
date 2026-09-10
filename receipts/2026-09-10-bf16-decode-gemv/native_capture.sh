#!/bin/zsh
set -euo pipefail
python=/Users/joshuawarren/src/mlx-bench-20260901/venv/bin/python
probe=/tmp/bf16-fixed-Bf16DecodeGemv.py
out=/tmp/bf16-native-Bf16DecodeGemv
rm -rf "$out"
mkdir -p "$out"
for shape in "896 896" "896 128" "896 4864" "4864 896" "896 151936"; do
  set -- $=shape
  /usr/bin/perl -e 'alarm shift; exec @ARGV' 900 "$python" "$probe" \
    "$out/bf16_${1}x${2}.npz" "$1" "$2" | tee -a "$out/capture.log"
done
