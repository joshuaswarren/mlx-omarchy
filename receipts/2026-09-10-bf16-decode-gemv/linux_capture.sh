#!/usr/bin/env bash
set -euo pipefail
python=$1
out=$2
probe=${3:-receipt-bf16/fixed_bf16.py}
mkdir -p "$out"
for shape in "896 896" "896 128" "896 4864" "4864 896" "896 151936"; do
  read -r k n <<<"$shape"
  timeout 900 "$python" "$probe" "$out/bf16_${k}x${n}.npz" "$k" "$n" | tee -a "$out/capture.log"
done
