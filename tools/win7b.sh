#!/bin/bash
# Jw16DecodeGap2 window 7b: multi-kernel rows8 class bitcheck (word kernel excluded - its ROWS>1 build
# is structurally broken: single main covers 4 of 32 columns).
set -u
OUT=/var/tmp/dg/w7; mkdir -p $OUT
PROD=/var/tmp/v072-venv-fused/bin/python3
CAND=/var/tmp/dg-venv-np/bin/python3
{ date -u; cat /proc/sys/kernel/random/boot_id; } > $OUT/env-7b.txt 2>&1
echo "--- multi rows8 micro bitcheck $(date -u +%FT%TZ)"
$PROD /var/tmp/dg/np_bitcheck.py $OUT/multiprod.json > $OUT/multiprod.log 2>&1
MLX_OMARCHY_GEMV_ROWS8=1 $CAND /var/tmp/dg/np_bitcheck.py $OUT/multion.json > $OUT/multion.log 2>&1
( cd $OUT && sha256sum ./multi*.json > SHA256SUMS-7b )
echo WIN7B-DONE $(date -u +%FT%TZ)
