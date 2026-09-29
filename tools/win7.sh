#!/bin/bash
# Jw16DecodeGap2 window 7: rows8 GEMV A/B — gemv_shapes (serving / cand gate-off / cand gate-on),
# dg bitcheck gate-on vs prod, d64+d256 cells n=5 (prod vs cand-on).
set -u
OUT=/var/tmp/dg/w7; mkdir -p $OUT
MODEL=$(echo ~/.cache/huggingface/hub/models--SiddhJagani--Qwen3.8-2B-mlx-4Bit/snapshots/0867d98bfb174b042d88461c0e*)
PROD=/var/tmp/v072-venv-fused/bin/python3
CAND=/var/tmp/dg-venv-np/bin/python3
B() { py=$1; shift; $py ~/bench-scripts/qwen38-mlx-bench.py --model $MODEL/ --prompts ~/bench-scripts/qwen38-2b-prompts.jsonl --limit 1 "$@"; }
{ date -u; uname -r; cat /proc/sys/kernel/random/boot_id; cat /proc/loadavg; cat /sys/class/thermal/thermal_zone*/temp | tr '\n' ' '; echo; for p in $PROD $CAND; do $p -m pip list 2>/dev/null | grep -i mlx; done; } > $OUT/env.txt 2>&1

echo "--- gemv shapes $(date -u +%FT%TZ)"
$PROD /var/tmp/appbar/gemv_shapes.py > $OUT/shapes-prod.json 2> $OUT/shapes-prod.log
$CAND /var/tmp/appbar/gemv_shapes.py > $OUT/shapes-off.json 2> $OUT/shapes-off.log
MLX_OMARCHY_GEMV_ROWS8=1 $CAND /var/tmp/appbar/gemv_shapes.py > $OUT/shapes-on.json 2> $OUT/shapes-on.log

echo "--- dg bitcheck $(date -u +%FT%TZ)"
$PROD /var/tmp/dg/dg_bitcheck.py $OUT/bits-prod.json > $OUT/bits-prod.log 2>&1
MLX_OMARCHY_GEMV_ROWS8=1 $CAND /var/tmp/dg/dg_bitcheck.py $OUT/bits-on.json > $OUT/bits-on.log 2>&1

for c in 64 256; do
  for arm in prod on; do
    echo "--- d$c $arm $(date -u +%FT%TZ)"
    if [ "$arm" = prod ]; then
      B $PROD --new-tokens $c --warmup 1 --passes 5 --prefill-tokens 512 --label dg-$arm-d$c-n5 --out $OUT/$arm-d$c.json > $OUT/$arm-d$c.log 2>&1
    else
      export MLX_OMARCHY_GEMV_ROWS8=1
      B $CAND --new-tokens $c --warmup 1 --passes 5 --prefill-tokens 512 --label dg-$arm-d$c-n5 --out $OUT/$arm-d$c.json > $OUT/$arm-d$c.log 2>&1
      unset MLX_OMARCHY_GEMV_ROWS8
    fi
  done
done
( cd $OUT && sha256sum ./*.json ./*.txt > SHA256SUMS )
echo WIN7-DONE $(date -u +%FT%TZ)
