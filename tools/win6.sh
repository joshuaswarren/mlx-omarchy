#!/bin/bash
# Jw16DecodeGap2 window 6: RMSNorm-into-q4-GEMV-prologue fold — dg bitcheck + group micro bitcheck vs serving,
# NIR dumps, paired decode cells n=5 (serving vs candidate), kill-switch arm (MLX_OMARCHY_FUSED_GEMV_NORM=0) d64.
set -u
OUT=/var/tmp/dg/w6; mkdir -p $OUT
MODEL=$(echo ~/.cache/huggingface/hub/models--SiddhJagani--Qwen3.8-2B-mlx-4Bit/snapshots/0867d98bfb174b042d88461c0e*)
PROD=/var/tmp/v072-venv-fused/bin/python3
CAND=/var/tmp/dg-venv-np/bin/python3
B() { py=$1; shift; $py ~/bench-scripts/qwen38-mlx-bench.py --model $MODEL/ --prompts ~/bench-scripts/qwen38-2b-prompts.jsonl --limit 1 "$@"; }
NORMPROBE() { $1 -c "import mlx.core as mx
x=mx.random.normal((1,2048)).astype(mx.bfloat16)
w=mx.random.normal((2048,)).astype(mx.bfloat16)
y=mx.fast.rms_norm(x,w,1e-6); mx.eval(y)"; }
{ date -u; uname -r; cat /proc/sys/kernel/random/boot_id; cat /proc/loadavg; cat /sys/class/thermal/thermal_zone*/temp | tr '\n' ' '; echo; for p in $PROD $CAND; do $p -m pip list 2>/dev/null | grep -i mlx; done; } > $OUT/env.txt 2>&1

echo "--- dg bitcheck $(date -u +%FT%TZ)"
$PROD /var/tmp/dg/dg_bitcheck.py $OUT/bits-prod.json > $OUT/bits-prod.log 2>&1
$CAND /var/tmp/dg/dg_bitcheck.py $OUT/bits-cand.json > $OUT/bits-cand.log 2>&1

echo "--- np group bitcheck $(date -u +%FT%TZ)"
$PROD /var/tmp/dg/np_bitcheck.py $OUT/np-prod.json > $OUT/np-prod.log 2>&1
$CAND /var/tmp/dg/np_bitcheck.py $OUT/np-cand.json > $OUT/np-cand.log 2>&1

echo "--- NIR dumps $(date -u +%FT%TZ)"
AGX_MESA_DEBUG=shaders MESA_SHADER_CACHE_DISABLE=true $CAND /var/tmp/dg/np_bitcheck.py $OUT/np-nir.json > $OUT/np-nir.log 2>&1
AGX_MESA_DEBUG=shaders MESA_SHADER_CACHE_DISABLE=true $PROD python3 -c "import mlx.core as mx
x=mx.random.normal((1,2048)).astype(mx.bfloat16)
w=mx.random.normal((2048,)).astype(mx.bfloat16)
y=mx.fast.rms_norm(x,w,1e-6); mx.eval(y)" > $OUT/fn-nir.log 2>&1

for c in 64 128 256 512; do
  for arm in prod cand; do
    case $arm in prod) py=$PROD ;; cand) py=$CAND ;; esac
    echo "--- d$c $arm $(date -u +%FT%TZ)"
    B $py --new-tokens $c --warmup 1 --passes 5 --prefill-tokens 512 --label dg-$arm-d$c-n5 --out $OUT/$arm-d$c.json > $OUT/$arm-d$c.log 2>&1
  done
done

echo "--- kill-switch arm d64 $(date -u +%FT%TZ)"
MLX_OMARCHY_FUSED_GEMV_NORM=0 B $CAND --new-tokens 64 --warmup 1 --passes 5 --prefill-tokens 512 --label dg-candoff-d64-n5 --out $OUT/candoff-d64.json > $OUT/candoff-d64.log 2>&1

( cd $OUT && sha256sum ./*.json ./*.txt > SHA256SUMS )
echo WIN6-DONE $(date -u +%FT%TZ)
