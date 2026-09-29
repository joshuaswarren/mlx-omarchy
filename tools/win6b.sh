#!/bin/bash
# Jw16DecodeGap2 window 6b: re-run the group micro bitcheck (fixed swiglu chain) + NIR dumps.
set -u
OUT=/var/tmp/dg/w6; mkdir -p $OUT
PROD=/var/tmp/v072-venv-fused/bin/python3
CAND=/var/tmp/dg-venv-np/bin/python3
{ date -u; cat /proc/sys/kernel/random/boot_id; cat /proc/loadavg; } > $OUT/env-6b.txt 2>&1
echo "--- np group bitcheck 6b $(date -u +%FT%TZ)"
$PROD /var/tmp/dg/np_bitcheck.py $OUT/np-prod.json > $OUT/np-prod.log 2>&1
$CAND /var/tmp/dg/np_bitcheck.py $OUT/np-cand.json > $OUT/np-cand.log 2>&1
echo "--- NIR dumps $(date -u +%FT%TZ)"
AGX_MESA_DEBUG=shaders MESA_SHADER_CACHE_DISABLE=true $CAND /var/tmp/dg/np_bitcheck.py $OUT/np-nir.json > $OUT/np-nir.log 2>&1
AGX_MESA_DEBUG=shaders MESA_SHADER_CACHE_DISABLE=true $PROD python3 -c "import mlx.core as mx
x=mx.random.normal((1,2048)).astype(mx.bfloat16)
w=mx.random.normal((2048,)).astype(mx.bfloat16)
y=mx.fast.rms_norm(x,w,1e-6); mx.eval(y)" > $OUT/fn-nir.log 2>&1
( cd $OUT && sha256sum ./np-*.json ./*nir* > SHA256SUMS-6b )
echo WIN6B-DONE $(date -u +%FT%TZ)
