#!/bin/bash
# Jw16DecodeNorm W3: Q4_WV2 decode-shape window (candidate venv /var/tmp/norm-venv).
# gemv_shapes decode rows: serving (ctl), cand env-off, cand + MLX_OMARCHY_Q4_WV2=1;
# then d64/d256 cells n=5 with the env on vs serving ctl; digests must pin.
set -u
OUT=/var/tmp/norm/gemv; mkdir -p "$OUT"
PYS=/var/tmp/v072-venv-fused/bin/python3
PYC=/var/tmp/norm-venv/bin/python3
BENCH=$HOME/bench-scripts/qwen38-mlx-bench.py
MODEL=$(echo ~/.cache/huggingface/hub/models--SiddhJagani--Qwen3.8-2B-mlx-4Bit/snapshots/0867d98bfb174b042d88461c0e*)
{ date -u +%FT%TZ; echo "boot $(cat /proc/sys/kernel/random/boot_id)"; echo "load $(cat /proc/loadavg)"; } > "$OUT/env.txt"
"$PYS" /var/tmp/appbar/gemv_shapes.py > "$OUT/shapes-serve.txt" 2>&1
"$PYC" /var/tmp/appbar/gemv_shapes.py > "$OUT/shapes-cand-off.txt" 2>&1
MLX_OMARCHY_Q4_WV2=1 "$PYC" /var/tmp/appbar/gemv_shapes.py > "$OUT/shapes-cand-wv2.txt" 2>&1
CELL() { local py="$1" tag="$2" n="$3"; shift 3; env "$@" "$py" "$BENCH" --model "$MODEL"/ \
  --prompts "$HOME/bench-scripts/qwen38-2b-prompts.jsonl" --limit 1 --warmup 1 --passes 5 \
  --prefill-tokens 512 --new-tokens "$n" --label "$tag-d$n" --out "$OUT/$tag-d$n.json" \
  > "$OUT/$tag-d$n.log" 2>&1; }
CELL "$PYS" ctl 64
CELL "$PYC" cand 64 MLX_OMARCHY_Q4_WV2=1
CELL "$PYS" ctl 256
CELL "$PYC" cand 256 MLX_OMARCHY_Q4_WV2=1
MLX_OMARCHY_Q4_WV2=1 "$PYC" /var/tmp/dg/dg_bitcheck.py "$OUT/bit-wv2.json" > "$OUT/bit-wv2.log" 2>&1
for f in "$OUT"/*.json; do echo "$(basename "$f"): $($PYS -c "import json;d=json.load(open('$f'));print(d.get('ordered_records_sha256','?')[:12], d.get('decode_tok_rate',{}).get('median'))" 2>/dev/null)"; done
grep -h "decode M=1" "$OUT"/shapes-*.txt
( cd "$OUT" && sha256sum ./* > SHA256SUMS 2>/dev/null )
echo NORM-W3-DONE
