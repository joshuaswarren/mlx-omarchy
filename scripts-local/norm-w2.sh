#!/bin/bash
# Jw16DecodeNorm W2: norm-subtree gate window (candidate venv /var/tmp/norm-venv).
# Arms: ctl = serving venv; cand = norm-venv + MLX_OMARCHY_NORM_SUBTREE=1;
#       ctl2 = serving again; ksoff = norm-venv env unset (inertness).
# Every arm: --passes 5, digest must pin; ksoff must equal pins and ctl noise.
set -u
OUT=/var/tmp/norm/gate; mkdir -p "$OUT"
PYS=/var/tmp/v072-venv-fused/bin/python3
PYC=/var/tmp/norm-venv2/bin/python3
BENCH=$HOME/bench-scripts/qwen38-mlx-bench.py
MODEL=$(echo ~/.cache/huggingface/hub/models--SiddhJagani--Qwen3.8-2B-mlx-4Bit/snapshots/0867d98bfb174b042d88461c0e*)
CELL() { local py="$1" tag="$2" n="$3"; shift 3; env "$@" "$py" "$BENCH" --model "$MODEL"/ \
  --prompts "$HOME/bench-scripts/qwen38-2b-prompts.jsonl" --limit 1 --warmup 1 --passes 5 \
  --prefill-tokens 512 --new-tokens "$n" --label "$tag-d$n" --out "$OUT/$tag-d$n.json" \
  > "$OUT/$tag-d$n.log" 2>&1; }
{ date -u +%FT%TZ; echo "boot $(cat /proc/sys/kernel/random/boot_id)"; echo "load $(cat /proc/loadavg)"; } > "$OUT/env.txt"
# d64 interleaved: ctl, cand, ctl2, ksoff
CELL "$PYS" ctl 64
CELL "$PYC" cand 64 MLX_OMARCHY_NORM_SUBTREE=1
CELL "$PYS" ctl2 64
CELL "$PYC" ksoff 64
for N in 128 256 512; do
  CELL "$PYS" ctl $N
  CELL "$PYC" cand $N MLX_OMARCHY_NORM_SUBTREE=1
done
# bitchecks: serving vs cand-env-on (must be row-identical), serving vs cand-env-off (inert)
MLX_OMARCHY_NORM_SUBTREE=1 "$PYC" /var/tmp/dg/dg_bitcheck.py "$OUT/bit-on.json" > "$OUT/bit-on.log" 2>&1
"$PYS" /var/tmp/dg/dg_bitcheck.py "$OUT/bit-serve.json" > "$OUT/bit-serve.log" 2>&1
"$PYC" /var/tmp/dg/dg_bitcheck.py "$OUT/bit-ksoff.json" > "$OUT/bit-ksoff.log" 2>&1
for f in "$OUT"/*.json; do echo "$(basename "$f"): $($PYS -c "import json;d=json.load(open('$f'));print(d.get('ordered_records_sha256','?')[:12], d.get('decode_tok_rate',{}).get('median'))" 2>/dev/null)"; done
( cd "$OUT" && sha256sum ./* > SHA256SUMS 2>/dev/null )
echo NORM-W2-DONE
