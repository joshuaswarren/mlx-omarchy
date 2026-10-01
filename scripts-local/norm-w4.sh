#!/bin/bash
# Jw16DecodeNorm W4: COMBINED candidate gate (norm subtree + Q4_WV2).
# Arms: ctl/ctl2 = serving venv; cand = norm-venv + NORM_SUBTREE=1 + Q4_WV2=1;
#       ksoff = norm-venv env unset. All --passes 5; digests must pin.
# Race battery + pf/logits cells run AFTER this window via their own scripts.
set -u
OUT=/var/tmp/norm/combined; mkdir -p "$OUT"
PYS=/var/tmp/v072-venv-fused/bin/python3
PYC=/var/tmp/norm-venv/bin/python3
BENCH=$HOME/bench-scripts/qwen38-mlx-bench.py
MODEL=$(echo ~/.cache/huggingface/hub/models--SiddhJagani--Qwen3.8-2B-mlx-4Bit/snapshots/0867d98bfb174b042d88461c0e*)
CELL() { local py="$1" tag="$2" n="$3"; shift 3; env "$@" "$py" "$BENCH" --model "$MODEL"/ \
  --prompts "$HOME/bench-scripts/qwen38-2b-prompts.jsonl" --limit 1 --warmup 1 --passes 5 \
  --prefill-tokens 512 --new-tokens "$n" --label "$tag-d$n" --out "$OUT/$tag-d$n.json" \
  > "$OUT/$tag-d$n.log" 2>&1; }
{ date -u +%FT%TZ; echo "boot $(cat /proc/sys/kernel/random/boot_id)"; echo "load $(cat /proc/loadavg)"; } > "$OUT/env.txt"
for N in 64 128 256 512; do
  CELL "$PYS" ctl $N
  CELL "$PYC" cand $N MLX_OMARCHY_NORM_SUBTREE=1 MLX_OMARCHY_Q4_WV2=1
  CELL "$PYS" ctl2 $N
  CELL "$PYC" ksoff $N
done
for f in "$OUT"/*.json; do echo "$(basename "$f"): $($PYS -c "import json;d=json.load(open('$f'));print(d.get('ordered_records_sha256','?')[:12], d.get('decode_tok_rate',{}).get('median'))" 2>/dev/null)"; done
( cd "$OUT" && sha256sum ./* > SHA256SUMS 2>/dev/null )
echo NORM-W4-DONE
