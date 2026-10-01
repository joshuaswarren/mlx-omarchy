#!/bin/bash
# Jw16DecodeNorm W1: per-class census on the DEPLOYED Fuse6 stack (1e7cf5c45 content).
# Diag venv; candidate envs OFF => behavior byte-identical to serving; every
# 5-pass run must hit its production digest pin.
set -u
OUT=/var/tmp/norm/out; mkdir -p "$OUT"
PY=/var/tmp/norm-venv/bin/python3
BENCH=$HOME/bench-scripts/qwen38-mlx-bench.py
MODEL=$(echo ~/.cache/huggingface/hub/models--SiddhJagani--Qwen3.8-2B-mlx-4Bit/snapshots/0867d98bfb174b042d88461c0e*)
grep -q "max_tokens=a.new_tokens" "$BENCH" || { echo "bench lacks max_tokens fix"; exit 1; }
B() { "$PY" "$BENCH" --model "$MODEL"/ --prompts "$HOME/bench-scripts/qwen38-2b-prompts.jsonl" --limit 1 --warmup 1 --passes 5 --prefill-tokens 512 "$@"; }
{ date -u +%FT%TZ; echo "boot $(cat /proc/sys/kernel/random/boot_id)"; echo "load $(cat /proc/loadavg)";
  "$PY" -c "import mlx.core as mx; print(mx.__version__)"; } > "$OUT/env.txt"
for N in 32 64 128; do
  MLX_OMARCHY_BARRIER_REASON=1 B --new-tokens $N --label norm-census-d$N \
    --out "$OUT/d$N-reason.json" > "$OUT/d$N-reason.log" 2> "$OUT/d$N-reason.err"
done
MLX_OMARCHY_BARRIER_REASON=1 MLX_OMARCHY_DAG_DUMP=1 MLX_OMARCHY_BARRIER_PAIRS=20000 MLX_OMARCHY_KV_TRACE=1 \
  "$PY" "$BENCH" --model "$MODEL"/ --prompts "$HOME/bench-scripts/qwen38-2b-prompts.jsonl" \
  --limit 1 --new-tokens 128 --warmup 1 --passes 1 --prefill-tokens 512 \
  --label norm-census-dag --out "$OUT/d128-dag.json" > "$OUT/d128-dag.log" 2> "$OUT/d128-dag.err"
MLX_OMARCHY_GPU_PROFILE=$OUT/profile.jsonl MLX_OMARCHY_GPU_PROFILE_LABEL=norm-d128 \
  B --new-tokens 128 --label norm-census-d128-prof \
  --out "$OUT/d128-profile.json" > "$OUT/d128-profile.log" 2> "$OUT/d128-profile.err"
for f in d32-reason d64-reason d128-reason d128-dag d128-profile; do
  "$PY" - "$OUT/$f.json" <<'PYE'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    print(sys.argv[1].split("/")[-1], d.get("decode_tok_rate", {}).get("median"), d.get("ordered_records_sha256", "?")[:12])
except Exception as e:
    print(sys.argv[1], "PARSE-FAIL", repr(e))
PYE
done
grep -h "reason\|calls=" "$OUT"/d*-reason.err | tail -12 || true
"$PY" /var/tmp/dg/profile_analyze.py "$OUT/profile.jsonl" > "$OUT/profile-analyze.txt" 2>&1 || echo "profile_analyze FAILED"
( cd "$OUT" && sha256sum ./*.json ./*.err ./*.txt ./profile.jsonl > SHA256SUMS 2>/dev/null )
echo NORM-W1-DONE
