#!/usr/bin/env bash
# TrigFix battery: dispatch counts + ctx1024 digest + 12-round
# interleaved A/B. Caller holds /tmp/m1-gpu.lock; never unlinks it.
set -euo pipefail
ROOT=/var/tmp/DecodeEpilogueFold
OUT=$ROOT/jwm1-out-trigfix
MODEL=/home/joshuawarren/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3
mkdir -p "$OUT"
date -u +"%Y-%m-%dT%H:%M:%SZ" | tee "$OUT/started.txt"
hostname | tee "$OUT/host.txt"
ls -li /tmp/m1-gpu.lock | tee "$OUT/lock.txt"
if flock -n /tmp/m1-gpu.lock -c true; then
  echo "ERROR: nested flock -n succeeded; we do not hold the lock" >&2
  exit 2
fi
echo "nested flock -n returned 1 (held)" | tee -a "$OUT/lock.txt"
echo "== dispatch counts =="
for side in base cand; do
  echo "-- $side --"
  MLX_DISABLE_COMPILE=1 HF_HUB_OFFLINE=1 \
    "$ROOT/venv-$side/bin/python" "$ROOT/scripts/dispatch_count.py" \
    --model "$MODEL" --tokens 8 \
    | tee "$OUT/dispatch-$side.json"
done
echo "-- cand fold-off --"
MLX_DISABLE_COMPILE=1 HF_HUB_OFFLINE=1 MLX_OMARCHY_FOLD_EPILOGUE=0 \
  "$ROOT/venv-cand/bin/python" "$ROOT/scripts/dispatch_count.py" \
  --model "$MODEL" --tokens 8 \
  | tee "$OUT/dispatch-cand-off.json"
echo "== ctx1024 digest =="
"$ROOT/venv-base/bin/python" "$ROOT/scripts/digest_ctx1024.py" \
  | tee "$OUT/digest.txt"
echo "== interleaved A/B, 12 rounds =="
WBASE="$(echo "$ROOT"/dist-base/mlx_omarchy-*.whl)"
WCAND="$(echo "$ROOT"/dist-cand/mlx_omarchy-*.whl)"
"$ROOT/venv-base/bin/python" "$ROOT/scripts/ab_decode.py" \
  --arm "base=$ROOT/venv-base/bin/python=$WBASE" \
  --arm "cand=$ROOT/venv-cand/bin/python=$WCAND" \
  --model "$MODEL" --manifest "$ROOT/cand/scripts/bench_matrix.json" \
  --bench "$ROOT/cand/scripts/bench_decode.py" \
  --rounds 12 --out "$OUT/ab.json" \
  | tee "$OUT/ab.txt"
date -u +"%Y-%m-%dT%H:%M:%SZ" | tee "$OUT/finished.txt"
echo "lock inode still $(stat -c %i /tmp/m1-gpu.lock)"
