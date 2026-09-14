#!/usr/bin/env bash
# One flock hold: C++ bit-exact tests, per-token dispatch counts, then
# the 5-round interleaved decode A/B. Caller must already hold
# /tmp/m1-gpu.lock. Never unlinks the lock.
set -euo pipefail
ROOT=/var/tmp/DecodeEpilogueFold
OUT="$ROOT/jw16-out"
MODEL=/home/joshuawarren/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3
BENCH="$ROOT/cand/scripts/bench_decode.py"
MANIFEST="$ROOT/cand/scripts/bench_matrix.json"
PROV="$ROOT/cand/scripts/mlx_provenance.py"
mkdir -p "$OUT"
date -u +"%Y-%m-%dT%H:%M:%SZ" | tee "$OUT/started.txt"
hostname | tee "$OUT/host.txt"
ls -li /tmp/m1-gpu.lock | tee "$OUT/lock.txt"
# Nested flock must fail while we hold it.
if flock -n /tmp/m1-gpu.lock -c true; then
  echo "ERROR: nested flock -n succeeded; we do not hold the lock" >&2
  exit 2
fi
echo "nested flock -n returned 1 (held)" | tee -a "$OUT/lock.txt"

echo "== C++ fused_chain gemv + kv_ops =="
"$ROOT/build-tests/tests/omarchy/omarchy_fused_chain_tests" -tc="*gemv*" \
  | tee "$OUT/fused_chain_gemv.txt" | tail -6
"$ROOT/build-tests/tests/omarchy/omarchy_kv_ops_tests" -tc="producer-direct*" \
  | tee "$OUT/kv_ops_direct.txt" | tail -6

echo "== dispatch counts =="
for side in base cand; do
  echo "-- $side --"
  MLX_DISABLE_COMPILE=1 HF_HUB_OFFLINE=1 \
    "$ROOT/venv-$side/bin/python" "$ROOT/dispatch_count.py" \
    --model "$MODEL" --tokens 8 \
    | tee "$OUT/dispatch-$side.json"
done
# Same cand wheel with the fold off: isolates the fold from the rest of
# the shader layout change.
echo "-- cand fold-off --"
MLX_DISABLE_COMPILE=1 HF_HUB_OFFLINE=1 MLX_OMARCHY_FOLD_EPILOGUE=0 \
  "$ROOT/venv-cand/bin/python" "$ROOT/dispatch_count.py" \
  --model "$MODEL" --tokens 8 \
  | tee "$OUT/dispatch-cand-off.json"

echo "== interleaved A/B, 5 rounds =="
"$ROOT/venv-base/bin/python" "$ROOT/ab_decode.py" \
  --arm "base=$ROOT/venv-base/bin/python=$ROOT/dist-base/mlx_omarchy-0.32.2.dev202609141639+b79a4b68-cp314-cp314-linux_aarch64.whl" \
  --arm "cand=$ROOT/venv-cand/bin/python=$ROOT/dist-cand/mlx_omarchy-0.32.2.dev202609141654+50870b69-cp314-cp314-linux_aarch64.whl" \
  --model "$MODEL" --manifest "$MANIFEST" --bench "$BENCH" \
  --rounds 5 --out "$OUT/ab.json" \
  | tee "$OUT/ab.txt"

date -u +"%Y-%m-%dT%H:%M:%SZ" | tee "$OUT/finished.txt"
echo "lock inode still $(stat -c %i /tmp/m1-gpu.lock)"
