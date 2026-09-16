#!/usr/bin/env bash
# jw16 12-round interleaved A/B for the attention-scoped RMSNorm GEMV fold:
# base = a21b3c81 (fold off, 201 dispatches, dist-cand wheel)
# vs qkv = branch tip code (+c2fc9246 wheel, code-identical to c3553eb2;
# receipts-only diff verified) with MLX_OMARCHY_FUSED_GEMV_RMSNORM_MLP=0
# (attention scope, 178 dispatches).
# Caller holds /tmp/m1-gpu.lock; never unlinks it. Land rule pins:
# short 7fd25a869ff21678 / ctx1024 7da83f06ec9f001d on every leg.
set -euo pipefail
ROOT=/var/tmp/SwigluRmsJw16
OUT="$ROOT/jw16-out-qkv"
MODEL=/home/joshuawarren/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3
BENCH="$ROOT/rms/scripts/bench_decode.py"
MANIFEST="$ROOT/rms/scripts/bench_matrix.json"
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
echo "-- base (a21b3c81, knob does not exist) --"
MLX_DISABLE_COMPILE=1 HF_HUB_OFFLINE=1 \
  "$ROOT/venv-cand/bin/python" "$ROOT/dispatch_count.py" \
  --model "$MODEL" --tokens 8 | tee "$OUT/dispatch-base.json"
echo "-- qkv knob=0 (attention scope) --"
MLX_DISABLE_COMPILE=1 HF_HUB_OFFLINE=1 MLX_OMARCHY_FUSED_GEMV_RMSNORM_MLP=0 \
  "$ROOT/venv-rms/bin/python" "$ROOT/dispatch_count.py" \
  --model "$MODEL" --tokens 8 | tee "$OUT/dispatch-qkv-off.json"
echo "-- qkv default (full fold, record) --"
MLX_DISABLE_COMPILE=1 HF_HUB_OFFLINE=1 \
  "$ROOT/venv-rms/bin/python" "$ROOT/dispatch_count.py" \
  --model "$MODEL" --tokens 8 | tee "$OUT/dispatch-qkv-default.json"
echo "== interleaved A/B, 12 rounds: base(a21b3c81) vs qkv knob=0 =="
"$ROOT/venv-base/bin/python" "$ROOT/ab_decode.py" \
  --arm "base=$ROOT/venv-cand/bin/python=$(echo "$ROOT"/dist-cand/mlx_omarchy-*.whl)" \
  --arm "qkv=$ROOT/venv-rms/bin/python=$(echo "$ROOT"/dist-rms/mlx_omarchy-*.whl)" \
  --model "$MODEL" --manifest "$MANIFEST" --bench "$BENCH" \
  --rounds 12 --env MLX_OMARCHY_FUSED_GEMV_RMSNORM_MLP=0 \
  --out "$OUT/ab.json" | tee "$OUT/ab.txt"
date -u +"%Y-%m-%dT%H:%M:%SZ" | tee "$OUT/finished.txt"
echo "lock inode still $(stat -c %i /tmp/m1-gpu.lock)"
