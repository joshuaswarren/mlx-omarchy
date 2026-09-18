#!/usr/bin/env bash
# Round 2 device closures: parakeet transcribe (correct interpreter),
# big-alloc discriminator, certified-wheel A/B on Ministral, gemma4_unified
# via mlx-lm main. Same lock protocol as run-matrix.sh.
set -uo pipefail
LOG=/tmp/rtmod-round2.log
: > "$LOG"
sudo systemctl stop llm-inference
sleep 2
exec 9>/tmp/m1-gpu.lock
flock -w 900 9 || { echo "LOCK_TIMEOUT" >> "$LOG"; sudo systemctl restart llm-inference; exit 3; }

CAND_PY=/tmp/venv-cand/bin/python
BASE_PY=/tmp/venv-base/bin/python
MAIN_PY=/tmp/venv-main/bin/python
CAND_PKG=/tmp/venv-cand/lib/python3.14/site-packages
BASE_PKG=/tmp/venv-base/lib/python3.14/site-packages
MAIN_PKG=/tmp/venv-main/lib/python3.14/site-packages

echo "== parakeet transcribe (candidate wheel, install-venv interpreter)" >> "$LOG"
{
  $CAND_PY $CAND_PKG/mlx/bin/mlx-omarchy-parakeet transcribe -o /tmp/rtmod-parakeet-out2 && echo TRANSCRIBE_OK
  find /tmp/rtmod-parakeet-out2 -type f -name "*.txt" -exec sh -c 'echo "--- $1"; cat "$1"; sha256sum "$1"' _ {} \;
  find /tmp/rtmod-parakeet-out2 -type f ! -name "*.txt" -exec sha256sum {} \;
} >> "$LOG" 2>&1

echo "== alloc discriminator (candidate wheel)" >> "$LOG"
for GB in 1 4 8; do
  echo "-- ${GB}GiB fp16 alloc+mul+eval" >> "$LOG"
  timeout 120 $CAND_PY -c "
import mlx.core as mx, time
n = $GB * 1024 * 1024 * 512
t0 = time.time()
a = mx.zeros((n,), mx.float16)
b = (a + 1.0)
b.eval()
mx.eval(b)
print('${GB}GiB eval ok', round(time.time()-t0, 2), 's')
" >> "$LOG" 2>&1
  echo "rc=$?" >> "$LOG"
done

echo "== Ministral-3-8B on CERTIFIED baseline wheel" >> "$LOG"
timeout 600 $BASE_PY /tmp/load-probe.py mlx-community/Ministral-3-8B-Instruct-2512-4bit 16 >> "$LOG" 2>&1
echo "rc=$?" >> "$LOG"

echo "== Ministral-3-8B on CERTIFIED baseline wheel, retry after cool-down" >> "$LOG"
sleep 10
timeout 600 $BASE_PY /tmp/load-probe.py mlx-community/Ministral-3-8B-Instruct-2512-4bit 16 >> "$LOG" 2>&1
echo "rc=$?" >> "$LOG"

echo "== gemma-4-12B via mlx-lm main 872ae88 on candidate wheel" >> "$LOG"
timeout 900 $MAIN_PY /tmp/load-probe.py mlx-community/gemma-4-12B-4bit 16 >> "$LOG" 2>&1
echo "rc=$?" >> "$LOG"

sudo systemctl restart llm-inference
sleep 15
echo "llm-inference after restart: $(systemctl is-active llm-inference)" >> "$LOG"
curl -s -o /dev/null -w "health http %{http_code}\n" http://127.0.0.1:8002/health >> "$LOG" 2>&1
KEY=$(sudo cat /etc/llm-inference/api-key)
curl -s --max-time 90 http://127.0.0.1:8002/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Say READY."}],"max_tokens":8}' | head -c 200 >> "$LOG" 2>&1
echo "" >> "$LOG"
echo "ROUND2_DONE" >> "$LOG"
