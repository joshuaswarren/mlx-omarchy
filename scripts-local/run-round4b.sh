#!/usr/bin/env bash
# Round 4b: Bonsai-2-27B headline generation rerun (no /usr/bin/time).
set -uo pipefail
LOG=/tmp/rtmod-round4b.log
: > "$LOG"
sudo systemctl stop llm-inference
sleep 2
exec 9>/tmp/m1-gpu.lock
flock -w 900 9 || { echo "LOCK_TIMEOUT" >> "$LOG"; sudo systemctl restart llm-inference; exit 3; }

CAND_PY=/tmp/venv-cand/bin/python
SNAP=$(ls -d ~/.cache/huggingface/hub/models--prism-ml--Ternary-Bonsai-2-27B-mlx-2bit/snapshots/*/ | head -1)

echo "== A. Bonsai-2-27B headline (cold, 24 tokens)" >> "$LOG"
{
  cd /tmp/Bonsai-demo
  t0=$(date +%s.%N)
  $CAND_PY scripts/mlx_generate_bonsai2.py \
    --model "$SNAP" \
    --prompt "The capital of France is" -n 24 --no-think
  rc=$?
  t1=$(date +%s.%N)
  echo "wall_total=$(echo "$t1 $t0" | awk '{printf "%.1f", $1-$2}')s rc=$rc"
} >> "$LOG" 2>&1

echo "== B. Bonsai-2-27B warm 48 tokens (tok/s basis)" >> "$LOG"
{
  cd /tmp/Bonsai-demo
  t0=$(date +%s.%N)
  $CAND_PY scripts/mlx_generate_bonsai2.py \
    --model "$SNAP" \
    --prompt "Write one sentence about the sea." -n 48 --no-think
  rc=$?
  t1=$(date +%s.%N)
  echo "wall_total=$(echo "$t1 $t0" | awk '{printf "%.1f", $1-$2}')s rc=$rc"
} >> "$LOG" 2>&1

sudo systemctl restart llm-inference
for i in $(seq 1 90); do
  sleep 2
  code=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8002/health 2>/dev/null)
  [ "$code" = "200" ] && break
done
echo "llm-inference after restart: $(systemctl is-active llm-inference) health=$code" >> "$LOG"
KEY=$(sudo cat /etc/llm-inference/api-key)
curl -s --max-time 90 http://127.0.0.1:8002/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Say READY."}],"max_tokens":8}' | head -c 200 >> "$LOG" 2>&1
echo "" >> "$LOG"
echo "ROUND4B_DONE" >> "$LOG"
