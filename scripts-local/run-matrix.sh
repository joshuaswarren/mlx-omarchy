#!/usr/bin/env bash
# Device load-matrix for the 0.32.3 bump candidate on jw16 (real Vulkan GPU).
# Host protocol: TAKE via hub before, hold /tmp/m1-gpu.lock, stop llm-inference,
# restart+CONFIRM after. Output: /tmp/rtmod-matrix.log
set -uo pipefail
LOG=/tmp/rtmod-matrix.log
: > "$LOG"
# The service holds /tmp/m1-gpu.lock via its ExecStart flock fd, so the
# service must be stopped BEFORE this script can take the lock.
sudo systemctl stop llm-inference
sleep 2
exec 9>/tmp/m1-gpu.lock
flock -w 900 9 || { echo "LOCK_TIMEOUT" >> "$LOG"; sudo systemctl restart llm-inference; exit 3; }
echo "llm-inference after stop: $(systemctl is-active llm-inference)" >> "$LOG"

V=/tmp/venv-cand/bin/python
PKG=/tmp/venv-cand/lib/python3.14/site-packages
PARAKEET=$PKG/mlx/bin/mlx-omarchy-parakeet

probe() {
  echo "== probe $1" >> "$LOG"
  timeout "${2:-600}" $V /tmp/load-probe.py "$1" 16 >> "$LOG" 2>&1
  echo "rc=$?" >> "$LOG"
}

probe mlx-community/Qwen2.5-0.5B-Instruct-4bit 300
probe mlx-community/Ministral-3-8B-Instruct-2512-4bit 600
probe prism-ml/Ternary-Bonsai-8B-mlx-2bit 600
probe mlx-community/gemma-4-12B-4bit 900
probe lmstudio-community/gemma-4-E4B-it-MLX-4bit 600
probe lmstudio-community/gemma-4-26B-A4B-it-QAT-MLX-4bit 900
probe mlx-community/gemma-4-31b-it-4bit 900
probe mlx-community/Qwen3.6-27B-mxfp4 900

echo "== probe prism-ml/Ternary-Bonsai-2-27B-mlx-2bit (artifact loader)" >> "$LOG"
SNAP=$(ls -d ~/.cache/huggingface/hub/models--prism-ml--Ternary-Bonsai-2-27B-mlx-2bit/snapshots/*/ | head -1)
timeout 900 $V /tmp/bonsai2-probe.py "$SNAP" 16 >> "$LOG" 2>&1
echo "rc=$?" >> "$LOG"

echo "== parakeet pin check" >> "$LOG"
{
  $PARAKEET download && echo DOWNLOAD_OK
  $PARAKEET verify && echo VERIFY_OK
  $PARAKEET transcribe -o /tmp/rtmod-parakeet-out && echo TRANSCRIBE_OK
  find /tmp/rtmod-parakeet-out -type f -exec sha256sum {} \;
} >> "$LOG" 2>&1

sudo systemctl restart llm-inference
sleep 8
echo "llm-inference after restart: $(systemctl is-active llm-inference)" >> "$LOG"
curl -s -o /dev/null -w "health http %{http_code}\n" http://127.0.0.1:8002/health >> "$LOG" 2>&1
KEY=$(sudo cat /etc/llm-inference/api-key)
curl -s http://127.0.0.1:8002/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Say READY."}],"max_tokens":8}' >> "$LOG" 2>&1
echo "" >> "$LOG"
echo "MATRIX_DONE" >> "$LOG"
