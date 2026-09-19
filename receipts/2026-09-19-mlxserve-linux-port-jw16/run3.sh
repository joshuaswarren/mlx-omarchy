#!/usr/bin/env bash
# GPU-window wrapper for the three-leg serve bench on jw16.
# Stops llm-inference, takes /tmp/m1-gpu.lock, benches, then restarts
# llm-inference and verifies it with /health + a real completion.
set -u
restart_service() {
  echo "== restarting llm-inference =="
  sudo -n systemctl start llm-inference 2>&1 || echo "ERROR: systemctl start failed"
  for i in $(seq 1 24); do
    if curl -sf --max-time 5 http://127.0.0.1:8002/health >/dev/null 2>&1; then
      echo "llm-inference /health 200 after ~$((i*5))s"
      break
    fi
    sleep 5
  done
  KEY=$(sudo -n cat /etc/llm-inference/api-key 2>/dev/null)
  echo "== real completion check =="
  curl -s --max-time 120 http://127.0.0.1:8002/v1/chat/completions \
    -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
    -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Say hi in one word."}],"max_tokens":16}' \
    | head -c 400
  echo
}

echo "== stopping llm-inference =="
if ! sudo -n systemctl stop llm-inference; then
  echo "FATAL: cannot stop llm-inference"
  exit 9
fi
sudo -n systemctl reset-failed llm-inference 2>/dev/null

exec 9>/tmp/m1-gpu.lock
if ! flock -w 180 9; then
  echo "FATAL: GPU lock still busy after 180s"
  restart_service
  exit 10
fi
echo "== GPU lock held $(date -Is) =="

trap 'restart_service; flock -u 9' EXIT

cd /tmp/servebench
/tmp/servebench/venv/bin/python bench3.py
rc=$?
echo "== bench3 rc=$rc =="
exit $rc
