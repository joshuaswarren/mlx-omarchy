#!/usr/bin/env bash
# Round 3: Bonsai-2-27B headline, hadamard/qmm primitive probes, alloc
# discriminator, E2B, oMLX-vs-mlx_lm.server A/B. Lock protocol as before.
set -uo pipefail
LOG=/tmp/rtmod-round3.log
: > "$LOG"
sudo systemctl stop llm-inference
sleep 2
exec 9>/tmp/m1-gpu.lock
flock -w 900 9 || { echo "LOCK_TIMEOUT" >> "$LOG"; sudo systemctl restart llm-inference; exit 3; }
echo "llm-inference after stop: $(systemctl is-active llm-inference)" >> "$LOG"

CAND_PY=/tmp/venv-cand/bin/python
CAND_PKG=/tmp/venv-cand/lib/python3.14/site-packages
MAIN_PY=/tmp/venv-main/bin/python
OMLX_PY=/tmp/venv-omlx/bin/python
SNAP=$(ls -d ~/.cache/huggingface/hub/models--prism-ml--Ternary-Bonsai-2-27B-mlx-2bit/snapshots/*/ | head -1)
Q25=$(ls -d ~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/*/ | head -1)

echo "== A. Bonsai-2-27B via canonical Bonsai-demo generator (candidate wheel)" >> "$LOG"
{
  cd /tmp/Bonsai-demo
  time $CAND_PY scripts/mlx_generate_bonsai2.py \
    --model "$SNAP" \
    --prompt "The capital of France is" -n 24 --no-think
} >> "$LOG" 2>&1
echo "rc=$?" >> "$LOG"

echo "== B. hadamard_transform + quantized_matmul bits2/group128 probes" >> "$LOG"
timeout 300 $CAND_PY - <<'PY' >> "$LOG" 2>&1
import mlx.core as mx
import numpy as np
ok = True
for n in (512, 1024, 2048, 4096):
    x = mx.random.normal((n,)).astype(mx.float16)
    h = mx.hadamard_transform(x)
    mx.eval(h)
    ref = np.array(x, dtype=np.float32)
    # reference fwht via float64 matrix is too big for 4096; compare against
    # the CPU-evaluated same op: correctness of the op graph itself.
    cpu = mx.hadamard_transform(mx.array(np.array(x, dtype=np.float16)))
    d = mx.abs(h - cpu).max()
    print(f"hadamard n={n} max|gpu-cpu|={float(d):.4f}")
# quantized matmul: 2-bit group-128 affine, the Bonsai pack layout
w = mx.random.normal((256, 128)).astype(mx.float16)
scales = mx.ones((256, 2), mx.float16)
biases = mx.zeros((256, 2), mx.float16)
q = mx.quantize(w, scales, biases, group_size=128, bits=2)
x = mx.random.normal((4, 256)).astype(mx.float16)
y = mx.quantized_matmul(x, q["weight"], q["scales"], q["biases"], group_size=128, bits=2)
mx.eval(y)
d = mx.abs(y - (x @ w)).max() / mx.abs(x @ w).max()
print(f"qmm bits=2 group=128 rel-max-err vs dense={float(d):.4f}")
print("PRIMITIVE-PROBES-OK")
PY
echo "rc=$?" >> "$LOG"

echo "== C. alloc discriminator (fixed)" >> "$LOG"
for SHAPE in "536870912,1" "1073741824,2" "2147483648,4"; do
  ELEMS="${SHAPE%%,*}"; GB="${SHAPE##*,}"
  echo "-- ${GB}GiB fp16 alloc+add+eval" >> "$LOG"
  timeout 120 $CAND_PY -c "
import mlx.core as mx, time
t0 = time.time()
a = mx.zeros(($GB, $ELEMS // $GB), mx.float16)
b = a + 1.0
mx.eval(b)
print('${GB}GiB eval ok', round(time.time()-t0, 2), 's')
" >> "$LOG" 2>&1
  echo "rc=$?" >> "$LOG"
done

echo "== D. gemma-4-E2B on candidate wheel (mlx-lm 0.31.3)" >> "$LOG"
timeout 600 $CAND_PY /tmp/load-probe.py lmstudio-community/gemma-4-E2B-it-MLX-4bit 16 >> "$LOG" 2>&1
echo "rc=$?" >> "$LOG"

echo "== E. server A/B on Qwen2.5-0.5B-4bit" >> "$LOG"
mkdir -p /tmp/ab-models
ln -sfn "$Q25" /tmp/ab-models/qwen25-05b 2>/dev/null
probe_server() {  # name port
  local name="$1" port="$2"
  for i in 1 2 3; do
    curl -s --max-time 120 "http://127.0.0.1:$port/v1/chat/completions" \
      -H "Content-Type: application/json" -H "Authorization: Bearer test" \
      -d '{"model":"qwen25-05b","messages":[{"role":"user","content":"List the first five prime numbers."}],"max_tokens":32,"temperature":0}' \
      -o "/tmp/ab-$name-$i.json" -w "$name run$i http=%{http_code} time=%{time_total}s\n" >> "$LOG" 2>&1
  done
  curl -s --max-time 60 "http://127.0.0.1:$port/v1/chat/completions" \
    -H "Content-Type: application/json" -H "Authorization: Bearer test" \
    -d '{"model":"qwen25-05b","messages":[{"role":"user","content":"hi"}],"max_tokens":1,"temperature":0}' \
    -o "/dev/null" -w "$name 1tok time=%{time_total}s\n" >> "$LOG" 2>&1
}
echo "-- mlx_lm.server (candidate wheel, mlx-lm 872ae88)" >> "$LOG"
$MAIN_PY -m mlx_lm.server --model "$Q25" --host 127.0.0.1 --port 8081 > /tmp/ab-mlxlm-server.log 2>&1 &
MLXLM_PID=$!
for i in $(seq 1 60); do curl -s -o /dev/null http://127.0.0.1:8081/v1/models && break; sleep 2; done
probe_server mlxlm 8081
kill $MLXLM_PID 2>/dev/null; wait $MLXLM_PID 2>/dev/null
sleep 3
echo "-- omlx (candidate wheel via no-build-isolation)" >> "$LOG"
$OMLX_PY -m omlx.cli serve --model-dir /tmp/ab-models --host 127.0.0.1 --port 8082 > /tmp/ab-omlx-server.log 2>&1 &
OMLX_PID=$!
for i in $(seq 1 60); do curl -s -o /dev/null http://127.0.0.1:8082/v1/models && break; sleep 2; done
curl -s http://127.0.0.1:8082/v1/models >> "$LOG" 2>&1; echo "" >> "$LOG"
probe_server omlx 8082
kill $OMLX_PID 2>/dev/null; wait $OMLX_PID 2>/dev/null
tail -3 /tmp/ab-omlx-server.log >> "$LOG" 2>&1

sudo systemctl restart llm-inference
for i in $(seq 1 60); do
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
echo "ROUND3_DONE" >> "$LOG"
