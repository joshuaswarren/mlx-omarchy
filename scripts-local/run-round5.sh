#!/usr/bin/env bash
# Round 5 (build14, 8e6b299f): foreign-waiter fix proofs.
#  S0 sanity: no sim env, basic sin passes.
#  S1 TEST_DROP_SUBMIT=1     - never-began class -> rung 1 (kick+resubmit fresh).
#  S2 TEST_DROP_SUBMIT=1,2   - two consecutive never-began drops -> rung 1 twice.
#  S3 TEST_DROP_SIGNAL=1     - executed-but-unsignaled class -> rung 2 (host-signal).
# All runs assert correct sin values (kernels really executed).
set -uo pipefail
LOG=/tmp/f1-round5.log
: > "$LOG"
sudo systemctl stop llm-inference
sleep 2
exec 9>/tmp/m1-gpu.lock
flock -w 900 9 || { echo "LOCK_TIMEOUT" >> "$LOG"; sudo systemctl restart llm-inference; exit 3; }

PY=/tmp/venv-proof/bin/python
export MLX_OMARCHY_TRACE_DISPATCH=1
export MLX_OMARCHY_HANG_NO_PROGRESS_NS=2000000000

SIN_PROBE='import mlx.core as mx; x=mx.array([0.0,0.5,1.0,-0.75],mx.float32); r=mx.sin(x); print("sin:",r.tolist(),flush=True); expect=[0.0,0.479425538604203,0.8414709848078965,-0.6816387600233341]; got=r.tolist(); ok=all(abs(a-b)<1e-6 for a,b in zip(got,expect)); print("SIN-OK" if ok else "SIN-BAD",flush=True); import sys; sys.exit(0 if ok else 5)'

TWOOP_PROBE='import mlx.core as mx; a=mx.exp(mx.array([1.0,2.0],mx.float32)); print("exp:",a.tolist(),flush=True); x=mx.array([0.0,0.5,1.0,-0.75],mx.float32); r=mx.sin(x); print("sin:",r.tolist(),flush=True); expect=[0.0,0.479425538604203,0.8414709848078965,-0.6816387600233341]; ok=abs(a.tolist()[0]-2.718281828459045)<1e-6 and all(abs(p-q)<1e-6 for p,q in zip(r.tolist(),expect)); print("TWOOP-OK" if ok else "TWOOP-BAD",flush=True); import sys; sys.exit(0 if ok else 5)'

run() {
  local name="$1" envs="$2" probe="$3"
  echo "=== $name ===" >> "$LOG"
  ( export $envs; timeout 150 "$PY" -c "$probe" ) >> "$LOG" 2>&1
  echo "--- rc=$? (0=pass 124=hang)" >> "$LOG"
}

run S0-sanity "MLX_OMARCHY_NOOP=1" "$SIN_PROBE"
run S1-drop1 "MLX_OMARCHY_TEST_DROP_SUBMIT=1" "$SIN_PROBE"
run S2-drop12 "MLX_OMARCHY_TEST_DROP_SUBMIT=1,2" "$TWOOP_PROBE"
run S3-strip1 "MLX_OMARCHY_TEST_DROP_SIGNAL=1" "$SIN_PROBE"
echo "=== ROUND5 PROOFS DONE ===" >> "$LOG"

sudo systemctl restart llm-inference
for i in $(seq 1 90); do
  sleep 2
  code=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8002/health 2>/dev/null)
  [ "$code" = "200" ] && break
done
echo "llm-inference after restart: $(systemctl is-active llm-inference) health=$code" >> "$LOG"
echo "ROUND5_DONE" >> "$LOG"
