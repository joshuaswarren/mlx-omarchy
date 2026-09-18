#!/usr/bin/env bash
set -uo pipefail
LOG=/tmp/rtmod-gates.log
: > "$LOG"
sudo systemctl stop llm-inference
sleep 2
exec 9>/tmp/m1-gpu.lock
flock -w 900 9 || { echo "LOCK_TIMEOUT" >> "$LOG"; sudo systemctl restart llm-inference; exit 3; }
B=~/src/mlx-omarchy-rtmod2/.work/build-tests/tests/omarchy
for t in omarchy_primitive_tests omarchy_runtime_tests omarchy_copy_offset_tests omarchy_matmul_family_tests omarchy_take_fill_tests omarchy_select_layout_tests omarchy_fast_ops_tests omarchy_fast_regression_tests omarchy_error_contract_tests omarchy_compiled_tape_tests omarchy_wrong_value_sweep_tests omarchy_capability_sim_tests; do
  echo "== $t" >> "$LOG"
  timeout 600 "$B/$t" >> "$LOG" 2>&1
  echo "rc=$?" >> "$LOG"
done
sudo systemctl restart llm-inference
for i in $(seq 1 90); do sleep 2; code=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8002/health 2>/dev/null); [ "$code" = "200" ] && break; done
echo "llm-inference: $(systemctl is-active llm-inference) health=$code" >> "$LOG"
echo "GATES_DONE" >> "$LOG"
