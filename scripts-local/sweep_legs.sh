#!/bin/sh
# Prefill affinity sweep on the venv-bqm1-0606 wheel (the README 8-core wheel).
# taskset -c 0 (1 E-core), -c 0,1 (2 E-cores), -c 0-3 (E-cluster),
# -c 4-7 (P-cluster), -c 4 (1 P-core). Unpinned point reuses D0606 legs.
set -eu
cd ~/benchq
BF16="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-bf16/snapshots/56d07e766edd7159fbe12ed12d9cf114bf38bf1e"
Q4="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3"
echo "START $(date '+%F %T') load: $(uptime)"
for P in BF16 Q4; do
  if [ "$P" = BF16 ]; then M="$BF16"; F=prompt-bf16.txt; else M="$Q4"; F=prompt-q4.txt; fi
  sh run_leg.sh ~/venv-bqm1-0606/bin/python "$M" "$F" 0 5 "S0606-$P-C0"
  sh run_leg.sh ~/venv-bqm1-0606/bin/python "$M" "$F" 0,1 5 "S0606-$P-C01"
  sh run_leg.sh ~/venv-bqm1-0606/bin/python "$M" "$F" 0-3 5 "S0606-$P-E"
  sh run_leg.sh ~/venv-bqm1-0606/bin/python "$M" "$F" 4-7 5 "S0606-$P-P"
  sh run_leg.sh ~/venv-bqm1-0606/bin/python "$M" "$F" 4 5 "S0606-$P-C4"
done
echo "END $(date '+%F %T') load: $(uptime)"
