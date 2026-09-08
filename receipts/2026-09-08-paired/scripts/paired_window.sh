#!/usr/bin/env bash
# Paired ANE+GPU measurement window on jwm1-linux.
# One flock on /tmp/m1-gpu.lock for this whole process tree; never unlink.
# Runs AFTER ParityBaseline release, quiet CPU, release wheel only.
set -uo pipefail
echo "hostname: $(hostname)"
date -u +"window-start %Y-%m-%dT%H:%M:%S.%6NZ"

exec 9>/tmp/m1-gpu.lock
if ! flock -x -w 60 9; then
  echo "ABORT: could not acquire /tmp/m1-gpu.lock within 60s (baseline still holding?)"
  exit 2
fi
echo "lock acquired (fd 9, tree $$)"

mkdir -p /tmp/paired-results

# ---- GPU half: release wheel, default device, no forced CPU anywhere ----
echo "== GPU bench (release wheel 255c2f93, mlx_omarchy 0.32.2.dev202609081618+c254867)"
~/venv-ane-paired-c254/bin/python /tmp/gpu_bench.py --out /tmp/paired-results/gpu-bench-c254-release.json
GPU_RC=$?
echo "gpu bench rc=$GPU_RC"

# ---- ANE half: load lifecycle6 module, benchmark candidates, unload ----
echo "== ANE timed (lifecycle6 harness, candidates b1+b2)"
bash /tmp/ane_timed.sh
ANE_RC=$?
echo "ane timed rc=$ANE_RC"

date -u +"window-end %Y-%m-%dT%H:%M:%S.%6NZ"
flock -u 9
echo "lock released"
echo "PAIRED_DONE gpu_rc=$GPU_RC ane_rc=$ANE_RC"
