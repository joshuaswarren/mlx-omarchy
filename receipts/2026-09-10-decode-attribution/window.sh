#!/usr/bin/env bash
# DecodeAttribution measurement window, fully queued:
#   1. queue on /tmp/m1-gpu.lock (FIFO behind whoever holds it)
#   2. wait for a quiet CPU (1-min loadavg < 1.0, 3 checks 20 s apart)
#   3. sample loadavg every 10 s while measuring
#   4. run the ablation arms, then the chain microbench
set -u
LOG=/tmp/dattr-window.log
exec >>"$LOG" 2>&1
echo "=== wrapper start $(date -u +%FT%TZ) pid $$"

timeout 14400 flock -w 14300 /tmp/m1-gpu.lock bash -s <<'INNER'
echo "=== lock held $(date -u +%FT%TZ) by $$"
cd ~/src/mlx-DecodeAttribution
R=receipts/2026-09-10-decode-attribution

# quiet gate
ok=0
for i in $(seq 1 135); do
  l=$(cut -d" " -f1 /proc/loadavg)
  if awk -v x="$l" 'BEGIN{exit !(x < 1.0)}'; then
    ok=$((ok+1))
    [ "$ok" -ge 3 ] && break
  else
    ok=0
  fi
  echo "quiet-gate wait $i load=$l $(date -u +%FT%TZ)"
  sleep 20
done
echo "quiet gate done ok=$ok load=$(cut -d" " -f1 /proc/loadavg)"

( while :; do echo "$(date +%s) $(cut -d" " -f1-3 /proc/loadavg)"; sleep 10; done ) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

python3 "$R/run_arms.py" \
  --python /home/joshuawarren/src/mlx-main-b6d662a8/.work/venv-run/bin/python \
  --ablate-python /home/joshuawarren/src/mlx-DecodeAttribution/.work/venv-ablate/bin/python \
  --wheel /home/joshuawarren/src/mlx-main-b6d662a8/dist/mlx_omarchy-0.32.2.dev202609101916+b6d662a-cp314-cp314-linux_aarch64.whl \
  --ablate-wheel /home/joshuawarren/src/mlx-DecodeAttribution/dist/mlx_omarchy-0.32.2.dev202609102308+0f79507-cp314-cp314-linux_aarch64.whl \
  --out "$R/arms"

MLX_DISABLE_COMPILE=1 HF_HUB_OFFLINE=1 \
  /home/joshuawarren/src/mlx-main-b6d662a8/.work/venv-run/bin/python \
  "$R/chain_microbench.py" --out "$R/microbench.json" --reps 30

echo "=== window done $(date -u +%FT%TZ)"
INNER
echo "=== wrapper exit rc=$? $(date -u +%FT%TZ)"
