#!/usr/bin/env bash
# Decode-loop structure window (wave/Bf16EncoderBoundary f5649936):
# per-token dispatch/submission/barrier counters around the real
# stream_generate decode loop, gate ON for the candidate wheel. One
# flock; no builds; quiet gate first.
set -uo pipefail
B="$HOME/src/mlx-enc-base"
C="$HOME/src/mlx-enc-cand"
R="$HOME/enc-ab-window4"
MODEL="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-bf16/snapshots/56d07e766edd7159fbe12ed12d9cf114bf38bf1e"

exec 9>/tmp/m1-gpu.lock
echo "$(date -u +%FT%TZ) waiting for m1-gpu.lock pid $$"
flock -w 3600 9 || { echo "FATAL: lock wait exceeded"; exit 4; }
echo "$(date -u +%FT%TZ) LOCK ACQUIRED"
mkdir -p "$R"
ls "$MODEL/config.json" >/dev/null || { echo "FATAL: model snapshot missing"; exit 3; }

ok=0
for i in $(seq 1 270); do
  load=$(cut -d" " -f1 /proc/loadavg)
  pass=$(awk -v l="$load" 'BEGIN{print (l<1.0)?1:0}')
  if [ "$pass" = 1 ]; then
    ok=$((ok+1))
    [ $ok -ge 3 ] && break
    sleep 20
  else
    ok=0
    sleep 20
  fi
done
echo "quiet gate done ok=$ok load=$(cut -d' ' -f1 /proc/loadavg)"
[ $ok -ge 3 ] || { echo "FATAL: quiet gate never passed"; exit 7; }

BASE_PY="$B/.work/venv-base/bin/python"
CAND_PY="$C/.work/venv-cand/bin/python"

echo "=== decode probe cand gate-on $(date -u +%H:%M:%S)"
timeout 600 env MLX_OMARCHY_DEFERRED_POST_BARRIERS=1 HF_HUB_OFFLINE=1 MLX_DISABLE_COMPILE=1 \
  "$CAND_PY" "$R/decode_probe.py" "$MODEL" \
  > "$R/decode-cand-on.txt" 2>&1
echo "cand-on exit=$?"
grep ^DECODEPROBE "$R/decode-cand-on.txt" || tail -3 "$R/decode-cand-on.txt"

echo "=== decode probe base $(date -u +%H:%M:%S)"
timeout 600 env HF_HUB_OFFLINE=1 MLX_DISABLE_COMPILE=1 \
  "$BASE_PY" "$R/decode_probe.py" "$MODEL" \
  > "$R/decode-base.txt" 2>&1
echo "base exit=$?"
grep ^DECODEPROBE "$R/decode-base.txt" || tail -3 "$R/decode-base.txt"

echo "$(date -u +%FT%TZ) window complete -- LOCK RELEASED"
