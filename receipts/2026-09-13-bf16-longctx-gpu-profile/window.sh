#!/usr/bin/env bash
set -euo pipefail

if [ "${LONGCTX_GPU_WINDOW_INNER:-0}" != 1 ]; then
  deadline=${DEADLINE_EPOCH:?set the parent-granted total deadline epoch}
  seconds=$((deadline - $(date +%s)))
  [ "$seconds" -gt 0 ] || { echo "FATAL total deadline has expired"; exit 2; }
  [ "$seconds" -le 240 ] || seconds=240
  echo "measurement_launcher_pid=$$ deadline_epoch=$deadline timeout_seconds=$seconds"
  exec timeout --foreground -k 10s "${seconds}s" env \
    LONGCTX_GPU_WINDOW_INNER=1 DEADLINE_EPOCH="$deadline" bash "$0"
fi

BASE="$HOME/src/mlx-bf16-grouped-candidate-6b1ac029"
PROFILE="$HOME/src/mlx-bf16-grouped-profile-6b1ac029"
PAYLOAD=${PAYLOAD_DIR:-/tmp/LongContextCostAttribution-gpu-profile}
SCRIPTS="$BASE/scripts"
PY="$BASE/.work/venv-run/bin/python"
DIAG_SITE="$PROFILE/.work/diag-site"
MODEL="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-bf16/snapshots/56d07e766edd7159fbe12ed12d9cf114bf38bf1e"
EXPECTED=6b1ac0296ba65a8e0075171ca9451e222ddff06b
R="$HOME/longctx-gpu-profile-window-$(date -u +%Y%m%dT%H%M%SZ)"
SAMPLER=

OWN_LOCK=1
if [ "${LOCK_ALREADY_HELD:-0}" = 1 ]; then flock -n 9 || { echo "FATAL inherited lock descriptor is not held"; exit 75; }
else exec 9>/tmp/m1-gpu.lock
  flock -n 9 || { echo "lock_acquired=false"; exit 75; }
fi
mkdir -p "$R"
exec > >(tee "$R/window.log") 2>&1
printf 'lock_acquired=true utc=%s pid=%s deadline_epoch=%s\n' \
  "$(date -u +%FT%TZ)" "$$" "$DEADLINE_EPOCH"
printf 'lease=Main-grant-LongContextCostAttribution phase=gpu-profile deadline_epoch=%s\n' \
  "$DEADLINE_EPOCH" > "$R/lease.txt"
printf 'pid=%s acquired_utc=%s\n' "$$" "$(date -u +%FT%TZ)" >> "$R/lease.txt"

cleanup() {
  rc=$?
  trap - EXIT INT TERM
  if [ -n "$SAMPLER" ]; then
    kill "$SAMPLER" 2>/dev/null || true
    wait "$SAMPLER" 2>/dev/null || true
  fi
  remaining=$(jobs -pr)
  if [ -n "$remaining" ]; then
    kill $remaining 2>/dev/null || true
    wait $remaining 2>/dev/null || true
  fi
  if [ -n "$(jobs -pr)" ]; then
    echo "CHILD_PIDS_NOT_CLEAR"
    rc=70
  else
    echo "CHILD_PIDS_CLEAR"
  fi
  printf 'released_utc=%s rc=%s\n' "$(date -u +%FT%TZ)" "$rc" >> "$R/lease.txt"
  if [ "$OWN_LOCK" = 1 ]; then flock -u 9; echo "lock_released=true utc=$(date -u +%FT%TZ) rc=$rc result_dir=$R"; else echo "lock_release_deferred_to_total_window=true utc=$(date -u +%FT%TZ) rc=$rc result_dir=$R"; fi
  exit "$rc"
}
trap cleanup EXIT INT TERM

shopt -s nullglob
release_wheels=("$BASE"/dist/mlx_omarchy-*+6b1ac029-*.whl)
diag_wheels=("$PROFILE"/dist/mlx_omarchy-*+diag.6b1ac02-*.whl)
shopt -u nullglob
[ "${#release_wheels[@]}" -eq 1 ] || { echo "FATAL expected one release wheel"; exit 2; }
[ "${#diag_wheels[@]}" -eq 1 ] || { echo "FATAL expected one diagnostic wheel"; exit 2; }
RELEASE_WHEEL=${release_wheels[0]}
DIAG_WHEEL=${diag_wheels[0]}
COMPUTE="$PROFILE/.work/mlx/mlx/backend/omarchy/compute.h"
for path in "$BASE/.git" "$PROFILE/.git" "$SCRIPTS/bench_decode.py" \
  "$SCRIPTS/bench_matrix.py" "$SCRIPTS/bench_matrix.json" \
  "$SCRIPTS/profile_analyze.py" "$PY" "$MODEL/config.json" \
  "$RELEASE_WHEEL" "$DIAG_WHEEL" "$DIAG_SITE/mlx" "$COMPUTE" \
  "$PAYLOAD/profile_bench.py" "$PAYLOAD/gpu_profile.py" \
  "$PAYLOAD/analyze_window.py"; do
  [ -e "$path" ] || { echo "FATAL missing $path"; exit 2; }
done
[ "$(git -C "$BASE" rev-parse HEAD)" = "$EXPECTED" ] || { echo "FATAL release source mismatch"; exit 3; }
[ "$(git -C "$PROFILE" rev-parse HEAD)" = "$EXPECTED" ] || { echo "FATAL diagnostic source mismatch"; exit 3; }
[ "$(sha256sum "$RELEASE_WHEEL" | cut -d' ' -f1)" = \
  fb4b96b0e52de3a07d62fea6766baa24f409b0b6b32afef1fd5f365e246e1c50 ] || {
  echo "FATAL release wheel digest mismatch"; exit 3;
}

cp "$COMPUTE" "$R/compute.h"
cp "$PAYLOAD/profile_bench.py" "$PAYLOAD/gpu_profile.py" \
  "$PAYLOAD/analyze_window.py" "$R/"
"$PY" - "$SCRIPTS" "$R" <<'PY'
import json
import pathlib
import sys
sys.path.insert(0, sys.argv[1])
import bench_matrix
root = pathlib.Path(sys.argv[2])
manifest = json.loads((pathlib.Path(sys.argv[1]) / "bench_matrix.json").read_text())
(root / "prompt-short.txt").write_text(bench_matrix.prompt_text(manifest, "short"))
(root / "prompt-longctx.txt").write_text(bench_matrix.prompt_text(manifest, "ctx1024"))
PY

PYTHONPATH="$DIAG_SITE" "$PY" - "$DIAG_WHEEL" "$RELEASE_WHEEL" "$EXPECTED" > "$R/identity.json" <<'PY'
import hashlib
import importlib.metadata
import json
import platform
import socket
import sys
import zipfile
from pathlib import Path
import mlx.core as mx
diag, release, commit = map(Path, sys.argv[1:])
with zipfile.ZipFile(diag) as archive:
    profiling = any(b"MLX_OMARCHY_GPU_PROFILE" in archive.read(name)
                    for name in archive.namelist())
    members = {name: hashlib.sha256(archive.read(name)).hexdigest()
               for name in archive.namelist() if name.endswith(".so")}
print(json.dumps({
    "host": socket.gethostname(),
    "kernel": platform.release(),
    "device_info": mx.device_info(),
    "source_commit": str(commit),
    "model_snapshot": "56d07e766edd7159fbe12ed12d9cf114bf38bf1e",
    "release_wheel": str(release),
    "release_wheel_sha256": hashlib.sha256(release.read_bytes()).hexdigest(),
    "diagnostic_wheel": str(diag),
    "diagnostic_wheel_sha256": hashlib.sha256(diag.read_bytes()).hexdigest(),
    "diagnostic_version": importlib.metadata.version("mlx-omarchy"),
    "profiling_literal_present": profiling,
    "diagnostic_shared_object_member_sha256": members,
}, sort_keys=True))
PY
cat "$R/identity.json"
df -Pk "$HOME" /tmp > "$R/storage.txt"

ok=0
for _ in 1 2 3 4 5 6; do
  load=$(cut -d' ' -f1 /proc/loadavg)
  if awk -v value="$load" 'BEGIN{exit !(value<1.0)}'; then
    ok=$((ok + 1))
    echo "quiet_sample=$ok load=$load"
    [ "$ok" -ge 3 ] && break
  else
    ok=0
    echo "quiet_reset load=$load"
  fi
  sleep 20
done
[ "$ok" -ge 3 ] || { echo "QUIET_GATE_FAILED_NO_VERDICT"; exit 5; }

(while :; do echo "$(date +%s) $(cut -d' ' -f1-3 /proc/loadavg)"; sleep 5; done) \
  > "$R/loadavg.txt" 2>&1 &
SAMPLER=$!
base_args=(--model "$MODEL" --tokens 32 --temp 0.0 --seed 0 \
  --warmup-tokens 4)

probe_control() {
  label=$1
  repeat=$2
  prompt=$(cat "$R/prompt-$label.txt")
  echo "probe_start mode=control label=$label repeat=$repeat utc=$(date -u +%FT%TZ)"
  HF_HUB_OFFLINE=1 MLX_DISABLE_COMPILE=1 timeout -k 5s 90s \
    "$PY" "$SCRIPTS/bench_decode.py" "${base_args[@]}" \
    --wheel "$RELEASE_WHEEL" --prompt "$prompt" \
    > "$R/control-$label-$repeat.log" 2>&1
  rc=$?
  echo "probe_end mode=control label=$label repeat=$repeat rc=$rc utc=$(date -u +%FT%TZ)"
  [ "$rc" -eq 0 ] || { cat "$R/control-$label-$repeat.log"; exit 6; }
}

probe_diag() {
  label=$1
  prompt=$(cat "$R/prompt-$label.txt")
  echo "probe_start mode=diagnostic label=$label utc=$(date -u +%FT%TZ)"
  PYTHONPATH="$DIAG_SITE" HF_HUB_OFFLINE=1 MLX_DISABLE_COMPILE=1 \
    MLX_OMARCHY_GPU_PROFILE="$R/profile-$label.jsonl" \
    MLX_OMARCHY_GPU_PROFILE_LABEL="$label" timeout -k 5s 90s \
    "$PY" "$PAYLOAD/profile_bench.py" "$SCRIPTS" \
    "$R/markers-$label.jsonl" "${base_args[@]}" \
    --wheel "$DIAG_WHEEL" --prompt "$prompt" > "$R/diag-$label.log" 2>&1
  rc=$?
  echo "probe_end mode=diagnostic label=$label rc=$rc utc=$(date -u +%FT%TZ)"
  [ "$rc" -eq 0 ] || { cat "$R/diag-$label.log"; exit 7; }
  "$PY" "$SCRIPTS/profile_analyze.py" "$R/profile-$label.jsonl" \
    --markers "$R/markers-$label.jsonl" --compute-h "$R/compute.h" \
    > "$R/profile-$label.analysis.txt"
}

for spec in short:1 longctx:1 longctx:2 short:2 short:3 longctx:3; do
  probe_control "${spec%%:*}" "${spec#*:}"
done
probe_diag short
probe_diag longctx
"$PY" "$PAYLOAD/analyze_window.py" "$R" | tee "$R/analysis.log"

"$PY" - <<'PY' | tee "$R/device-reopen.json"
import json
import mlx.core as mx
info = mx.device_info()
assert info.get("device_name") == "Apple M1 (G13G B1)", info
print(json.dumps({"device_reopen": True, "device_info": info}, sort_keys=True))
PY

echo "WINDOW_COMPLETE result_dir=$R"
