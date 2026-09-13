#!/usr/bin/env bash
set -euo pipefail

if [ "${BF16_SDPA_256_INNER:-0}" != 1 ]; then
  deadline=${DEADLINE_EPOCH:?set Main-granted total deadline epoch}
  seconds=$((deadline - $(date +%s)))
  [ "$seconds" -gt 0 ] || { echo "FATAL expired deadline"; exit 2; }
  [ "$seconds" -le 900 ] || seconds=900
  echo "launcher_pid=$$ deadline_epoch=$deadline timeout_seconds=$seconds"
  exec timeout --foreground -k 10s "${seconds}s" env \
    BF16_SDPA_256_INNER=1 DEADLINE_EPOCH="$deadline" \
    BUNDLE_PATH="${BUNDLE_PATH:?set uploaded source bundle path}" bash "$0"
fi

BASE="$HOME/src/mlx-bf16-grouped-candidate-6b1ac029"
CANDIDATE="$HOME/src/mlx-bf16-sdpa-256-ff815ada"
EXPECTED_BASE=6b1ac0296ba65a8e0075171ca9451e222ddff06b
EXPECTED_CANDIDATE=ff815ada8a1730fdf9762f2cbd5c591d99a9c06b
EXPECTED_BUNDLE_SHA=a35fed5495451e6efbd83aaa0df3361c53252f20a15de6ea2cd8692969e68452
EXPECTED_CONTROL_SHA=fb4b96b0e52de3a07d62fea6766baa24f409b0b6b32afef1fd5f365e246e1c50
EXPECTED_IDS=ff502900d2a179a5
MODEL="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-bf16/snapshots/56d07e766edd7159fbe12ed12d9cf114bf38bf1e"
PY="$BASE/.work/venv-run/bin/python"
SCRIPTS="$BASE/scripts"
R="$HOME/bf16-sdpa-256-window-$(date -u +%Y%m%dT%H%M%SZ)"
SAMPLER=

exec 9>/tmp/m1-gpu.lock
flock -n 9 || { echo "lock_acquired=false"; exit 75; }
mkdir -p "$R"
exec > >(tee "$R/window.log") 2>&1
printf 'lock_acquired=true utc=%s pid=%s deadline_epoch=%s\n' \
  "$(date -u +%FT%TZ)" "$$" "$DEADLINE_EPOCH"
printf 'lease=Main-grant-LongContextCostAttribution phase=bf16-sdpa-256-release deadline_epoch=%s\n' \
  "$DEADLINE_EPOCH" > "$R/lease.txt"

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
  flock -u 9
  echo "lock_released=true utc=$(date -u +%FT%TZ) rc=$rc result_dir=$R"
  exit "$rc"
}
trap cleanup EXIT INT TERM

for path in "$BASE/.git" "$SCRIPTS/prepare-mlx.sh" "$SCRIPTS/build-wheel.sh" \
  "$SCRIPTS/bench_decode.py" "$SCRIPTS/bench_matrix.py" \
  "$SCRIPTS/bench_matrix.json" "$PY" "$MODEL/config.json" "$BUNDLE_PATH"; do
  [ -e "$path" ] || { echo "FATAL missing $path"; exit 2; }
done
[ "$(git -C "$BASE" rev-parse HEAD)" = "$EXPECTED_BASE" ] || {
  echo "FATAL control source mismatch"; exit 3;
}
[ "$(sha256sum "$BUNDLE_PATH" | cut -d' ' -f1)" = "$EXPECTED_BUNDLE_SHA" ] || {
  echo "FATAL source bundle digest mismatch"; exit 3;
}
git -C "$BASE" bundle verify "$BUNDLE_PATH"
python3 - "$HOME" "$R/storage.json" <<'PY'
import json, shutil, sys
usage = shutil.disk_usage(sys.argv[1])
result = {"free_bytes": usage.free, "total_bytes": usage.total}
open(sys.argv[2], "w").write(json.dumps(result, sort_keys=True) + "\n")
assert usage.free >= 8 * 1024**3, result
print(json.dumps(result, sort_keys=True))
PY
df -Pk "$HOME" /tmp > "$R/storage.txt"

if [ -e "$CANDIDATE/.git" ]; then
  [ "$(git -C "$CANDIDATE" rev-parse HEAD)" = "$EXPECTED_CANDIDATE" ] || {
    echo "FATAL existing candidate path has wrong commit"; exit 3;
  }
  [ -z "$(git -C "$CANDIDATE" status --porcelain)" ] || {
    echo "FATAL existing candidate path is dirty"; exit 3;
  }
else
  [ ! -e "$CANDIDATE" ] || { echo "FATAL candidate path exists without .git"; exit 3; }
  git -C "$BASE" fetch "$BUNDLE_PATH" HEAD
  [ "$(git -C "$BASE" rev-parse FETCH_HEAD)" = "$EXPECTED_CANDIDATE" ] || {
    echo "FATAL fetched candidate commit mismatch"; exit 3;
  }
  git -C "$BASE" worktree add --detach "$CANDIDATE" "$EXPECTED_CANDIDATE"
fi
[ "$(git -C "$CANDIDATE" rev-parse HEAD)" = "$EXPECTED_CANDIDATE" ]
[ -z "$(git -C "$CANDIDATE" status --porcelain)" ]

shopt -s nullglob
candidate_wheels=("$CANDIDATE"/dist/mlx_omarchy-*+ff815ad-*.whl)
shopt -u nullglob
if [ "${#candidate_wheels[@]}" -ne 1 ]; then
  echo "build_start utc=$(date -u +%FT%TZ) jobs=2"
  set +e
  (cd "$CANDIDATE" && timeout -k 10s 600s env CMAKE_BUILD_PARALLEL_LEVEL=2 \
    DEV_RELEASE=1 MLX_OMARCHY_WORK_DIR="$CANDIDATE/.work-release" \
    ./scripts/build-wheel.sh) > "$R/build.log" 2>&1
  build_rc=$?
  set -e
  echo "build_end utc=$(date -u +%FT%TZ) rc=$build_rc"
  [ "$build_rc" -eq 0 ] || { tail -40 "$R/build.log"; exit 4; }
fi
shopt -s nullglob
control_wheels=("$BASE"/dist/mlx_omarchy-*+6b1ac029-*.whl)
candidate_wheels=("$CANDIDATE"/dist/mlx_omarchy-*+ff815ad-*.whl)
shopt -u nullglob
[ "${#control_wheels[@]}" -eq 1 ] || { echo "FATAL expected one control wheel"; exit 2; }
[ "${#candidate_wheels[@]}" -eq 1 ] || { echo "FATAL expected one candidate wheel"; exit 2; }
CONTROL=${control_wheels[0]}
CAND=${candidate_wheels[0]}
[ "$(sha256sum "$CONTROL" | cut -d' ' -f1)" = "$EXPECTED_CONTROL_SHA" ] || {
  echo "FATAL control wheel digest mismatch"; exit 3;
}
{
  echo "host=$(hostname)"
  echo "kernel=$(uname -r)"
  echo "control_commit=$(git -C "$BASE" rev-parse HEAD)"
  echo "candidate_commit=$(git -C "$CANDIDATE" rev-parse HEAD)"
  echo "control_wheel=$CONTROL"
  echo "control_sha256=$(sha256sum "$CONTROL" | cut -d' ' -f1)"
  echo "candidate_wheel=$CAND"
  echo "candidate_sha256=$(sha256sum "$CAND" | cut -d' ' -f1)"
} | tee "$R/identity.txt"

"$PY" - "$SCRIPTS" "$R/prompt.txt" <<'PY'
import json, pathlib, sys
sys.path.insert(0, sys.argv[1])
import bench_matrix
manifest = json.loads((pathlib.Path(sys.argv[1]) / "bench_matrix.json").read_text())
pathlib.Path(sys.argv[2]).write_text(bench_matrix.prompt_text(manifest, "ctx1024"))
PY

ok=0
for _ in 1 2 3 4 5 6; do
  load=$(cut -d' ' -f1 /proc/loadavg)
  if python3 -c 'import sys; raise SystemExit(float(sys.argv[1]) >= 1.0)' "$load"; then
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
prompt=$(cat "$R/prompt.txt")
args=(--model "$MODEL" --tokens 32 --temp 0.0 --seed 0 --warmup-tokens 4 \
  --prompt "$prompt")
probe() {
  side=$1
  repeat=$2
  if [ "$side" = control ]; then wheel=$CONTROL; else wheel=$CAND; fi
  echo "probe_start side=$side repeat=$repeat utc=$(date -u +%FT%TZ)"
  set +e
  HF_HUB_OFFLINE=1 MLX_DISABLE_COMPILE=1 timeout -k 5s 90s \
    "$PY" "$SCRIPTS/bench_decode.py" "${args[@]}" --wheel "$wheel" \
    > "$R/$side-$repeat.log" 2>&1
  rc=$?
  set -e
  echo "probe_end side=$side repeat=$repeat rc=$rc utc=$(date -u +%FT%TZ)"
  [ "$rc" -eq 0 ] || { cat "$R/$side-$repeat.log"; exit 6; }
}
for spec in control:1 candidate:1 candidate:2 control:2 control:3 candidate:3; do
  probe "${spec%%:*}" "${spec#*:}"
done

"$PY" - "$R" "$EXPECTED_IDS" <<'PY' | tee "$R/analysis.json"
import json, pathlib, statistics, sys
root = pathlib.Path(sys.argv[1])
expected = sys.argv[2]
rows = {}
for side, stamp in (("control", "6b1ac029"), ("candidate", "ff815ad")):
    values = []
    for repeat in (1, 2, 3):
        text = (root / f"{side}-{repeat}.log").read_text()
        assert f"+{stamp}" in text, (side, repeat, "provenance")
        payload = json.loads(text.strip().splitlines()[-1])
        assert payload["ids_sha256_16"] == expected, (side, repeat, payload)
        assert payload["generated"] == 32 and payload["prompt_tokens"] == 1053
        assert payload["device"] == "Apple M1 (G13G B1)"
        values.append(payload["decode_tps"])
    rows[side] = {"decode_tps": values, "median_decode_tps": statistics.median(values)}
control = rows["control"]["median_decode_tps"]
candidate = rows["candidate"]["median_decode_tps"]
result = {
    "schema": "bf16-sdpa-256-release-control/1",
    "generated_ids_sha256_16": expected,
    "rows": rows,
    "candidate_minus_control_tps": candidate - control,
    "candidate_vs_control_percent": (candidate / control - 1.0) * 100.0,
    "candidate_faster_by_median": candidate > control,
}
print(json.dumps(result, indent=2, sort_keys=True))
PY

"$PY" - <<'PY' | tee "$R/device-reopen.json"
import json, mlx.core as mx
print(json.dumps(mx.device_info(), sort_keys=True))
PY
