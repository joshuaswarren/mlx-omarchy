#!/usr/bin/env bash
set -uo pipefail

TREE="$HOME/src/mlx-bf16-grouped-candidate-6b1ac029"
SCRIPTS="$TREE/scripts"
PY="$TREE/.work/venv-run/bin/python"
MODEL="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-bf16/snapshots/56d07e766edd7159fbe12ed12d9cf114bf38bf1e"
WHEELS=("$TREE"/dist/mlx_omarchy-*+6b1ac029-*.whl)
CPU_CLOCK=/tmp/LongContextCostAttribution-cpu_clock.py
SCOPE_PERF=/tmp/LongContextCostAttribution-scope_perf.py
CONTENDERS=/tmp/LongContextCostAttribution-contenders.py
R="$HOME/longctx-cost-window-$(date -u +%Y%m%dT%H%M%SZ)"
SAMPLER=

exec 9>/tmp/m1-gpu.lock
if ! flock -n 9; then
  echo "lock_acquired=false"
  exit 75
fi
mkdir -p "$R"
exec > >(tee "$R/window.log") 2>&1
printf 'lock_acquired=true utc=%s pid=%s\n' "$(date -u +%FT%TZ)" "$$"
printf 'lease=Main-grant-LongContextCostAttribution max_seconds=900\n' > "$R/lease.txt"
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
    echo "terminating_child_pids=$remaining"
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

[ "${#WHEELS[@]}" -eq 1 ] || { echo "FATAL expected one qualified wheel, got ${#WHEELS[@]}"; exit 2; }
WHEEL=${WHEELS[0]}
for path in "$TREE/.git" "$SCRIPTS/bench_decode.py" "$SCRIPTS/bench_matrix.py" \
            "$SCRIPTS/bench_matrix.json" "$PY" "$MODEL/config.json" "$WHEEL" \
            "$CPU_CLOCK" "$SCOPE_PERF" "$CONTENDERS"; do
  [ -e "$path" ] || { echo "FATAL missing $path"; exit 2; }
done
command -v perf >/dev/null || { echo "FATAL perf missing"; exit 2; }

{
  echo "host=$(hostname)"
  echo "kernel=$(uname -r)"
  echo "tree_commit=$(git -C "$TREE" rev-parse HEAD)"
  echo "tree_status=$(git -C "$TREE" status --porcelain | wc -l)"
  echo "wheel=$WHEEL"
  echo "wheel_sha256=$(sha256sum "$WHEEL" | cut -d' ' -f1)"
  echo "model_snapshot=56d07e766edd7159fbe12ed12d9cf114bf38bf1e"
  echo "compile_mode=MLX_DISABLE_COMPILE=1"
  pacman -Q mesa-honeykrisp-omarchy 2>/dev/null || true
  df -Pk "$HOME" /tmp
  "$PY" - <<'PY'
import json
import mlx.core as mx
print(json.dumps({"mlx_version": mx.__version__, "device_info": mx.device_info()}, sort_keys=True))
PY
} > "$R/identity.txt"
cat "$R/identity.txt"
[ "$(git -C "$TREE" rev-parse HEAD)" = "6b1ac0296ba65a8e0075171ca9451e222ddff06b" ] || { echo "FATAL tree commit mismatch"; exit 3; }
[ "$(sha256sum "$WHEEL" | cut -d' ' -f1)" = "fb4b96b0e52de3a07d62fea6766baa24f409b0b6b32afef1fd5f365e246e1c50" ] || { echo "FATAL wheel digest mismatch"; exit 3; }

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
cp "$CPU_CLOCK" "$SCOPE_PERF" "$CONTENDERS" "$R/"

if ! "$PY" "$CONTENDERS" > "$R/contenders.txt"; then
  echo "FATAL accelerator contender found:"
  cat "$R/contenders.txt"
  exit 4
fi

ok=0
for _ in 1 2 3 4 5 6; do
  load=$(cut -d' ' -f1 /proc/loadavg)
  if awk -v l="$load" 'BEGIN{exit !(l<1.0)}'; then
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

(while :; do echo "$(date +%s) $(cut -d' ' -f1-3 /proc/loadavg)"; sleep 5; done) > "$R/loadavg.txt" 2>&1 &
SAMPLER=$!

a=(--model "$MODEL" --tokens 32 --temp 0.0 --seed 0 --warmup-tokens 4 --wheel "$WHEEL")
probe() {
  mode=$1
  label=$2
  prompt_file=$3
  prompt=$(cat "$prompt_file")
  echo "probe_start mode=$mode label=$label utc=$(date -u +%FT%TZ)"
  if [ "$mode" = control ]; then
    HF_HUB_OFFLINE=1 MLX_DISABLE_COMPILE=1 timeout -k 5s 120s \
      "$PY" "$SCRIPTS/bench_decode.py" "${a[@]}" --prompt "$prompt" \
      > "$R/$mode-$label.log" 2>&1
  else
    HF_HUB_OFFLINE=1 MLX_DISABLE_COMPILE=1 timeout -k 5s 120s \
      "$PY" "$CPU_CLOCK" "$SCRIPTS" "${a[@]}" --prompt "$prompt" \
      > "$R/$mode-$label.log" 2>&1
  fi
  rc=$?
  echo "probe_end mode=$mode label=$label rc=$rc utc=$(date -u +%FT%TZ)"
  [ "$rc" -eq 0 ] || { cat "$R/$mode-$label.log"; exit 6; }
}

for spec in \
  "short-1|$R/prompt-short.txt" "longctx-1|$R/prompt-longctx.txt" \
  "longctx-2|$R/prompt-longctx.txt" "short-2|$R/prompt-short.txt" \
  "short-3|$R/prompt-short.txt" "longctx-3|$R/prompt-longctx.txt"; do
  probe control "${spec%%|*}" "${spec#*|}"
done
for spec in \
  "short-1|$R/prompt-short.txt" "longctx-1|$R/prompt-longctx.txt" \
  "longctx-2|$R/prompt-longctx.txt" "short-2|$R/prompt-short.txt" \
  "short-3|$R/prompt-short.txt" "longctx-3|$R/prompt-longctx.txt"; do
  probe cpu "${spec%%|*}" "${spec#*|}"
done

for label in short longctx; do
  prompt=$(cat "$R/prompt-$label.txt")
  echo "perf_start label=$label utc=$(date -u +%FT%TZ)"
  HF_HUB_OFFLINE=1 MLX_DISABLE_COMPILE=1 timeout -k 5s 150s \
    perf record -q --clockid mono -e cpu-clock:u -F 1001 --call-graph dwarf \
    -o "$R/perf-$label.data" -- "$PY" "$CPU_CLOCK" "$SCRIPTS" \
    "${a[@]}" --prompt "$prompt" > "$R/perf-$label.log" 2>&1
  rc=$?
  echo "perf_end label=$label rc=$rc utc=$(date -u +%FT%TZ)"
  [ "$rc" -eq 0 ] || { cat "$R/perf-$label.log"; exit 7; }
  timeout -k 5s 150s "$PY" "$SCOPE_PERF" "$R" "$label" | tee "$R/perf-$label-scope.json"
done

set +e
env -u MLX_DISABLE_COMPILE HF_HUB_OFFLINE=1 timeout -k 5s 90s \
  "$PY" "$SCRIPTS/bench_decode.py" "${a[@]}" --prompt Hi \
  > "$R/compiled-bf16-refusal.log" 2>&1
refusal_rc=$?
set -e
[ "$refusal_rc" -ne 0 ] || { echo "FATAL compiled BF16 unexpectedly succeeded"; exit 8; }
grep -q 'Compiled tape bfloat16 is refused' "$R/compiled-bf16-refusal.log" || { cat "$R/compiled-bf16-refusal.log"; exit 8; }
echo "compiled_bf16_refusal_rc=$refusal_rc"

"$PY" - "$R" <<'PY'
import json
import pathlib
import statistics
import sys
root = pathlib.Path(sys.argv[1])
pins = {"short": "f26175202f3dabe9", "longctx": "ff502900d2a179a5"}
summary = {"control": {}, "instrumented": {}, "perf": {}}
for mode, target in (("control", "control"), ("cpu", "instrumented")):
    for prompt in pins:
        records = []
        for rep in range(1, 4):
            text = (root / f"{mode}-{prompt}-{rep}.log").read_text()
            values = []
            for line in text.splitlines():
                try:
                    values.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
            engine = [x for x in values if x.get("engine") == "bench_decode"]
            clocks = [x for x in values if x.get("instrument") == "canonical_decode_cpu_clock_v1"]
            if len(engine) != 1 or engine[0].get("generated") != 32 or engine[0].get("ids_sha256_16") != pins[prompt]:
                raise SystemExit(f"invalid engine record: {mode}-{prompt}-{rep}")
            if "verified=match" not in text or "+6b1ac029" not in text:
                raise SystemExit(f"invalid provenance: {mode}-{prompt}-{rep}")
            if mode == "cpu" and (len(clocks) != 1 or clocks[0].get("intervals") != 31 or clocks[0].get("ids_sha256_16") != pins[prompt]):
                raise SystemExit(f"invalid clock record: {mode}-{prompt}-{rep}")
            record = dict(engine[0])
            record["wall_ms_per_token"] = 1000.0 / engine[0]["decode_tps"] if mode == "control" else clocks[0]["wall_ms_per_token"]
            if clocks:
                record["process_cpu_ms_per_token"] = clocks[0]["process_cpu_ms_per_token"]
                record["python_thread_cpu_ms_per_token"] = clocks[0]["python_thread_cpu_ms_per_token"]
            records.append(record)
        target_prompt = {"records": records}
        for metric in ("wall_ms_per_token", "process_cpu_ms_per_token", "python_thread_cpu_ms_per_token"):
            vals = [r[metric] for r in records if metric in r]
            if vals:
                target_prompt[metric] = {"values": vals, "median": statistics.median(vals), "range": max(vals) - min(vals)}
        summary[target][prompt] = target_prompt
for prompt in pins:
    text = (root / f"perf-{prompt}.log").read_text()
    values = []
    for line in text.splitlines():
        try:
            values.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    engine = [x for x in values if x.get("engine") == "bench_decode"]
    clock = [x for x in values if x.get("instrument") == "canonical_decode_cpu_clock_v1"]
    if len(engine) != 1 or len(clock) != 1 or engine[0].get("ids_sha256_16") != pins[prompt] or clock[0].get("ids_sha256_16") != pins[prompt]:
        raise SystemExit(f"invalid perf record: {prompt}")
    summary["perf"][prompt] = {"engine": engine[0], "clock": clock[0]}
c_short = summary["control"]["short"]["wall_ms_per_token"]["median"]
c_long = summary["control"]["longctx"]["wall_ms_per_token"]["median"]
i_short = summary["instrumented"]["short"]["wall_ms_per_token"]["median"]
i_long = summary["instrumented"]["longctx"]["wall_ms_per_token"]["median"]
p_short = summary["instrumented"]["short"]["process_cpu_ms_per_token"]["median"]
p_long = summary["instrumented"]["longctx"]["process_cpu_ms_per_token"]["median"]
summary["attribution"] = {
    "control_longctx_minus_short_ms_per_token": c_long - c_short,
    "instrumented_longctx_minus_short_ms_per_token": i_long - i_short,
    "instrument_overhead_delta_ms_per_token": (i_long - i_short) - (c_long - c_short),
    "process_cpu_longctx_minus_short_ms_per_token": p_long - p_short,
    "control_non_cpu_residual_delta_ms_per_token": (c_long - c_short) - (p_long - p_short),
}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary["attribution"], sort_keys=True))
print("LONGCTX_ATTRIBUTION_WINDOW_VALID")
PY

"$PY" - <<'PY' | tee "$R/device-reopen.json"
import json
import mlx.core as mx
info = mx.device_info()
assert info.get("device_name") == "Apple M1 (G13G B1)", info
print(json.dumps({"device_reopen": True, "device_info": info}, sort_keys=True))
PY

echo "WINDOW_COMPLETE result_dir=$R"
