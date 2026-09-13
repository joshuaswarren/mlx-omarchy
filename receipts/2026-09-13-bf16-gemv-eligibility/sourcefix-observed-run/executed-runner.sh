#!/usr/bin/env bash
set -euo pipefail

if [ "${SOURCEFIX_CONTINUATION_INNER:-0}" != 1 ]; then
  deadline=${OUTER_DEADLINE_EPOCH:?}
  seconds=$((deadline - $(date +%s)))
  [ "$seconds" -gt 0 ] || { echo "FATAL original outer deadline expired"; exit 2; }
  exec timeout --foreground -k 120s "${seconds}s" env SOURCEFIX_CONTINUATION_INNER=1 \
    OUTER_DEADLINE_EPOCH="$deadline" ACQUIRE_BY_EPOCH="${ACQUIRE_BY_EPOCH:?}" \
    STAGE_DIR="${STAGE_DIR:?}" bash "$0"
fi

BASE="$HOME/src/mlx-bf16-grouped-candidate-6b1ac029"
CANDIDATE="$HOME/src/mlx-bf16-gemv-sourcefix-4f90815c"
EXPECTED_BASE=6b1ac0296ba65a8e0075171ca9451e222ddff06b
EXPECTED_CANDIDATE=4f90815c056cbb29a2ff4e1a958856b85af6390a
EXPECTED_CANDIDATE_SHA=f05c64c9acb17d5280d5ea7a2f0644ca44f97d85ea01a51b4e291d7d07d9f656
EXPECTED_PREFLIGHT_SHA=172c68ba5a33721b3f5eccb372c602a2f7a53fc3651f74afd9a2fe46481c8845
EXPECTED_IDS=ff502900d2a179a5
PY="$BASE/.work/venv-run/bin/python"
CONTROL_SITE="$BASE/.work/venv-run/lib/python3.14/site-packages"
CANDIDATE_SITE="$CANDIDATE/.work-sourcefix/candidate-site"
SCRIPTS="$BASE/scripts"
WORK="$CANDIDATE/.work-sourcefix"
TEST_BUILD="$WORK/tests-build"
IDENTITY="$STAGE_DIR/package-runtime-identity.py"
PREFLIGHT="$STAGE_DIR/sourcefix-continuation-preflight.sh"
MODEL="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-bf16/snapshots/56d07e766edd7159fbe12ed12d9cf114bf38bf1e"
R="$HOME/bf16-gemv-sourcefix-continuation-$(date -u +%Y%m%dT%H%M%SZ)"
SAMPLER=

[ "$(sha256sum "$PREFLIGHT" | cut -d' ' -f1)" = "$EXPECTED_PREFLIGHT_SHA" ] || { echo "FATAL preflight digest mismatch"; exit 3; }
STAGE_DIR="$STAGE_DIR" "$PREFLIGHT"
[ "$(date +%s)" -le "$ACQUIRE_BY_EPOCH" ] || { echo "FATAL acquisition deadline missed"; exit 74; }
exec 9>/tmp/m1-gpu.lock
flock -n 9 || { echo "lock_acquired=false"; exit 75; }
acquired=$(date +%s)
mkdir -p "$R"
exec > >(tee "$R/window.log") 2>&1
echo "lock_acquired=true epoch=$acquired utc=$(date -u +%FT%TZ) pid=$$ original_outer_deadline=$OUTER_DEADLINE_EPOCH"
printf 'lease=Main-grant-LongContextCostAttribution phase=gemv-sourcefix-no-build expected=%s outer_deadline=%s\n' \
  "$EXPECTED_CANDIDATE" "$OUTER_DEADLINE_EPOCH" > "$R/lease.txt"
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
    echo CHILD_PIDS_NOT_CLEAR
    rc=70
  else
    echo CHILD_PIDS_CLEAR
  fi
  printf 'released_epoch=%s released_utc=%s rc=%s\n' "$(date +%s)" "$(date -u +%FT%TZ)" "$rc" >> "$R/lease.txt"
  flock -u 9
  echo "lock_released=true epoch=$(date +%s) utc=$(date -u +%FT%TZ) rc=$rc result_dir=$R"
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 124' TERM

shopt -s nullglob
control_wheels=("$BASE"/dist/mlx_omarchy-*+6b1ac029-*.whl)
candidate_wheels=("$CANDIDATE"/dist/mlx_omarchy-*+4f90815-*.whl)
shopt -u nullglob
CONTROL=${control_wheels[0]}
CANDIDATE_WHEEL=${candidate_wheels[0]}
[ "$(sha256sum "$CANDIDATE_WHEEL" | cut -d' ' -f1)" = "$EXPECTED_CANDIDATE_SHA" ]
printf 'host=%s\nkernel=%s\ncontrol_commit=%s\ncandidate_commit=%s\ncontrol_wheel=%s\ncontrol_sha256=%s\ncandidate_wheel=%s\ncandidate_sha256=%s\n' \
  "$(hostname)" "$(uname -r)" "$EXPECTED_BASE" "$EXPECTED_CANDIDATE" \
  "$CONTROL" "$(sha256sum "$CONTROL" | cut -d' ' -f1)" \
  "$CANDIDATE_WHEEL" "$EXPECTED_CANDIDATE_SHA" | tee "$R/identity.txt"
printf 'inherited_MLX_OMARCHY_FUSED_CHAIN=%q\ninherited_MLX_OMARCHY_FUSED_GEMV=%q\ninherited_PYTHONPATH=%q\ninherited_LD_LIBRARY_PATH=%q\n' \
  "${MLX_OMARCHY_FUSED_CHAIN-}" "${MLX_OMARCHY_FUSED_GEMV-}" "${PYTHONPATH-}" "${LD_LIBRARY_PATH-}" > "$R/inherited-env.txt"

env -u MLX_OMARCHY_FUSED_CHAIN -u MLX_OMARCHY_FUSED_GEMV -u LD_LIBRARY_PATH \
  PYTHONPATH="$CONTROL_SITE" "$PY" "$IDENTITY" \
  "$CONTROL" "$PY" "$CONTROL_SITE" "+6b1ac029" > "$R/control-runtime.json"
env -u MLX_OMARCHY_FUSED_CHAIN -u MLX_OMARCHY_FUSED_GEMV -u LD_LIBRARY_PATH \
  PYTHONPATH="$CANDIDATE_SITE" "$PY" "$IDENTITY" \
  "$CANDIDATE_WHEEL" "$PY" "$CANDIDATE_SITE" "+4f90815" > "$R/candidate-runtime.json"
"$PY" - "$R/control-runtime.json" "$R/candidate-runtime.json" <<'PY' | tee "$R/runtime-provenance.json"
import json, pathlib, sys
control = json.loads(pathlib.Path(sys.argv[1]).read_text())
candidate = json.loads(pathlib.Path(sys.argv[2]).read_text())
control_hashes = set(control["loaded_mappings"].values())
candidate_hashes = set(candidate["loaded_mappings"].values())
assert control_hashes != candidate_hashes, (control_hashes, candidate_hashes)
print(json.dumps({
    "control_version": control["version"],
    "candidate_version": candidate["version"],
    "control_loaded_mappings": control["loaded_mappings"],
    "candidate_loaded_mappings": candidate["loaded_mappings"],
    "loaded_images_differ": True,
}, indent=2, sort_keys=True))
PY

env -u PYTHONPATH -u LD_LIBRARY_PATH cmake -S "$WORK/mlx" -B "$TEST_BUILD" \
  -DCMAKE_BUILD_TYPE=Release -DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_CPU=ON \
  -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF -DMLX_BUILD_TESTS=ON \
  -DMLX_BUILD_PYTHON_BINDINGS=OFF -DMLX_BUILD_EXAMPLES=OFF -DMLX_BUILD_BENCHMARKS=OFF \
  2>&1 | tee "$R/test-configure.log"
env -u PYTHONPATH -u LD_LIBRARY_PATH CMAKE_BUILD_PARALLEL_LEVEL=2 \
  cmake --build "$TEST_BUILD" --target omarchy_fused_chain_tests 2>&1 | tee "$R/test-build.log"
TEST_BIN="$TEST_BUILD/tests/omarchy/omarchy_fused_chain_tests"
[ -x "$TEST_BIN" ] || { echo "FATAL test binary absent"; exit 4; }
env -u MLX_OMARCHY_FUSED_CHAIN -u MLX_OMARCHY_FUSED_GEMV -u PYTHONPATH -u LD_LIBRARY_PATH \
  timeout -k 5s 120s "$TEST_BIN" --success=true \
  --test-case='eager dense bf16 decode gemv groups separate flatten views' \
  > "$R/separate-flatten-test.log" 2>&1
cat "$R/separate-flatten-test.log"
grep -Fq '[doctest] Status: SUCCESS!' "$R/separate-flatten-test.log"
! grep -Fq 'Skipping:' "$R/separate-flatten-test.log"
grep -Eq 'assertions:[[:space:]]+[1-9][0-9]*' "$R/separate-flatten-test.log"
env -u MLX_OMARCHY_FUSED_CHAIN -u MLX_OMARCHY_FUSED_GEMV -u PYTHONPATH -u LD_LIBRARY_PATH \
  timeout -k 5s 180s "$TEST_BIN" --success=true \
  --test-case='eager dense bf16 decode gemv*' > "$R/dense-gemv-suite.log" 2>&1
cat "$R/dense-gemv-suite.log"
grep -Fq '[doctest] Status: SUCCESS!' "$R/dense-gemv-suite.log"
! grep -Fq 'Skipping:' "$R/dense-gemv-suite.log"
grep -Eq 'assertions:[[:space:]]+[1-9][0-9]*' "$R/dense-gemv-suite.log"
"$PY" - "$R/candidate-runtime.json" "$R/inherited-env.txt" "$R/separate-flatten-test.log" <<'PY' | tee "$R/fusion-gate.json"
import json, pathlib, re, sys
runtime = json.loads(pathlib.Path(sys.argv[1]).read_text())
env = pathlib.Path(sys.argv[2]).read_text()
test = pathlib.Path(sys.argv[3]).read_text()
assert "inherited_MLX_OMARCHY_FUSED_CHAIN=''" in env, env
assert "inherited_MLX_OMARCHY_FUSED_GEMV=''" in env, env
assert runtime["loaded_mappings"], runtime
assert "eager dense bf16 decode gemv groups separate flatten views" in test, test
assert "Skipping:" not in test, test
assert "Status: SUCCESS!" in test, test
match = re.search(r"assertions:\s+(\d+)\s+\|\s+(\d+) passed", test)
assert match and int(match.group(1)) > 0 and match.group(1) == match.group(2), test
print(json.dumps({
    "source_bound_installed_library": True,
    "environment_overrides_absent": True,
    "separate_flatten_regression_executed": True,
    "dispatch_count_and_output_assertions_passed": int(match.group(1)),
}, indent=2, sort_keys=True))
PY

"$PY" - "$SCRIPTS" "$R/prompt.txt" <<'PY'
import json, pathlib, sys
sys.path.insert(0, sys.argv[1])
import bench_matrix
matrix = json.loads((pathlib.Path(sys.argv[1]) / "bench_matrix.json").read_text())
pathlib.Path(sys.argv[2]).write_text(bench_matrix.prompt_text(matrix, "ctx1024"))
PY
ok=0
for _ in $(seq 1 18); do
  load=$(cut -d' ' -f1 /proc/loadavg)
  if python3 -c 'import sys; raise SystemExit(float(sys.argv[1]) >= 1.0)' "$load"; then
    ok=$((ok+1)); echo "quiet_sample=$ok load=$load"; [ "$ok" -ge 3 ] && break
  else
    ok=0; echo "quiet_reset load=$load"
  fi
  sleep 10
done
[ "$ok" -ge 3 ] || { echo QUIET_GATE_FAILED_NO_VERDICT; exit 5; }
(while :; do echo "$(date +%s) $(cut -d' ' -f1-3 /proc/loadavg)"; sleep 5; done) > "$R/loadavg.txt" 2>&1 &
SAMPLER=$!
prompt=$(cat "$R/prompt.txt")
args=(--model "$MODEL" --tokens 32 --temp 0.0 --seed 0 --warmup-tokens 4 --prompt "$prompt")
probe() {
  side=$1; repeat=$2
  if [ "$side" = control ]; then site=$CONTROL_SITE; wheel=$CONTROL; else site=$CANDIDATE_SITE; wheel=$CANDIDATE_WHEEL; fi
  echo "probe_start side=$side repeat=$repeat epoch=$(date +%s)"
  set +e
  env -u MLX_OMARCHY_FUSED_CHAIN -u MLX_OMARCHY_FUSED_GEMV -u LD_LIBRARY_PATH \
    PYTHONPATH="$site" HF_HUB_OFFLINE=1 MLX_DISABLE_COMPILE=1 \
    timeout -k 5s 90s "$PY" "$SCRIPTS/bench_decode.py" "${args[@]}" \
    --wheel "$wheel" > "$R/$side-$repeat.log" 2>&1
  rc=$?
  set -e
  echo "probe_end side=$side repeat=$repeat rc=$rc epoch=$(date +%s)"
  [ "$rc" -eq 0 ] || { cat "$R/$side-$repeat.log"; exit 6; }
  cat "$R/$side-$repeat.log"
}
for spec in control:1 candidate:1 candidate:2 control:2 control:3 candidate:3; do
  probe "${spec%%:*}" "${spec#*:}"
done
"$PY" - "$R" "$EXPECTED_IDS" <<'PY' | tee "$R/analysis.json"
import json, pathlib, statistics, sys
root = pathlib.Path(sys.argv[1]); expected = sys.argv[2]
def last_json(path):
    values = [json.loads(line) for line in path.read_text().splitlines() if line.startswith("{")]
    assert values, path
    return values[-1]
rows = {side: [last_json(root / f"{side}-{i}.log") for i in range(1, 4)] for side in ("control", "candidate")}
for side_rows in rows.values():
    assert [row["ids_sha256_16"] for row in side_rows] == [expected] * 3, side_rows
    assert [row["generated"] for row in side_rows] == [32] * 3, side_rows
    assert [row["prompt_tokens"] for row in side_rows] == [1053] * 3, side_rows
control = statistics.median(row["decode_tps"] for row in rows["control"])
candidate = statistics.median(row["decode_tps"] for row in rows["candidate"])
print(json.dumps({
    "schema": "bf16-gemv-sourcefix-release-control/1",
    "candidate_faster_by_median": candidate > control,
    "candidate_minus_control_tps": candidate - control,
    "candidate_vs_control_percent": (candidate / control - 1.0) * 100.0,
    "generated_ids_sha256_16": expected,
    "loaded_images_differ": True,
    "rows": {side: {
        "decode_tps": [row["decode_tps"] for row in side_rows],
        "ids_sha256_16": [row["ids_sha256_16"] for row in side_rows],
        "median_decode_tps": statistics.median(row["decode_tps"] for row in side_rows),
    } for side, side_rows in rows.items()},
}, indent=2, sort_keys=True))
PY
[ "$(date +%s)" -lt "$OUTER_DEADLINE_EPOCH" ] || { echo FATAL_FIXED_DEADLINE_EXCEEDED; exit 8; }
echo "WINDOW_COMPLETE result_dir=$R"
