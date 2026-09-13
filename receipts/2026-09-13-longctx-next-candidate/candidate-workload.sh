#!/usr/bin/env bash
set -euo pipefail

BASE="$HOME/src/mlx-longctx-base-e4d079c"
CANDIDATE="$HOME/src/mlx-longctx-exp-cache-2278ca25"
EXPECTED_BASE=e4d079c883b8606dddef7720bd302c5756be0692
EXPECTED_CANDIDATE=2278ca259618c51cd5ec99fa68c272ffe55b69a5
EXPECTED_BASE_SHADER=e41b54a666e5284921593cd5c3fdac4fca26a4f475abf54cc5483967e88e8706
EXPECTED_CANDIDATE_SHADER=b58a0b9c80cf01d0520ef66ee5fbbc2f16ddb22aefd2efec41c1f80557db5f5c
EXPECTED_IDS=ff502900d2a179a5
MODEL="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-bf16/snapshots/56d07e766edd7159fbe12ed12d9cf114bf38bf1e"
CONTROL_PY="$BASE/.work-longctx/control-venv/bin/python"
CANDIDATE_PY="$CANDIDATE/.work-longctx/candidate-venv/bin/python"
SCRIPTS="$BASE/scripts"
R="$(dirname "$0")/results"
IDENTITY="$R/runtime_identity.py"
GPU_RUNNER="$R/bench_gpu.py"

[ "$(hostname -s)" = jwm1-linux ]
[ -d "$MODEL" ]
[ "$(git -C "$BASE" rev-parse HEAD)" = "$EXPECTED_BASE" ]
[ "$(git -C "$CANDIDATE" rev-parse HEAD)" = "$EXPECTED_CANDIDATE" ]
[ -z "$(git -C "$BASE" status --porcelain --untracked-files=no)" ]
[ -z "$(git -C "$CANDIDATE" status --porcelain --untracked-files=no)" ]
[ "$(sha256sum "$BASE/overlay/mlx/backend/omarchy/shaders/sdpa_decode_native.comp" | cut -d' ' -f1)" = "$EXPECTED_BASE_SHADER" ]
[ "$(sha256sum "$CANDIDATE/overlay/mlx/backend/omarchy/shaders/sdpa_decode_native.comp" | cut -d' ' -f1)" = "$EXPECTED_CANDIDATE_SHADER" ]
[ -x "$CONTROL_PY" ]
[ -x "$CANDIDATE_PY" ]
test -f "$SCRIPTS/bench_decode.py"
[ ! -e "$R" ]
mkdir "$R"

shopt -s nullglob
control_wheels=("$BASE"/dist/mlx_omarchy-*+e4d079c-*.whl)
candidate_wheels=("$CANDIDATE"/dist/mlx_omarchy-*+2278ca2-*.whl)
shopt -u nullglob
[ "${#control_wheels[@]}" -eq 1 ]
[ "${#candidate_wheels[@]}" -eq 1 ]
CONTROL_WHEEL=${control_wheels[0]}
CANDIDATE_WHEEL=${candidate_wheels[0]}
CONTROL_SHA=$(sha256sum "$CONTROL_WHEEL" | cut -d' ' -f1)
CANDIDATE_SHA=$(sha256sum "$CANDIDATE_WHEEL" | cut -d' ' -f1)
printf 'host=%s\nkernel=%s\ncontrol_commit=%s\ncandidate_commit=%s\ncontrol_shader_sha256=%s\ncandidate_shader_sha256=%s\ncontrol_wheel=%s\ncontrol_wheel_sha256=%s\ncandidate_wheel=%s\ncandidate_wheel_sha256=%s\nmodel=%s\n' \
  "$(hostname -s)" "$(uname -r)" "$EXPECTED_BASE" "$EXPECTED_CANDIDATE" \
  "$EXPECTED_BASE_SHADER" "$EXPECTED_CANDIDATE_SHADER" \
  "$CONTROL_WHEEL" "$CONTROL_SHA" "$CANDIDATE_WHEEL" "$CANDIDATE_SHA" \
  "$MODEL" | tee "$R/build-runtime-identity.txt"

cat > "$IDENTITY" <<'PY'
import base64
import csv
import hashlib
import importlib.metadata
import io
import json
import os
import pathlib
import sys
import zipfile

wheel = pathlib.Path(sys.argv[1]).resolve()
expected_python = pathlib.Path(sys.argv[2]).absolute()
expected_prefix = pathlib.Path(sys.argv[3]).resolve()
expected_stamp = sys.argv[4]
assert pathlib.Path(sys.executable).absolute() == expected_python
assert pathlib.Path(sys.prefix).resolve() == expected_prefix
assert "PYTHONPATH" not in os.environ
assert "LD_LIBRARY_PATH" not in os.environ
with zipfile.ZipFile(wheel) as archive:
    names = archive.namelist()
    record_name = next(name for name in names if name.endswith(".dist-info/RECORD"))
    records = {
        row[0]: row[1]
        for row in csv.reader(io.TextIOWrapper(archive.open(record_name)))
    }
    members = [name for name in names if name.endswith(".so")]
    wheel_hashes = {}
    for name in members:
        data = archive.read(name)
        digest = hashlib.sha256(data).digest()
        encoded = base64.urlsafe_b64encode(digest).decode().rstrip("=")
        assert records[name] == "sha256=" + encoded, (name, records.get(name))
        wheel_hashes[name] = hashlib.sha256(data).hexdigest()
import mlx.core as mx
mapped = []
for line in pathlib.Path("/proc/self/maps").read_text().splitlines():
    value = line.split()[-1]
    if "/mlx/" in value and value.endswith(".so") and value not in mapped:
        mapped.append(value)
assert len(mapped) == 2, mapped
mapped_hashes = {}
for value in mapped:
    path = pathlib.Path(value).resolve()
    assert path.is_relative_to(expected_prefix), (path, expected_prefix)
    member = next(name for name in members if name.endswith("/" + path.name))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == wheel_hashes[member], (path, digest, wheel_hashes[member])
    mapped_hashes[str(path)] = digest
version = importlib.metadata.version("mlx-omarchy")
assert expected_stamp in version, version
device = mx.device_info()
assert device["device_name"] == "Apple M1 (G13G B1)", device
print(json.dumps({
    "python": str(expected_python),
    "prefix": str(expected_prefix),
    "version": version,
    "wheel": str(wheel),
    "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
    "loaded_mappings": mapped_hashes,
    "device": device,
}, indent=2, sort_keys=True))
PY
cat > "$GPU_RUNNER" <<'PY'
import runpy
import sys

import mlx.core as mx

script = sys.argv[1]
sys.argv = sys.argv[1:]
mx.set_default_device(mx.gpu)
runpy.run_path(script, run_name="__main__")
PY

env -u PYTHONPATH -u LD_LIBRARY_PATH -u MLX_OMARCHY_FUSED_CHAIN -u MLX_OMARCHY_FUSED_GEMV \
  "$CONTROL_PY" "$IDENTITY" "$CONTROL_WHEEL" "$CONTROL_PY" \
  "$(dirname "$(dirname "$CONTROL_PY")")" +e4d079c > "$R/control-runtime.json"
env -u PYTHONPATH -u LD_LIBRARY_PATH -u MLX_OMARCHY_FUSED_CHAIN -u MLX_OMARCHY_FUSED_GEMV \
  "$CANDIDATE_PY" "$IDENTITY" "$CANDIDATE_WHEEL" "$CANDIDATE_PY" \
  "$(dirname "$(dirname "$CANDIDATE_PY")")" +2278ca2 > "$R/candidate-runtime.json"
"$CONTROL_PY" - "$R/control-runtime.json" "$R/candidate-runtime.json" <<'PY' | tee "$R/runtime-provenance.json"
import json
import pathlib
import sys

control = json.loads(pathlib.Path(sys.argv[1]).read_text())
candidate = json.loads(pathlib.Path(sys.argv[2]).read_text())
control_hashes = set(control["loaded_mappings"].values())
candidate_hashes = set(candidate["loaded_mappings"].values())
assert control_hashes
assert candidate_hashes
assert control_hashes != candidate_hashes
assert control["wheel_sha256"] != candidate["wheel_sha256"]
print(json.dumps({
    "control_wheel_sha256": control["wheel_sha256"],
    "candidate_wheel_sha256": candidate["wheel_sha256"],
    "loaded_images_differ": True,
}, indent=2, sort_keys=True))
PY

"$CONTROL_PY" - "$SCRIPTS" "$R/prompt.txt" <<'PY'
import json
import pathlib
import sys

sys.path.insert(0, sys.argv[1])
import bench_matrix
matrix = json.loads((pathlib.Path(sys.argv[1]) / "bench_matrix.json").read_text())
pathlib.Path(sys.argv[2]).write_text(bench_matrix.prompt_text(matrix, "ctx1024"))
PY

quiet=0
for _ in $(seq 1 18); do
  load=$(cut -d' ' -f1 /proc/loadavg)
  if "$CONTROL_PY" -c 'import sys; raise SystemExit(float(sys.argv[1]) >= 1.0)' "$load"; then
    quiet=$((quiet + 1))
    printf 'quiet_sample=%s load=%s\n' "$quiet" "$load"
    [ "$quiet" -ge 3 ] && break
  else
    quiet=0
    printf 'quiet_reset load=%s\n' "$load"
  fi
  sleep 10
done
[ "$quiet" -ge 3 ] || { echo QUIET_GATE_FAILED_NO_VERDICT; exit 5; }
prompt=$(cat "$R/prompt.txt")
args=(--model "$MODEL" --tokens 32 --temp 0.0 --seed 0 --warmup-tokens 4 --prompt "$prompt")
probe() {
  local side=$1
  local repeat=$2
  local python wheel
  if [ "$side" = control ]; then
    python=$CONTROL_PY
    wheel=$CONTROL_WHEEL
  else
    python=$CANDIDATE_PY
    wheel=$CANDIDATE_WHEEL
  fi
  printf 'probe_start side=%s repeat=%s epoch=%s load=%s\n' \
    "$side" "$repeat" "$(date +%s)" "$(cut -d' ' -f1 /proc/loadavg)"
  set +e
  env -u PYTHONPATH -u LD_LIBRARY_PATH -u MLX_OMARCHY_FUSED_CHAIN -u MLX_OMARCHY_FUSED_GEMV \
    HF_HUB_OFFLINE=1 MLX_DISABLE_COMPILE=1 \
    timeout --foreground --kill-after=5s 85s "$python" "$GPU_RUNNER" "$SCRIPTS/bench_decode.py" \
    "${args[@]}" --wheel "$wheel" > "$R/$side-$repeat.log" 2>&1
  local rc=$?
  set -e
  printf 'probe_end side=%s repeat=%s rc=%s epoch=%s\n' \
    "$side" "$repeat" "$rc" "$(date +%s)"
  [ "$rc" -eq 0 ] || { cat "$R/$side-$repeat.log"; exit 6; }
  cat "$R/$side-$repeat.log"
}
for spec in control:1 candidate:1 candidate:2 control:2 control:3 candidate:3; do
  probe "${spec%%:*}" "${spec##*:}"
done

"$CONTROL_PY" - "$R" "$EXPECTED_IDS" <<'PY' | tee "$R/analysis.json"
import json
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])
expected = sys.argv[2]
def last_json(path):
    values = [json.loads(line) for line in path.read_text().splitlines() if line.startswith("{")]
    assert values, path
    return values[-1]
rows = {
    side: [last_json(root / f"{side}-{index}.log") for index in range(1, 4)]
    for side in ("control", "candidate")
}
for side_rows in rows.values():
    assert [row["ids_sha256_16"] for row in side_rows] == [expected] * 3
    assert [row["generated"] for row in side_rows] == [32] * 3
    assert [row["prompt_tokens"] for row in side_rows] == [1053] * 3
    assert [row["device"] for row in side_rows] == ["Apple M1 (G13G B1)"] * 3
control = statistics.median(row["decode_tps"] for row in rows["control"])
candidate = statistics.median(row["decode_tps"] for row in rows["candidate"])
print(json.dumps({
    "schema": "bf16-sdpa-exp-cache-release-control/1",
    "candidate_faster_by_median": candidate > control,
    "candidate_minus_control_tps": candidate - control,
    "candidate_vs_control_percent": (candidate / control - 1.0) * 100.0,
    "generated_ids_sha256_16": expected,
    "rows": {
        side: {
            "decode_tps": [row["decode_tps"] for row in side_rows],
            "ids_sha256_16": [row["ids_sha256_16"] for row in side_rows],
            "median_decode_tps": statistics.median(row["decode_tps"] for row in side_rows),
        }
        for side, side_rows in rows.items()
    },
}, indent=2, sort_keys=True))
PY
printf 'workload_matrix_complete=true\n'
