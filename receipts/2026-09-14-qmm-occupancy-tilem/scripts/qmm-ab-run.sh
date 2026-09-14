#!/usr/bin/env bash
# jw16 A/B for the occupancy-selected coopmat prefill M-tile.
#
# One outer flock -w 60 on /tmp/m1-gpu.lock held across every measured
# run; never stolen, never unlinked. No ANE, no accel0, no module
# load/unload, no reboot, no SET write, no driver change. Nothing under
# ~/src/mlx-omarchy is written: the worktree, wheel, venv and outputs
# all live under /var/tmp.
#
# Arms are one wheel plus an env knob, not two builds:
#   MLX_OMARCHY_QMM_COOPMAT_WG_PER_CORE=0  -> shipped pick (32 rows)
#   unset                                  -> compiled-in floor (16/core)
# so the baseline arm runs the byte-identical 32-row SPIR-V.
set -euo pipefail
ROOT=/var/tmp/qmm-tilem
OUT=${1:?usage: qmm-ab-run.sh <outdir>}
WHEEL=$(echo "$ROOT"/dist-ab/mlx_omarchy-*.whl)
VENV="$OUT/venv"
MODEL=/home/joshuawarren/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3
LOCK=/tmp/m1-gpu.lock
mkdir -p "$OUT"

power() {
  for f in /sys/class/power_supply/*/online /sys/class/power_supply/*/status; do
    [[ -r "$f" ]] && echo "$f=$(cat "$f")"
  done
}

echo "== identity =="
{
  echo "date_utc=$(date -u +%FT%TZ)"
  echo "host=$(hostname)"
  echo "arch=$(uname -m)"
  echo "kernel=$(uname -r)"
  echo "boot_id=$(cat /proc/sys/kernel/random/boot_id)"
  echo "nproc=$(nproc)"
  echo "worktree_commit=$(git -C "$ROOT" rev-parse HEAD)"
  echo "worktree_describe=$(git -C "$ROOT" describe --always --dirty)"
  echo "worktree_dirty=$(git -C "$ROOT" status --porcelain --untracked-files=no | wc -l)"
  echo "base_commit=$(git -C "$ROOT" rev-parse HEAD^)"
  echo "wheel=$WHEEL"
  echo "wheel_sha256=$(sha256sum "$WHEEL" | cut -d' ' -f1)"
  echo "model=$MODEL"
  echo "mesa_pkg=$(pacman -Q mesa-honeykrisp-omarchy 2>/dev/null || pacman -Q mesa 2>/dev/null || echo unknown)"
  echo "llm_inference_service=$(systemctl is-active llm-inference.service 2>/dev/null || true)"
  echo "ane_module=$(grep -c . /proc/modules >/dev/null; awk '/^ane /{print $1, $2, $3}' /proc/modules || echo absent)"
} | tee "$OUT/provenance.txt"
power | tee "$OUT/power-before.txt" >/dev/null

echo "== venv =="
if [[ ! -x "$VENV/bin/python" ]]; then
  cp -a /home/joshuawarren/.local/share/mlx-omarchy-test-venv "$VENV"
fi
"$VENV/bin/python" -m pip install -q --no-deps --no-index --force-reinstall \
  "$WHEEL"
PY="$VENV/bin/python"
"$PY" -c "
import mlx.core as mx, json
print(json.dumps({k: str(v) for k, v in mx.metal.device_info().items()}, indent=1, sort_keys=True))" \
  | tee "$OUT/device.txt" >/dev/null
(vulkaninfo --summary 2>/dev/null |
   grep -E "driverName|driverInfo|apiVersion|deviceName|driverID" || true) \
  | tee "$OUT/driver.txt" >/dev/null
cp "$ROOT"/overlay/mlx/backend/omarchy/shaders/qmm_coopmat.comp "$OUT/"
sha256sum "$OUT/qmm_coopmat.comp" >> "$OUT/provenance.txt"

env_base=(env -i HOME=/home/joshuawarren PATH=/usr/bin:/bin
  MLX_DISABLE_COMPILE=1 HF_HUB_OFFLINE=1)

echo "== measured runs (locked) =="
exec 9<>"$LOCK"  # <> never truncates: the lock file keeps its inode
flock -w 60 9 || { echo "could not take $LOCK" >&2; exit 1; }
{
  echo "lock=$LOCK inode=$(stat -c %i "$LOCK")"
  echo "nested_flock_n_while_held=$(flock -n "$LOCK" -c true; echo $?)"
} | tee "$OUT/lock.txt"

# --- per-shape sweep: fresh process per floor, the env is read once ---
: > "$OUT/shape-ab.jsonl"
for pc in 0 4 8 16 29 200 100000; do
  echo "-- shape sweep wg_per_core=$pc"
  "${env_base[@]}" MLX_OMARCHY_QMM_COOPMAT_WG_PER_CORE="$pc" \
    "$PY" "$OUT/qmm_shape_ab.py" --reps 60 >> "$OUT/shape-ab.jsonl"
done
echo "-- shape sweep default (env unset)"
"${env_base[@]}" "$PY" "$OUT/qmm_shape_ab.py" --reps 60 \
  >> "$OUT/shape-ab.jsonl"

# --- end to end: pinned bench_decode protocol, fresh process per leg ---
prompt_short=$("$PY" -c "
import json, sys
sys.path.insert(0, '$ROOT/scripts')
import bench_matrix
m = json.load(open('$ROOT/scripts/bench_matrix.json'))
sys.stdout.write(bench_matrix.prompt_text(m, 'short'))")
prompt_ctx=$("$PY" -c "
import json, sys
sys.path.insert(0, '$ROOT/scripts')
import bench_matrix
m = json.load(open('$ROOT/scripts/bench_matrix.json'))
sys.stdout.write(bench_matrix.prompt_text(m, 'ctx1024'))")
echo "prompt_short_bytes=${#prompt_short} prompt_ctx_bytes=${#prompt_ctx}" \
  | tee -a "$OUT/provenance.txt"

run_leg() {  # arm tag, env assignment array name, prompt, reps
  local arm="$1" pc="$2" tag="$3" prompt="$4" reps="$5" i
  local -a extra=()
  [[ "$pc" != "unset" ]] && extra=(MLX_OMARCHY_QMM_COOPMAT_WG_PER_CORE="$pc")
  # One unmeasured warm run per arm+leg so the Mesa pipeline cache is
  # populated before any quoted rate (the cache is left enabled here:
  # these are clean rates, not the profiler's).
  "${env_base[@]}" "${extra[@]+"${extra[@]}"}" "$PY" "$ROOT/scripts/bench_decode.py" \
    --model "$MODEL" --prompt "$prompt" --tokens 32 --temp 0 --seed 0 \
    > "$OUT/warm-$arm-$tag.log" 2>&1 || true
  for i in $(seq 1 "$reps"); do
    echo "-- leg arm=$arm tag=$tag rep=$i"
    "${env_base[@]}" "${extra[@]+"${extra[@]}"}" "$PY" "$ROOT/scripts/bench_decode.py" \
      --model "$MODEL" --prompt "$prompt" --tokens 32 --temp 0 --seed 0 \
      --wheel "$WHEEL" 2>&1 | tee "$OUT/leg-$arm-$tag-rep$i.log"
  done
}

run_leg baseline 0 short "$prompt_short" 1
run_leg candidate unset short "$prompt_short" 1
run_leg baseline 0 ctx1024 "$prompt_ctx" 3
run_leg candidate unset ctx1024 "$prompt_ctx" 3

power | tee "$OUT/power-after.txt" >/dev/null
exec 9>&-
{
  echo "nested_flock_n_after_release=$(flock -n "$LOCK" -c true; echo $?)"
  echo "lock_inode_after=$(stat -c %i "$LOCK")"
  echo "llm_inference_service_after=$(systemctl is-active llm-inference.service 2>/dev/null || true)"
} | tee -a "$OUT/lock.txt"
echo "== done: $OUT =="
