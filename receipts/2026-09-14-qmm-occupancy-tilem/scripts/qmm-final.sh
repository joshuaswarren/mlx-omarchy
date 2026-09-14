#!/usr/bin/env bash
# Final jw16 pass for the landed design: shipped 32-row pick (arm
# "off", MLX_OMARCHY_QMM_COOPMAT_WG_PER_CORE=0) against the compiled-in
# floor of 6 workgroups per core (arm "on", env unset). Arms alternate
# round by round inside one lock hold so drift hits both equally.
set -euo pipefail
ROOT=/var/tmp/qmm-tilem
OUT=${1:?usage: qmm-final.sh <outdir>}
ROUNDS=${2:-5}
WHEEL=$(echo "$ROOT"/dist-ab/mlx_omarchy-*.whl)
PY="$OUT/venv/bin/python"
MODEL=/home/joshuawarren/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3
LOCK=/tmp/m1-gpu.lock
mkdir -p "$OUT"

"$PY" -m pip install -q --no-deps --no-index --force-reinstall "$WHEEL"
{
  echo "date_utc=$(date -u +%FT%TZ)"
  echo "host=$(hostname) arch=$(uname -m) kernel=$(uname -r)"
  echo "boot_id=$(cat /proc/sys/kernel/random/boot_id)"
  echo "glibc=$(pacman -Q glibc) python=$("$PY" -V 2>&1)"
  echo "commit=$(git -C "$ROOT" rev-parse HEAD)"
  echo "base=$(git -C "$ROOT" rev-parse HEAD^)"
  echo "dirty=$(git -C "$ROOT" status --porcelain --untracked-files=no | wc -l)"
  echo "wheel=$WHEEL"
  echo "wheel_sha256=$(sha256sum "$WHEEL" | cut -d' ' -f1)"
  echo "mesa=$(pacman -Q mesa-honeykrisp-omarchy 2>/dev/null || echo unknown)"
  echo "llm_inference_service=$(systemctl is-active llm-inference.service || true)"
  echo "cpu_features=$(awk -F': ' '/^Features/{print $2; exit}' /proc/cpuinfo)"
} | tee "$OUT/provenance-final.txt"

sdir=$(echo "$ROOT"/.work/mlx/build/*/mlx.core/mlx/backend/omarchy/shaders)
git -C "$ROOT" show HEAD^:overlay/mlx/backend/omarchy/shaders/qmm_coopmat.comp \
  > /tmp/qmm-shipped-final.comp
glslc -O --target-env=vulkan1.3 /tmp/qmm-shipped-final.comp \
  -o /tmp/qmm-shipped-final.spv
{
  echo "glslc=$(glslc --version | head -1)"
  sha256sum /tmp/qmm-shipped-final.spv "$sdir"/qmm_coopmat_f16.spv \
    "$sdir"/qmm_coopmat_m16_f16.spv
} | tee "$OUT/spirv-gate.txt"

exec 9<>"$LOCK"
flock -w 60 9 || { echo "could not take $LOCK" >&2; exit 1; }
echo "lock_inode=$(stat -c %i "$LOCK") nested_flock_n=$(flock -n "$LOCK" -c true; echo $?)" \
  | tee "$OUT/lock-final.txt"

run_shapes() {  # arm, round, m
  local arm="$1" r="$2" m="$3"
  local -a extra=()
  [[ "$arm" == off ]] && extra=(MLX_OMARCHY_QMM_COOPMAT_WG_PER_CORE=0)
  env -i HOME=/home/joshuawarren PATH=/usr/bin:/bin MLX_DISABLE_COMPILE=1 \
    HF_HUB_OFFLINE=1 "${extra[@]+"${extra[@]}"}" \
    "$PY" "$OUT/qmm_shape_ab.py" --reps 40 --m "$m" 2>/dev/null |
    "$PY" -c "
import json, sys
d = json.loads(sys.stdin.read())
d['arm'] = '$arm'; d['round'] = $r
print(json.dumps(d, sort_keys=True))"
}

run_leg() {  # arm, tag, prompt, round
  local arm="$1" tag="$2" prompt="$3" r="$4"
  local -a extra=()
  [[ "$arm" == off ]] && extra=(MLX_OMARCHY_QMM_COOPMAT_WG_PER_CORE=0)
  env -i HOME=/home/joshuawarren PATH=/usr/bin:/bin MLX_DISABLE_COMPILE=1 \
    HF_HUB_OFFLINE=1 "${extra[@]+"${extra[@]}"}" \
    "$PY" "$ROOT/scripts/bench_decode.py" --model "$MODEL" \
    --prompt "$prompt" --tokens 32 --temp 0 --seed 0 --wheel "$WHEEL" \
    > "$OUT/final-leg-$arm-$tag-r$r.log" 2>&1
  grep -E '^\{' "$OUT/final-leg-$arm-$tag-r$r.log" |
    "$PY" -c "
import json, sys
d = json.loads(sys.stdin.read())
d['arm'] = '$arm'; d['tag'] = '$tag'; d['round'] = $r
print(json.dumps(d, sort_keys=True))"
}

: > "$OUT/final-shapes.jsonl"
for r in $(seq 1 "$ROUNDS"); do
  for arm in off on; do
    for m in 1053 30; do
      run_shapes "$arm" "$r" "$m" >> "$OUT/final-shapes.jsonl"
    done
  done
done

prompt_short=$("$PY" -c "
import json, sys
sys.path.insert(0, '$ROOT/scripts')
import bench_matrix
sys.stdout.write(bench_matrix.prompt_text(
    json.load(open('$ROOT/scripts/bench_matrix.json')), 'short'))")
prompt_ctx=$("$PY" -c "
import json, sys
sys.path.insert(0, '$ROOT/scripts')
import bench_matrix
sys.stdout.write(bench_matrix.prompt_text(
    json.load(open('$ROOT/scripts/bench_matrix.json')), 'ctx1024'))")

# Unmeasured warm pass per arm and leg so the Mesa pipeline cache is
# populated before any quoted rate; the cache stays enabled because
# these are clean rates, not profiled ones.
: > "$OUT/final-legs.jsonl"
for arm in off on; do
  run_leg "$arm" short "$prompt_short" 0 >/dev/null
  run_leg "$arm" ctx1024 "$prompt_ctx" 0 >/dev/null
done
for r in $(seq 1 "$ROUNDS"); do
  for arm in off on; do
    run_leg "$arm" short "$prompt_short" "$r" >> "$OUT/final-legs.jsonl"
    run_leg "$arm" ctx1024 "$prompt_ctx" "$r" >> "$OUT/final-legs.jsonl"
  done
done

exec 9>&-
{
  echo "nested_flock_n_after_release=$(flock -n "$LOCK" -c true; echo $?)"
  echo "lock_inode_after=$(stat -c %i "$LOCK")"
  echo "llm_inference_service_after=$(systemctl is-active llm-inference.service || true)"
} | tee -a "$OUT/lock-final.txt"
wc -l "$OUT/final-shapes.jsonl" "$OUT/final-legs.jsonl"
