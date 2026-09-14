#!/usr/bin/env bash
# Base-M1 half of the coopmat M-tile gate, on jwm1-linux.
#
# What this has to establish, in order:
#  1. The jw16-built wheel is usable here at all: scripts/mlx_provenance.py
#     must report verified=match before any number is quoted. Nobody had
#     installed a jw16-built wheel on jwm1 before; every prior measured
#     leg on either host used a wheel built by the host that ran it.
#  2. The selector reads the core count at RUNTIME. A wheel built on a
#     32-core part must still see 8 cores here. Proof is where the
#     n=128 threshold sits: with 8 cores the 32-row grid of 132
#     workgroups clears a floor of 16 per core (128) and fails 17 (136),
#     so the pick must change between 16 and 17 and NOT between 4 and 5,
#     which is where it would sit if 32 cores had been baked in.
#  3. No regression at 1053, where the floor of 6 per core (48) must not
#     fire on any real shape.
#  4. The short-prompt leg measured, because at m=30 the grids are small
#     enough that it may fire here too.
#  5. The pinned generated-ID digests, both legs.
set -euo pipefail
OUT=${1:?usage: qmm-jwm1.sh <outdir>}
ROUNDS=${2:-5}
WHEEL=$(echo "$OUT"/mlx_omarchy-*.whl)
VENV="$OUT/venv"
SCRIPTS="$OUT/scripts"
MODEL=/home/joshuawarren/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3
LOCK=/tmp/m1-gpu.lock

power() {
  for f in /sys/class/power_supply/*/online /sys/class/power_supply/*/status; do
    [[ -r "$f" ]] && echo "$f=$(cat "$f")"
  done
}

{
  echo "date_utc=$(date -u +%FT%TZ)"
  echo "host=$(hostname) arch=$(uname -m) kernel=$(uname -r)"
  echo "boot_id=$(cat /proc/sys/kernel/random/boot_id)"
  echo "nproc=$(nproc)"
  echo "glibc=$(pacman -Q glibc)"
  echo "mesa=$(pacman -Q mesa-honeykrisp-omarchy 2>/dev/null || pacman -Q mesa 2>/dev/null || echo unknown)"
  echo "wheel=$WHEEL"
  echo "wheel_sha256=$(sha256sum "$WHEEL" | cut -d' ' -f1)"
  echo "wheel_built_on=jw16mbp1-linux"
  echo "llm_inference_service=$(systemctl is-active llm-inference.service 2>/dev/null || true)"
  echo "cpu_features=$(awk -F': ' '/^Features/{print $2; exit}' /proc/cpuinfo)"
} | tee "$OUT/provenance.txt"
power | tee "$OUT/power-before.txt" >/dev/null

if [[ ! -x "$VENV/bin/python" ]]; then
  cp -a /home/joshuawarren/venv-agxgen "$VENV"
fi
PY="$VENV/bin/python"
"$PY" -m pip install -q --no-deps --no-index --force-reinstall "$WHEEL"
echo "python=$("$PY" -V 2>&1)" | tee -a "$OUT/provenance.txt"

# Gate 1: cross-host wheel provenance. Nothing below is quotable if this
# is not verified=match.
"$PY" "$SCRIPTS/mlx_provenance.py" --expect-wheel "$WHEEL" \
  > "$OUT/provenance-gate.txt" 2>&1 || true
"$PY" - "$WHEEL" "$SCRIPTS" <<'PY' | tee -a "$OUT/provenance-gate.txt"
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[2])
from mlx_provenance import installed_provenance, provenance_line
prov = installed_provenance(expect_wheel=Path(sys.argv[1]))
print(provenance_line(prov))
print("verified=" + str(prov["verified"]))
print(json.dumps({k: str(v) for k, v in prov.items()}, sort_keys=True))
PY
grep -q "verified=match" "$OUT/provenance-gate.txt" || {
  echo "REFUSING: jw16-built wheel is not verified=match on jwm1" >&2
  exit 3
}
"$PY" -c "
import mlx.core as mx, json
print(json.dumps({k: str(v) for k, v in mx.device_info().items()},
                 indent=1, sort_keys=True))" | tee "$OUT/device.txt" >/dev/null
(vulkaninfo --summary 2>/dev/null |
   grep -E "driverName|driverInfo|apiVersion|deviceName|driverID" || true) \
  | tee "$OUT/driver.txt" >/dev/null

# The 32-row SPIR-V identity, re-derived with THIS host's glslc so the
# result is not a property of jw16's shader compiler.
if command -v glslc >/dev/null; then
  glslc -O --target-env=vulkan1.3 "$OUT/qmm_coopmat_shipped.comp" \
    -o "$OUT/shipped.spv"
  glslc -O --target-env=vulkan1.3 "$OUT/qmm_coopmat_new.comp" \
    -o "$OUT/new-default.spv"
  glslc -O --target-env=vulkan1.3 -DTILE_ROWS=16 \
    "$OUT/qmm_coopmat_new.comp" -o "$OUT/new-m16.spv"
  { echo "glslc=$(glslc --version | head -1)"
    sha256sum "$OUT/shipped.spv" "$OUT/new-default.spv" "$OUT/new-m16.spv"
  } | tee "$OUT/spirv-gate.txt"
else
  echo "glslc absent on this host" | tee "$OUT/spirv-gate.txt"
fi

exec 9<>"$LOCK"
flock -w 120 9 || { echo "could not take $LOCK" >&2; exit 1; }
echo "lock_inode=$(stat -c %i "$LOCK") nested_flock_n=$(flock -n "$LOCK" -c true; echo $?)" \
  | tee "$OUT/lock.txt"

shape_run() {  # label, m, env-value-or-unset, round
  local label="$1" m="$2" pc="$3" r="$4"
  local -a extra=()
  [[ "$pc" != unset ]] && extra=(MLX_OMARCHY_QMM_COOPMAT_WG_PER_CORE="$pc")
  env -i HOME=/home/joshuawarren PATH=/usr/bin:/bin MLX_DISABLE_COMPILE=1 \
    HF_HUB_OFFLINE=1 "${extra[@]+"${extra[@]}"}" \
    "$PY" "$OUT/qmm_shape_ab.py" --reps 40 --m "$m" --cores 8 2>/dev/null |
    "$PY" -c "
import json, sys
d = json.loads(sys.stdin.read())
d['label'] = '$label'; d['round'] = $r
print(json.dumps(d, sort_keys=True))"
}

# Gates 3 and 4: shipped pick against the compiled-in floor, both legs'
# m, arms alternating.
: > "$OUT/shapes.jsonl"
for r in $(seq 1 "$ROUNDS"); do
  for m in 1053 30; do
    shape_run off "$m" 0 "$r" >> "$OUT/shapes.jsonl"
    shape_run on "$m" unset "$r" >> "$OUT/shapes.jsonl"
  done
done

# Gate 2: the runtime-core-count discriminator, at the 8-core threshold
# and at the 32-core one.
: > "$OUT/threshold.jsonl"
for r in 1 2 3; do
  for pc in 4 5 16 17; do
    shape_run "pc$pc" 1053 "$pc" "$r" >> "$OUT/threshold.jsonl"
  done
done

leg() {  # arm, tag, prompt, round
  local arm="$1" tag="$2" prompt="$3" r="$4"
  local -a extra=()
  [[ "$arm" == off ]] && extra=(MLX_OMARCHY_QMM_COOPMAT_WG_PER_CORE=0)
  env -i HOME=/home/joshuawarren PATH=/usr/bin:/bin MLX_DISABLE_COMPILE=1 \
    HF_HUB_OFFLINE=1 "${extra[@]+"${extra[@]}"}" \
    "$PY" "$SCRIPTS/bench_decode.py" --model "$MODEL" --prompt "$prompt" \
    --tokens 32 --temp 0 --seed 0 --wheel "$WHEEL" \
    > "$OUT/leg-$arm-$tag-r$r.log" 2>&1
  grep -E '^\{' "$OUT/leg-$arm-$tag-r$r.log" | "$PY" -c "
import json, sys
d = json.loads(sys.stdin.read())
d['arm'] = '$arm'; d['tag'] = '$tag'; d['round'] = $r
print(json.dumps(d, sort_keys=True))"
}

prompt_short=$("$PY" -c "
import json, sys
sys.path.insert(0, '$SCRIPTS')
import bench_matrix
sys.stdout.write(bench_matrix.prompt_text(
    json.load(open('$SCRIPTS/bench_matrix.json')), 'short'))")
prompt_ctx=$("$PY" -c "
import json, sys
sys.path.insert(0, '$SCRIPTS')
import bench_matrix
sys.stdout.write(bench_matrix.prompt_text(
    json.load(open('$SCRIPTS/bench_matrix.json')), 'ctx1024'))")
echo "prompt_short_bytes=${#prompt_short} prompt_ctx_bytes=${#prompt_ctx}" \
  | tee -a "$OUT/provenance.txt"

# Gate 5: the pinned digests, plus the rate no-regression. One
# unmeasured warm pass per arm and leg; the Mesa shader cache stays
# enabled because these are clean rates.
: > "$OUT/legs.jsonl"
for arm in off on; do
  leg "$arm" short "$prompt_short" 0 >/dev/null
  leg "$arm" ctx1024 "$prompt_ctx" 0 >/dev/null
done
for r in $(seq 1 "$ROUNDS"); do
  for arm in off on; do
    leg "$arm" short "$prompt_short" "$r" >> "$OUT/legs.jsonl"
    leg "$arm" ctx1024 "$prompt_ctx" "$r" >> "$OUT/legs.jsonl"
  done
done

power | tee "$OUT/power-after.txt" >/dev/null
exec 9>&-
{
  echo "nested_flock_n_after_release=$(flock -n "$LOCK" -c true; echo $?)"
  echo "lock_inode_after=$(stat -c %i "$LOCK")"
  echo "llm_inference_service_after=$(systemctl is-active llm-inference.service 2>/dev/null || true)"
  echo "boot_id_after=$(cat /proc/sys/kernel/random/boot_id)"
} | tee -a "$OUT/lock.txt"
wc -l "$OUT/shapes.jsonl" "$OUT/threshold.jsonl" "$OUT/legs.jsonl"
