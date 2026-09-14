#!/usr/bin/env bash
# jwm1 end-to-end legs: the pinned generated-ID digests and a rate
# no-regression, shipped pick against the compiled-in floor, arms
# alternating inside one lock hold.
#
# Separate from qmm-jwm1.sh because the venv that run cloned
# (venv-agxgen) has no mlx_lm; this clones venv-sdpa-after, which has
# mlx_lm 0.31.3, and force-installs the same jw16-built wheel.
set -euo pipefail
OUT=${1:?usage: qmm-jwm1-legs.sh <outdir>}
ROUNDS=${2:-5}
WHEEL=$(echo "$OUT"/mlx_omarchy-*.whl)
VENV="$OUT/venv-lm"
SCRIPTS="$OUT/scripts"
MODEL=/home/joshuawarren/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3
LOCK=/tmp/m1-gpu.lock

if [[ ! -x "$VENV/bin/python" ]]; then
  cp -a /home/joshuawarren/venv-sdpa-after "$VENV"
fi
PY="$VENV/bin/python"
"$PY" -m pip install -q --no-deps --no-index --force-reinstall "$WHEEL"
{
  echo "legs_venv=$VENV cloned_from=/home/joshuawarren/venv-sdpa-after"
  echo "legs_python=$("$PY" -V 2>&1)"
  echo "mlx_lm=$("$PY" -c 'import mlx_lm; print(mlx_lm.__version__)')"
  echo "boot_id=$(cat /proc/sys/kernel/random/boot_id)"
  echo "date_utc=$(date -u +%FT%TZ)"
} | tee "$OUT/provenance-legs.txt"
"$PY" - "$WHEEL" "$SCRIPTS" <<'PY' | tee -a "$OUT/provenance-legs.txt"
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[2])
from mlx_provenance import installed_provenance, provenance_line
prov = installed_provenance(expect_wheel=Path(sys.argv[1]))
print(provenance_line(prov))
print("verified=" + str(prov["verified"]))
PY
grep -q "verified=match" "$OUT/provenance-legs.txt" || {
  echo "REFUSING: wheel not verified=match in the legs venv" >&2; exit 3; }

exec 9<>"$LOCK"
flock -w 300 9 || { echo "could not take $LOCK" >&2; exit 1; }
echo "lock_inode=$(stat -c %i "$LOCK") nested_flock_n=$(flock -n "$LOCK" -c true; echo $?)" \
  | tee "$OUT/lock-legs.txt"

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

exec 9>&-
{
  echo "nested_flock_n_after_release=$(flock -n "$LOCK" -c true; echo $?)"
  echo "lock_inode_after=$(stat -c %i "$LOCK")"
  echo "boot_id_after=$(cat /proc/sys/kernel/random/boot_id)"
  echo "llm_inference_service_after=$(systemctl is-active llm-inference.service 2>/dev/null || true)"
} | tee -a "$OUT/lock-legs.txt"
wc -l "$OUT/legs.jsonl"
