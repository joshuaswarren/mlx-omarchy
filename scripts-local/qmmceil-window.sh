#!/usr/bin/env bash
# QmmPrefillCeiling lane: bounded GPU-window stages on an M1 host.
# Usage: qmmceil-window.sh <build|measure|pins> <venv-template> <model-path>
#   build   - sync source assumed done; build diag wheel + private venv
#   measure - phase profiles (m=30 short, m=1053 ctx), shape census,
#             kernel-isolated probe arms 0/3/4/5/6 (digest + perf)
#   pins    - bench_decode short + ctx1024 digest gates for one arm
#             (env QMMCEIL_PIN_ARM)
# Run under flock -w 900 /tmp/m1-gpu.lock. stdout to files, never pipes.
set -euo pipefail

STAGE="${1:?stage: build|measure|pins}"
VENV_TPL="${2:?venv template path}"
MODEL="${3:?model path}"

ROOT=/var/tmp/qmmceil
SRC="$ROOT/src"
OUT="$ROOT/out"
mkdir -p "$OUT"
cd "$SRC"

resolve_commit() {
  if [[ -n "${MLX_OMARCHY_SOURCE_COMMIT:-}" ]]; then printf '%s' "$MLX_OMARCHY_SOURCE_COMMIT"; return; fi
  git -C "$SRC" rev-parse --short=7 HEAD 2>/dev/null || true
}
COMMIT="$(resolve_commit)"

libmlx_identity() {  # $1 = python
  "$1" - <<'PY'
import hashlib, re, sys
libs = set()
for line in open("/proc/self/maps", encoding="utf-8", errors="replace"):
    m = re.search(r"(/\S+/libmlx\.so\S*)$", line.strip())
    if m:
        libs.add(m.group(1))
for p in sorted(libs):
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    print(f"libmlx_identity path={p} sha256={h}")
import mlx.core as mx
print("mlx_version", mx.__version__)
PY
}

case "$STAGE" in
build)
  export MLX_OMARCHY_SOURCE_COMMIT="$COMMIT"
  export MLX_OMARCHY_WORK_DIR="$ROOT/work"
  bash scripts/build-wheel.sh --diagnostics > "$OUT/build-diag.log" 2>&1
  WHEEL=$(ls "$SRC"/dist/mlx_omarchy-*cp314*.whl | head -1)
  echo "wheel=$WHEEL" | tee "$OUT/wheel.txt"
  sha256sum "$WHEEL" > "$OUT/wheel.sha256"
  rm -rf "$ROOT/venv"
  cp -a "$VENV_TPL" "$ROOT/venv"
  "$ROOT/venv/bin/python" -m pip install --no-deps --no-index --force-reinstall \
    "$WHEEL" > "$OUT/venv-install.log" 2>&1
  "$ROOT/venv/bin/python" - <<'PY' > "$OUT/identity.txt" 2>&1
import mlx.core as mx, subprocess
print(mx.__version__)
PY
  libmlx_identity "$ROOT/venv/bin/python" >> "$OUT/identity.txt"
  tail -3 "$OUT/identity.txt"
  ;;

measure)
  PY="$ROOT/venv/bin/python"
  export HF_HUB_OFFLINE=1 MLX_DISABLE_COMPILE=1
  export MLX_OMARCHY_WORK_DIR="$ROOT/work"
  # Exact manifest prompts (short = 30 tokens, ctx1024 = 1053 tokens);
  # end='' so no trailing newline changes the chat-template token count.
  SHORT=$("$PY" -c "import sys; sys.path.insert(0,'$SRC/scripts'); import bench_matrix, json; print(bench_matrix.prompt_text(json.load(open('$SRC/scripts/bench_matrix.json')),'short'), end='')")
  CTX=$("$PY" -c "import sys; sys.path.insert(0,'$SRC/scripts'); import bench_matrix, json; print(bench_matrix.prompt_text(json.load(open('$SRC/scripts/bench_matrix.json')),'ctx1024'), end='')")
  # libmlx identity for this stage
  libmlx_identity "$PY" > "$OUT/identity-measure.txt"
  # Phase profiles: prefill + 2 decode tokens, GPU profile stream.
  for leg in short ctx; do
    P=$SHORT; [[ $leg == ctx ]] && P=$CTX
    MLX_OMARCHY_GPU_PROFILE="$OUT/profile-$leg.jsonl" \
      MLX_OMARCHY_GPU_PROFILE_LABEL="qmmceil-$leg" \
      MESA_SHADER_CACHE_DISABLE=true \
      "$PY" scripts/profile_generate.py --model "$MODEL" --prompt "$P" \
        --max-tokens 2 --temp 0 --seed 0 \
        --markers "$OUT/markers-$leg.jsonl" \
        > "$OUT/gen-$leg.log" 2>&1
  done
  # Kernel-isolated probe arms, one fresh process per arm.
  for arm in 0 3 4 5 6; do
    MLX_OMARCHY_QMM_COOP_BENCH=$arm \
      MESA_SHADER_CACHE_DISABLE=true \
      "$PY" scripts-local/qmm_coop_bench_probe.py \
      > "$OUT/probe-arm$arm.jsonl" 2> "$OUT/probe-arm$arm.err" || true
  done
  echo MEASURE-DONE
  ;;

pins)
  PY="$ROOT/venv/bin/python"
  ARM="${QMMCEIL_PIN_ARM:?QMMCEIL_PIN_ARM required}"
  export HF_HUB_OFFLINE=1 MLX_DISABLE_COMPILE=1
  P=$("$PY" -c "import sys; sys.path.insert(0,'$SRC/scripts'); import bench_matrix, json; print(bench_matrix.prompt_text(json.load(open('$SRC/scripts/bench_matrix.json')),'${QMMCEIL_PIN_PROMPT:?QMMCEIL_PIN_PROMPT required}'), end='')")
  MLX_OMARCHY_QMM_COOP_BENCH=$ARM \
    "$PY" scripts/bench_decode.py --model "$MODEL" --prompt "$P" \
    --tokens 32 --temp 0.0 --seed 0 --warmup-tokens 4 \
    > "$OUT/pin-$QMMCEIL_PIN_PROMPT-arm$ARM.json" 2>&1 || true
  tail -2 "$OUT/pin-$QMMCEIL_PIN_PROMPT-arm$ARM.json"
  echo PINS-DONE
  ;;
esac
