#!/bin/sh
# TCF-1: TapeCorruptionFix acceptance on jwm1 (branch tape-corruption-fix @ 6cc0c07).
set -eu
REPO="$HOME/src/mlx-omarchy-bqm1-build"
cd "$REPO"
git fetch --quiet origin tape-corruption-fix
git checkout --quiet --detach 6cc0c07
echo "testing commit $(git rev-parse --short HEAD)"
OUTDIR="$HOME/benchq/wheels/tcf"
REQS="$HOME/benchq/support-reqs-clean.txt"
grep -v -e "^mlx-omarchy" -e "^mlx-lm==" "$HOME/benchq/support-reqs.txt" > "$REQS"
if ! ls "$OUTDIR"/*.whl >/dev/null 2>&1; then
  sh scripts/build-wheel.sh 2>&1 | grep -E "source commit|sha256:"
  mkdir -p "$OUTDIR"
  cp dist/mlx_omarchy-*.whl "$OUTDIR"/
fi
VENV="$HOME/venv-bqm1-tcf"
"$VENV/bin/pip" install -q --no-cache-dir --force-reinstall "$OUTDIR"/*.whl
WHEEL=$(ls "$OUTDIR"/*.whl)
MEMBER=$(unzip -p "$WHEEL" mlx/lib/libmlx.so | sha256sum | cut -d" " -f1)
LIB="$VENV/lib/python3.14/site-packages/mlx/lib/libmlx.so"
LOADED=$(sha256sum "$LIB" | cut -d" " -f1)
echo "[gate] wheel=$(basename "$WHEEL")"
echo "[gate] member=$MEMBER"
echo "[gate] loaded=$LOADED"
[ "$MEMBER" = "$LOADED" ] || { echo "FATAL: payload mismatch" >&2; exit 8; }
"$VENV/bin/python" -c "import mlx.core as mx; print('mx.__version__', mx.__version__)"
Q4="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3"
P="What is the capital of France? Answer in one word."

echo "== LEG 1: shapeless-reuse unit probe =="
RC=0
env -u MLX_DISABLE_COMPILE -u MLX_OMARCHY_ALLOW_NON_APPLE \
  MLX_OMARCHY_ALLOW_UNSAFE_COMPILE=1 \
  "$VENV/bin/python" scripts/probe_shapeless_reuse.py \
  > "$HOME/benchq/logs/tcf-leg1.log" 2>&1 || RC=$?
echo "LEG1 rc=$RC"
cat "$HOME/benchq/logs/tcf-leg1.log"
if [ "$RC" -ne 0 ]; then
  echo "TCF-1 HALT: LEG 1 failed (fix incomplete)"
  exit 3
fi

echo "== LEG 2: baseline corruption check 5x =="
i=1
while [ "$i" -le 5 ]; do
  LOG="$HOME/benchq/logs/tcf-leg2.run$i.log"
  env -u MLX_DISABLE_COMPILE -u MLX_OMARCHY_ALLOW_NON_APPLE \
    MLX_OMARCHY_ALLOW_UNSAFE_COMPILE=1 \
    "$VENV/bin/python" -m mlx_lm generate \
    --model "$Q4" --prompt "$P" --max-tokens 32 --temp 0 --seed 0 \
    > "$LOG" 2>&1 || { echo "LEG2 run$i rc!=0"; tail -4 "$LOG"; exit 3; }
  grep -q "Paris" "$LOG" || { echo "LEG2 run$i: no Paris (corruption)"; tail -4 "$LOG"; exit 3; }
  grep -m1 -E "shapeless compiled fragment" "$LOG" || echo "LEG2 run$i: no tape-ran notice this run"
  echo "LEG2 run$i PASS"
  i=$((i + 1))
done

echo "== LEG 3a: 20 more France runs =="
i=6
while [ "$i" -le 25 ]; do
  LOG="$HOME/benchq/logs/tcf-leg3.run$i.log"
  env -u MLX_DISABLE_COMPILE -u MLX_OMARCHY_ALLOW_NON_APPLE \
    MLX_OMARCHY_ALLOW_UNSAFE_COMPILE=1 \
    "$VENV/bin/python" -m mlx_lm generate \
    --model "$Q4" --prompt "$P" --max-tokens 32 --temp 0 --seed 0 \
    > "$LOG" 2>&1 || { echo "LEG3 run$i rc!=0"; tail -4 "$LOG"; exit 3; }
  grep -q "Paris" "$LOG" || { echo "LEG3 run$i: no Paris"; tail -4 "$LOG"; exit 3; }
  echo "LEG3 run$i PASS"
  i=$((i + 1))
done

echo "== LEG 3b: differential harness =="
env -u MLX_DISABLE_COMPILE -u MLX_OMARCHY_ALLOW_NON_APPLE \
  MLX_OMARCHY_ALLOW_UNSAFE_COMPILE=1 \
  "$VENV/bin/python" scripts/probe_tape_eager.py \
  > "$HOME/benchq/logs/tcf-leg3-probe.log" 2>&1
echo "probe rc=$?"; tail -3 "$HOME/benchq/logs/tcf-leg3-probe.log"
env -u MLX_DISABLE_COMPILE -u MLX_OMARCHY_ALLOW_NON_APPLE \
  MLX_OMARCHY_ALLOW_UNSAFE_COMPILE=1 \
  "$VENV/bin/python" scripts/differential_compile.py --mode realpath \
  --model "$Q4" --prompt "What is the capital of France?" --steps 32 \
  --out-dir "$HOME/benchq/tcf-realpath" \
  > "$HOME/benchq/logs/tcf-leg3-realpath.log" 2>&1
echo "realpath rc=$?"; tail -3 "$HOME/benchq/logs/tcf-leg3-realpath.log"

echo "== LEG 3c: C++ batteries at 6cc0c07 =="
sh scripts/prepare-mlx.sh > /dev/null 2>&1
cmake -S .work/mlx -B .work/mlx/build -DMLX_BUILD_TESTS=ON -DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_CPU=ON -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF > /dev/null 2>&1
cmake --build .work/mlx/build --target omarchy_compiled_tape_tests omarchy_eq_math_tests omarchy_runtime_tests -j"$(nproc)" 2>&1 | tail -1
for T in omarchy_compiled_tape_tests omarchy_eq_math_tests omarchy_runtime_tests; do
  LOG="$HOME/benchq/logs/tcf-battery-$T.log"
  if MLX_OMARCHY_ALLOW_UNSAFE_COMPILE=1 .work/mlx/build/tests/omarchy/$T > "$LOG" 2>&1; then
    echo "$T: rc=0"; tail -3 "$LOG" | head -2
  else
    echo "$T: rc=$?"; tail -8 "$LOG"
  fi
done

echo "== LEG 4: poison regression check (separate env) =="
i=1
while [ "$i" -le 5 ]; do
  LOG="$HOME/benchq/logs/tcf-leg4.run$i.log"
  env -u MLX_DISABLE_COMPILE -u MLX_OMARCHY_ALLOW_NON_APPLE \
    MLX_OMARCHY_ALLOW_UNSAFE_COMPILE=1 MLX_OMARCHY_POISON_FREED=1 \
    "$VENV/bin/python" -m mlx_lm generate \
    --model "$Q4" --prompt "$P" --max-tokens 32 --temp 0 --seed 0 \
    > "$LOG" 2>&1 || { echo "LEG4 run$i rc!=0"; tail -4 "$LOG"; exit 4; }
  grep -q "Paris" "$LOG" || { echo "LEG4 run$i: no Paris"; tail -4 "$LOG"; exit 4; }
  echo "LEG4 run$i PASS"
  i=$((i + 1))
done
echo "TCF-1 COMPLETE: all legs green"
