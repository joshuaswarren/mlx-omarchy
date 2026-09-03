#!/bin/sh
# CompiledFailClosed verification at e600ad6+: auto-eager gate, no env vars.
cd ~/src/mlx-omarchy-bqm1-build
git fetch --quiet origin main
git checkout --quiet --detach origin/main
echo "testing commit $(git rev-parse --short HEAD)"
cd ~/src/mlx-omarchy-bqm1-build
OUTDIR="$HOME/benchq/wheels/gate2"
if ! ls "$OUTDIR"/*.whl >/dev/null 2>&1; then
  sh scripts/build-wheel.sh 2>&1 | grep "\[receipt\]"
  mkdir -p "$OUTDIR"
  cp dist/mlx_omarchy-*.whl "$OUTDIR"/
fi
REQS="$HOME/benchq/support-reqs-clean.txt"
VENV="$HOME/venv-bqm1-gate2"
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --no-cache-dir "$OUTDIR"/*.whl
  "$VENV/bin/pip" install -q --no-cache-dir --no-deps mlx-lm==0.31.3
  "$VENV/bin/pip" install -q --no-cache-dir -r "$REQS"
fi
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
CLEANENV="env -u MLX_DISABLE_COMPILE -u MLX_OMARCHY_ALLOW_NON_APPLE -u MLX_OMARCHY_ALLOW_UNSAFE_COMPILE"

echo "== CHECK 1 (FLAGSHIP): no env vars, mlx_lm generate =="
$CLEANENV "$VENV/bin/python" -m mlx_lm generate \
  --model "$Q4" --prompt "$P" --max-tokens 32 --temp 0 --seed 0 \
  > "$HOME/benchq/logs/gate2-check1.out" 2> "$HOME/benchq/logs/gate2-check1.err" \
  || echo "CHECK1 rc=$?"
echo "stdout text: $(sed -n "2,3p" "$HOME/benchq/logs/gate2-check1.out")"
echo "warning count: $(grep -c "Compiled tapes are disabled on this Apple GPU" "$HOME/benchq/logs/gate2-check1.err" || true)"
grep -n "Compiled tapes are disabled" "$HOME/benchq/logs/gate2-check1.err" | head -2
grep -c "Traceback" "$HOME/benchq/logs/gate2-check1.err" || echo "no exception"

echo "== CHECK 2: compile ordering probe =="
$CLEANENV "$VENV/bin/python" scripts/probe_compile_ordering.py 2>&1 | tail -12 || echo "CHECK2 rc=$?"

echo "== CHECK 3: backstop refusal after enable_compile =="
$CLEANENV "$VENV/bin/python" - <<'EOF' 2>&1 | tail -4
import mlx.core as mx
mx.enable_compile()
def f(x):
    return mx.sigmoid(mx.broadcast_to(x, (2, x.shape[0])))
try:
    print("out:", mx.compile(f)(mx.array([1.0, 2.0, 3.0])).tolist())
except RuntimeError as exc:
    print("REFUSED:", str(exc)[:140])
EOF

echo "== CHECK 4: battery with override =="
cd ~/src/mlx-omarchy-bqm1-build
cmake -S .work/mlx -B .work/mlx/build -DMLX_BUILD_TESTS=ON -DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_CPU=ON -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF > /dev/null 2>&1
cmake --build .work/mlx/build --target omarchy_compiled_tape_tests -j"$(nproc)" 2>&1 | tail -1
if MLX_OMARCHY_ALLOW_UNSAFE_COMPILE=1 .work/mlx/build/tests/omarchy/omarchy_compiled_tape_tests \
    > "$HOME/benchq/logs/gate2-tapetests-override.log" 2>&1; then echo "WITH override rc=0"; else echo "WITH override rc=$?"; fi
tail -4 "$HOME/benchq/logs/gate2-tapetests-override.log"
if .work/mlx/build/tests/omarchy/omarchy_compiled_tape_tests \
    > "$HOME/benchq/logs/gate2-tapetests-nooverride.log" 2>&1; then echo "WITHOUT override rc=0"; else echo "WITHOUT override rc=$? (fail-closed case passing is expected)"; fi
tail -4 "$HOME/benchq/logs/gate2-tapetests-nooverride.log"

echo "== eager unaffected =="
if MLX_DISABLE_COMPILE=1 "$VENV/bin/python" -m mlx_lm generate \
    --model "$Q4" --prompt "$P" --max-tokens 32 --temp 0 --seed 0 \
    > "$HOME/benchq/logs/gate2-eager.log" 2>&1; then
  grep -q "Paris" "$HOME/benchq/logs/gate2-eager.log" && echo "EAGER: clean Paris" || echo "EAGER: no Paris"
else
  echo "EAGER: FAILED"
fi
