#!/bin/sh
# Q4 1K prefill window 1 (inner): build + fork coopmat tests + schedule
# screen + attribution probes on jwm1. Must run under ONE top-level
# flock /tmp/m1-gpu.lock (see window1.sh wrapper). Quiet gate: machine
# free of other GPU work before the window was announced.
set -eu
ulimit -c 0
Q4SHA=$1
REPO=$HOME/src/mlx-omarchy-bqm1-build
OUT=$HOME/benchq/q4prefill-window1
mkdir -p "$OUT"
cd "$REPO"
git fetch --quiet origin wave/Q4PrefillLongCtx
test "$(git rev-parse FETCH_HEAD)" = "$Q4SHA" || { echo "SHA-MISMATCH $(git rev-parse FETCH_HEAD)"; exit 8; }
git checkout --quiet --detach "$Q4SHA"
echo "[window1] source at $(git rev-parse --short HEAD)"
sh scripts/prepare-mlx.sh > "$OUT/prepare.log" 2>&1
echo "[window1] staged"

cmake -S .work/mlx -B .work/build-q4 -G Ninja \
  -DMLX_BUILD_OMARCHY=ON -DCMAKE_BUILD_TYPE=Release \
  -DMLX_BUILD_TESTS=ON -DMLX_BUILD_EXAMPLES=OFF -DMLX_BUILD_BENCHMARKS=OFF \
  > "$OUT/configure.log" 2>&1
cmake --build .work/build-q4 --target omarchy_matmul_family_tests -j8 \
  > "$OUT/build-tests.log" 2>&1
echo "[window1] tests built"

AGX_SIMDMAT=1 timeout 900 .work/build-q4/tests/omarchy/omarchy_matmul_family_tests \
  --test-case="qmm coopmat prefill matches host reference at Qwen shapes" \
  > "$OUT/m1-qmm-coopmat-test.log" 2>&1 \
  || echo "[window1] WARN coopmat host-reference test nonzero"
AGX_SIMDMAT=1 timeout 900 .work/build-q4/tests/omarchy/omarchy_matmul_family_tests \
  --test-case="qmm coopmat output is bit-identical across x offset alignment" \
  > "$OUT/m1-qmm-offset-parity.log" 2>&1 \
  || echo "[window1] WARN offset-parity test nonzero"
grep -E "test cases:|Status:" "$OUT/m1-qmm-coopmat-test.log" "$OUT/m1-qmm-offset-parity.log" || true
echo "[window1] coopmat tests done; building wheel"

sh scripts/build-wheel.sh > "$OUT/wheel-build.log" 2>&1
grep -E "receipt" "$OUT/wheel-build.log" | tail -5
cp dist/mlx_omarchy-*.whl "$OUT/"

VENV=$HOME/venv-q4tile
rm -rf "$VENV"
python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --no-cache-dir "$OUT"/mlx_omarchy-*.whl
"$VENV/bin/pip" install -q --no-cache-dir --no-deps mlx-lm==0.31.3
WHEEL=$(ls "$OUT"/mlx_omarchy-*.whl)
MEMBER=$(unzip -p "$WHEEL" mlx/lib/libmlx.so | sha256sum | cut -d' ' -f1)
LIB="$VENV/lib/python3.14/site-packages/mlx/lib/libmlx.so"
LOADED=$(sha256sum "$LIB" | cut -d' ' -f1)
echo "[gate] wheel=$(basename "$WHEEL")"
echo "[gate] member=$MEMBER"
echo "[gate] loaded=$LOADED"
[ "$MEMBER" = "$LOADED" ] || { echo "FATAL payload mismatch"; exit 8; }
"$VENV/bin/python" -c "import mlx.core as mx; print('[gate] mx.__version__', mx.__version__)"

cp "$REPO/scripts-local/q4_prefill_probe.py" "$OUT/"
echo "[window1] qmm schedule screen (arms 0..3)"
for arm in 0 1 2 3; do
  MLX_OMARCHY_QMM_COOP_TILE=$arm AGX_SIMDMAT=1 \
    "$VENV/bin/python" "$OUT/q4_prefill_probe.py" qmm \
    > "$OUT/probe-qmm-arm$arm.log" 2>&1
  tail -1 "$OUT/probe-qmm-arm$arm.log"
done
echo "[window1] attribution probes (attn, norms, lmhead)"
MLX_OMARCHY_QMM_COOP_TILE=0 AGX_SIMDMAT=1 \
  "$VENV/bin/python" "$OUT/q4_prefill_probe.py" attn norms lmhead \
  > "$OUT/probe-attrib.log" 2>&1
tail -1 "$OUT/probe-attrib.log"
echo "[window1] DONE"
