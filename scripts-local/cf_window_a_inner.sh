#!/bin/sh
# CoopmatTflopClose window A (inner): bench-arm build + functional gates
# + bit-identity + arm timings on jwm1. Runs under ONE top-level
# flock /tmp/m1-gpu.lock held by the wrapper; quiet gate announced in
# hub before the wrapper starts. Commit pinned by SHA argument.
set -eu
ulimit -c 0
CFSHA=$1
REPO=$HOME/src/mlx-omarchy-cfbench
OUT=$HOME/benchq/qmm-coop-bench
mkdir -p "$OUT"
cd "$REPO"
git fetch --quiet origin wave/CoopmatTflopClose
test "$(git rev-parse FETCH_HEAD)" = "$CFSHA" || { echo "SHA-MISMATCH $(git rev-parse FETCH_HEAD)"; exit 8; }
git checkout --quiet --detach "$CFSHA"
echo "[cfA] source at $(git rev-parse --short HEAD)"
bash scripts/prepare-mlx.sh > "$OUT/prepare.log" 2>&1
echo "[cfA] staged"

cmake -S .work/mlx -B .work/build-cf -G Ninja \
  -DMLX_BUILD_OMARCHY=ON -DCMAKE_BUILD_TYPE=Release \
  -DMLX_BUILD_TESTS=ON -DMLX_BUILD_EXAMPLES=OFF -DMLX_BUILD_BENCHMARKS=OFF \
  > "$OUT/configure.log" 2>&1
cmake --build .work/build-cf --target omarchy_matmul_family_tests -j8 \
  > "$OUT/build-tests.log" 2>&1
echo "[cfA] tests built"

# Functional gate 1: shipped kernel against the host reference bound.
AGX_SIMDMAT=1 timeout 900 .work/build-cf/tests/omarchy/omarchy_matmul_family_tests \
  --test-case="qmm coopmat prefill matches host reference at Qwen shapes" \
  > "$OUT/m1-qmm-coopmat-test.log" 2>&1 || echo "[cfA] WARN coopmat host-reference nonzero"
# Functional gates 2/3: digest-preserving bench arms must land inside
# the same host-reference bound through the full dispatch stack.
for arm in 1 2; do
  MLX_OMARCHY_QMM_COOP_BENCH=$arm AGX_SIMDMAT=1 \
    timeout 900 .work/build-cf/tests/omarchy/omarchy_matmul_family_tests \
    --test-case="qmm coopmat prefill matches host reference at Qwen shapes" \
    > "$OUT/m1-qmm-coopmat-arm$arm-test.log" 2>&1 || echo "[cfA] WARN arm$arm host-reference nonzero"
done
# Functional gate 4: shipped-kernel offset parity unchanged.
AGX_SIMDMAT=1 timeout 900 .work/build-cf/tests/omarchy/omarchy_matmul_family_tests \
  --test-case="qmm coopmat output is bit-identical across x offset alignment" \
  > "$OUT/m1-qmm-offset-parity.log" 2>&1 || echo "[cfA] WARN offset-parity nonzero"
grep -hE "test cases:|Status:" "$OUT"/m1-qmm-coopmat*-test.log "$OUT/m1-qmm-offset-parity.log" || true
echo "[cfA] functional gates done; building wheel"

sh scripts/build-wheel.sh > "$OUT/wheel-build.log" 2>&1
grep -E "receipt" "$OUT/wheel-build.log" | tail -3
cp dist/mlx_omarchy-*.whl "$OUT/"

VENV=$HOME/venv-qmmcoop
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

cp "$REPO/scripts-local/qmm_coop_bench_probe.py" "$OUT/"
echo "[cfA] arm probes (0..4), 4 warmups + 30 timed evals each"
for arm in 0 1 2 3 4; do
  MLX_OMARCHY_QMM_COOP_BENCH=$arm AGX_SIMDMAT=1 \
    "$VENV/bin/python" "$OUT/qmm_coop_bench_probe.py" \
    > "$OUT/probe-arm$arm.log" 2>&1
  tail -1 "$OUT/probe-arm$arm.log"
done

python3 - "$OUT" <<'PY'
import json, sys
out = sys.argv[1]
def load(arm):
    rows = {}
    for line in open(f"{out}/probe-arm{arm}.log"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if isinstance(r, dict) and "shape" in r:
            rows[r["shape"]] = r
    return rows
base = load(0)
verdict = {}
identity_ok = True
for arm in (1, 2):
    rows = load(arm)
    for shape, b in base.items():
        r = rows.get(shape)
        ok = r is not None and r["f16_digest"] == b["f16_digest"]
        identity_ok = identity_ok and ok
        verdict[f"arm{arm}:{shape}"] = {
            "bit_identical": ok,
            "digest": r["f16_digest"] if r else None,
            "base_digest": b["f16_digest"],
            "median_ms": r["median_ms"] if r else None,
            "base_ms": b["median_ms"],
        }
ceil = {}
for arm in (3, 4):
    rows = load(arm)
    for shape, b in base.items():
        r = rows.get(shape)
        ceil[f"arm{arm}:{shape}"] = {
            "median_ms": r["median_ms"] if r else None,
            "tflops": r["tflops"] if r else None,
        }
result = {
    "schema": "mlx-omarchy/qmm-coop-bench/1",
    "bit_identity_gate": "PASS" if identity_ok else "FAIL",
    "identity": verdict,
    "ceilings": ceil,
    "base": {s: {"median_ms": b["median_ms"], "tflops": b["tflops"],
                 "digest": b["f16_digest"]} for s, b in base.items()},
}
json.dump(result, open(f"{out}/bench-verdict.json", "w"), indent=1)
print("BENCH-GATE", result["bit_identity_gate"])
print(json.dumps(result["base"], indent=1))
print(json.dumps(result["ceilings"], indent=1))
PY
echo "[cfA] DONE"
