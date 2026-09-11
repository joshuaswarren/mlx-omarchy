#!/bin/sh
# CoopmatIlpChains window A (inner): ILP-arm build + functional gates +
# bit-identity + chain-ladder and candidate timings + AGX dumps on jwm1.
# Runs under ONE top-level flock /tmp/m1-gpu.lock held by the wrapper.
# Commit pinned by SHA argument. Wall-anchored timing only.
set -eu
ulimit -c 0
CFSHA=$1
REPO=$HOME/src/mlx-omarchy-cfbench
OUT=$HOME/benchq/qmm-coop-bench
mkdir -p "$OUT"
cd "$REPO"
git fetch --quiet origin wave/CoopmatIlpChains
test "$(git rev-parse FETCH_HEAD)" = "$CFSHA" || { echo "SHA-MISMATCH $(git rev-parse FETCH_HEAD)"; exit 8; }
git checkout --quiet --detach "$CFSHA"
echo "[cfic] source at $(git rev-parse --short HEAD)"
bash scripts/prepare-mlx.sh > "$OUT/prepare.log" 2>&1
echo "[cfic] staged"

cmake -S .work/mlx -B .work/build-cfic -G Ninja \
  -DMLX_BUILD_OMARCHY=ON -DCMAKE_BUILD_TYPE=Release \
  -DMLX_BUILD_TESTS=ON -DMLX_BUILD_EXAMPLES=OFF -DMLX_BUILD_BENCHMARKS=OFF \
  > "$OUT/configure.log" 2>&1
cmake --build .work/build-cfic --target omarchy_matmul_family_tests -j8 \
  > "$OUT/build-tests.log" 2>&1
echo "[cfic] tests built"

# Functional gate 1: shipped kernel against the host reference bound.
AGX_SIMDMAT=1 timeout 900 .work/build-cfic/tests/omarchy/omarchy_matmul_family_tests \
  --test-case="qmm coopmat prefill matches host reference at Qwen shapes" \
  > "$OUT/m1-qmm-coopmat-test.log" 2>&1 || echo "[cfic] WARN coopmat host-reference nonzero"
# Functional gates 2-4: the digest-preserving ILP arms must land inside
# the same host-reference bound through the full dispatch stack, and the
# in-process arm bit-identity regression must pass.
AGX_SIMDMAT=1 timeout 900 .work/build-cfic/tests/omarchy/omarchy_matmul_family_tests \
  --test-case="qmm coopmat ILP schedule arms reproduce the shipped kernel bitwise" \
  > "$OUT/m1-qmm-ilp-arms-test.log" 2>&1 || echo "[cfic] WARN ilp-arms nonzero"
for arm in 8 9 10; do
  MLX_OMARCHY_QMM_COOP_BENCH=$arm AGX_SIMDMAT=1 \
    timeout 900 .work/build-cfic/tests/omarchy/omarchy_matmul_family_tests \
    --test-case="qmm coopmat prefill matches host reference at Qwen shapes" \
    > "$OUT/m1-qmm-coopmat-arm$arm-test.log" 2>&1 || echo "[cfic] WARN arm$arm host-reference nonzero"
done
# Functional gate 5: shipped-kernel offset parity unchanged.
AGX_SIMDMAT=1 timeout 900 .work/build-cfic/tests/omarchy/omarchy_matmul_family_tests \
  --test-case="qmm coopmat output is bit-identical across x offset alignment" \
  > "$OUT/m1-qmm-offset-parity.log" 2>&1 || echo "[cfic] WARN offset-parity nonzero"
grep -hE "test cases:|Status:" "$OUT"/m1-qmm-coopmat-test.log \
  "$OUT"/m1-qmm-coopmat-arm{8,9,10}-test.log "$OUT/m1-qmm-ilp-arms-test.log" \
  "$OUT/m1-qmm-offset-parity.log" || true
grep -hE "\[qmm-ilp-arms\]" "$OUT/m1-qmm-ilp-arms-test.log" || true
echo "[cfic] functional gates done; building wheel"

rm -f "$OUT"/mlx_omarchy-*.whl
sh scripts/build-wheel.sh > "$OUT/wheel-build.log" 2>&1
grep -E "receipt" "$OUT/wheel-build.log" | tail -3
cp dist/mlx_omarchy-*.whl "$OUT/"

VENV=$HOME/venv-qmmcoop-ilp
rm -rf "$VENV"
python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --no-cache-dir numpy
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
echo "[cfic] arm probes (0..10), 4 warmups + 30 timed evals each"
for arm in 0 1 2 3 4 5 6 7 8 9 10; do
  MLX_OMARCHY_QMM_COOP_BENCH=$arm AGX_SIMDMAT=1 \
    "$VENV/bin/python" "$OUT/qmm_coop_bench_probe.py" \
    > "$OUT/probe-arm$arm.log" 2>&1
  tail -1 "$OUT/probe-arm$arm.log"
done

# AGX dumps of the MulAdd loop for the shipped kernel, the double-buffer
# arm, and the 4-chain ladder rung: static chains-in-flight evidence.
for arm in 0 8 9 7; do
  MLX_OMARCHY_QMM_COOP_BENCH=$arm AGX_SIMDMAT=1 \
    AGX_MESA_DEBUG=shaders MESA_SHADER_CACHE_DISABLE=true \
    "$VENV/bin/python" - <<PY > "$OUT/agx-dump-arm$arm.log" 2>&1
import numpy as np
import mlx.core as mx

w = mx.random.normal((896, 896), key=mx.random.key(7)) * 0.5
w, s, b = mx.quantize(w.astype(mx.float32), 64, 4)
x = mx.random.normal((32, 896), key=mx.random.key(11)).astype(mx.float16)
out = mx.quantized_matmul(x, w, s.astype(mx.float16), b.astype(mx.float16), True, 64, 4)
mx.eval(out)
print("DUMP-DISPATCH-DONE")
PY
  tail -1 "$OUT/agx-dump-arm$arm.log"
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
for arm in (8, 9, 10):
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
            "tflops": r["tflops"] if r else None,
            "base_tflops": b["tflops"],
        }
ladder = {}
for arm in (3, 4, 5, 6, 7):
    rows = load(arm)
    for shape, b in base.items():
        r = rows.get(shape)
        ladder[f"arm{arm}:{shape}"] = {
            "median_ms": r["median_ms"] if r else None,
            "tflops": r["tflops"] if r else None,
        }
result = {
    "schema": "mlx-omarchy/qmm-coop-ilp-bench/1",
    "bit_identity_gate": "PASS" if identity_ok else "FAIL",
    "identity": verdict,
    "ladder": ladder,
    "base": {s: {"median_ms": b["median_ms"], "tflops": b["tflops"],
                 "digest": b["f16_digest"]} for s, b in base.items()},
}
json.dump(result, open(f"{out}/bench-verdict.json", "w"), indent=1)
print("BENCH-GATE", result["bit_identity_gate"])
print(json.dumps(result["base"], indent=1))
print(json.dumps(result["ladder"], indent=1))
print(json.dumps(result["identity"], indent=1))
PY
echo "[cfic] DONE"