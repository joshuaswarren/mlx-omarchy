#!/bin/sh
# CoopmatIlpChains window A2 (inner): re-gate and re-measure arms 9/10
# after the step-layout stride fix (ce14a392). Same flow as window A,
# reduced to the affected arms. One top-level flock held by the wrapper.
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
echo "[cfic2] source at $(git rev-parse --short HEAD)"
bash scripts/prepare-mlx.sh > "$OUT/prepare-a2.log" 2>&1
echo "[cfic2] staged"

cmake -S .work/mlx -B .work/build-cfic -G Ninja \
  -DMLX_BUILD_OMARCHY=ON -DCMAKE_BUILD_TYPE=Release \
  -DMLX_BUILD_TESTS=ON -DMLX_BUILD_EXAMPLES=OFF -DMLX_BUILD_BENCHMARKS=OFF \
  > "$OUT/configure-a2.log" 2>&1
cmake --build .work/build-cfic --target omarchy_matmul_family_tests -j8 \
  > "$OUT/build-tests-a2.log" 2>&1
echo "[cfic2] tests built"

# In-process bit-identity regression: arms 8/9/10 vs shipped, both shapes,
# plus host-reference bound.
AGX_SIMDMAT=1 timeout 900 .work/build-cfic/tests/omarchy/omarchy_matmul_family_tests \
  --test-case="qmm coopmat ILP schedule arms reproduce the shipped kernel bitwise" \
  > "$OUT/m1-qmm-ilp-arms-a2-test.log" 2>&1 || echo "[cfic2] WARN ilp-arms nonzero"
for arm in 9 10; do
  MLX_OMARCHY_QMM_COOP_BENCH=$arm AGX_SIMDMAT=1 \
    timeout 900 .work/build-cfic/tests/omarchy/omarchy_matmul_family_tests \
    --test-case="qmm coopmat prefill matches host reference at Qwen shapes" \
    > "$OUT/m1-qmm-coopmat-arm$arm-a2-test.log" 2>&1 || echo "[cfic2] WARN arm$arm host-reference nonzero"
done
grep -hE "test cases:|Status:" "$OUT/m1-qmm-ilp-arms-a2-test.log" \
  "$OUT"/m1-qmm-coopmat-arm{9,10}-a2-test.log || true
grep -hE "\[qmm-ilp-arms\]" "$OUT/m1-qmm-ilp-arms-a2-test.log" || true
echo "[cfic2] functional gates done; building wheel"

rm -f "$OUT"/mlx_omarchy-*.whl
sh scripts/build-wheel.sh > "$OUT/wheel-build-a2.log" 2>&1
grep -E "receipt" "$OUT/wheel-build-a2.log" | tail -3
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

cp "$REPO/scripts-local/qmm_coop_bench_probe.py" "$OUT/"
echo "[cfic2] arm probes (0, 8, 9, 10), 4 warmups + 30 timed evals each"
for arm in 0 8 9 10; do
  MLX_OMARCHY_QMM_COOP_BENCH=$arm AGX_SIMDMAT=1 \
    "$VENV/bin/python" "$OUT/qmm_coop_bench_probe.py" \
    > "$OUT/probe-a2-arm$arm.log" 2>&1
  tail -1 "$OUT/probe-a2-arm$arm.log"
done

# Fresh AGX dump of the fixed arm 9 loop.
MLX_OMARCHY_QMM_COOP_BENCH=9 AGX_SIMDMAT=1 \
  AGX_MESA_DEBUG=shaders MESA_SHADER_CACHE_DISABLE=true \
  "$VENV/bin/python" - <<PY > "$OUT/agx-dump-a2-arm9.log" 2>&1
import mlx.core as mx

w = mx.random.normal((896, 896), key=mx.random.key(7)) * 0.5
w, s, b = mx.quantize(w.astype(mx.float32), 64, 4)
x = mx.random.normal((32, 896), key=mx.random.key(11)).astype(mx.float16)
out = mx.quantized_matmul(x, w, s.astype(mx.float16), b.astype(mx.float16), True, 64, 4)
mx.eval(out)
print("DUMP-DISPATCH-DONE")
PY
tail -1 "$OUT/agx-dump-a2-arm9.log"

python3 - "$OUT" <<'PY'
import json, sys
out = sys.argv[1]
def load(tag, arm):
    rows = {}
    for line in open(f"{out}/probe-{tag}-arm{arm}.log"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if isinstance(r, dict) and "shape" in r:
            rows[r["shape"]] = r
    return rows
base = load("a2", 0)
identity_ok = True
res = {}
for arm in (8, 9, 10):
    rows = load("a2", arm)
    for shape, b in base.items():
        r = rows.get(shape)
        ok = r is not None and r["f16_digest"] == b["f16_digest"]
        identity_ok = identity_ok and ok
        res[f"arm{arm}:{shape}"] = {
            "bit_identical": ok,
            "median_ms": r["median_ms"] if r else None,
            "base_ms": b["median_ms"],
            "tflops": r["tflops"] if r else None,
            "base_tflops": b["tflops"],
        }
result = {
    "schema": "mlx-omarchy/qmm-coop-ilp-bench-a2/1",
    "bit_identity_gate": "PASS" if identity_ok else "FAIL",
    "cells": res,
    "base": {s: {"median_ms": b["median_ms"], "tflops": b["tflops"],
                 "digest": b["f16_digest"]} for s, b in base.items()},
}
json.dump(result, open(f"{out}/bench-a2-verdict.json", "w"), indent=1)
print("A2-BENCH-GATE", result["bit_identity_gate"])
for k, v in res.items():
    print(k, v.get("tflops"), "vs", v.get("base_tflops"),
          "ident" if v.get("bit_identical") else "DIFF")
PY
echo "[cfic2] DONE"