#!/bin/sh
# CoopmatTflopClose window A6 (inner): probe-only. The wheel built and
# gated by window A run 4 (commit 1aa2999b: fixed arms, all four
# functional gates SUCCESS) already sits in $OUT; build nothing, just
# install it, verify the payload hash, and run the five probe arms
# plus the digest gate. ~4 min of GPU time.
set -eu
ulimit -c 0
OUT=$HOME/benchq/qmm-coop-bench
VENV=$HOME/venv-qmmcoop
WHEEL=$(ls "$OUT"/mlx_omarchy-*.whl)
MEMBER=$(unzip -p "$WHEEL" mlx/lib/libmlx.so | sha256sum | cut -d' ' -f1)
echo "[cfA6] wheel=$(basename "$WHEEL")"
echo "[cfA6] member=$MEMBER"
# Payload hash recorded by the run-4 window (functional gates passed on
# this exact binary: default + arm1 + arm2 host-reference, offset parity).
test "$MEMBER" = "c33fcd1e1775f4f8366018273f89dde0d1e3eef85d29c6dbdc1a4cc9ca944079" \
  || { echo "FATAL wheel hash drift"; exit 8; }

rm -rf "$VENV"
python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --no-cache-dir numpy
"$VENV/bin/pip" install -q --no-cache-dir "$WHEEL"
"$VENV/bin/pip" install -q --no-cache-dir --no-deps mlx-lm==0.31.3
LIB="$VENV/lib/python3.14/site-packages/mlx/lib/libmlx.so"
LOADED=$(sha256sum "$LIB" | cut -d' ' -f1)
echo "[gate] loaded=$LOADED"
[ "$MEMBER" = "$LOADED" ] || { echo "FATAL payload mismatch"; exit 8; }
"$VENV/bin/python" -c "import mlx.core as mx; print('[gate] mx.__version__', mx.__version__)"

cp "$HOME/src/mlx-omarchy-cfbench/scripts-local/qmm_coop_bench_probe.py" "$OUT/" \
  || cp "$OUT/qmm_coop_bench_probe.py" "$OUT/qmm_coop_bench_probe.py"
echo "[cfA6] arm probes (0..4), 4 warmups + 30 timed evals each"
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
print(json.dumps(result, indent=1))
PY
echo "[cfA6] DONE"
