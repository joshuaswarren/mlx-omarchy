#!/usr/bin/env bash
# Window B2 stage 1 on jwm1-linux (split per Main's re-sequencing):
#   CPU prep OUTSIDE the lock: candidate wheel build, fresh venvs, suite
#   binaries built twice (base shader from mlx-main-b6d662a8, cand shader
#   from the branch).
#   GPU work under a SHORT capped flock: driver gate + the 144-cell probe
#   flags sweep gate. Lock released immediately after.
# Stage 2 (m1_b2_stage2.sh) re-acquires later for suites+attribution+matrix
# after Q4GemvNativeMapping and PythonHostParity take their windows.
# Waits are capped: flock -w 1800. Never hold this lock across a build.
set -uo pipefail
cd ~/src/mlx-Bf16PrefillClose
R=receipts/2026-09-10-bf16-prefill-close
mkdir -p "$R/m1-logs"
base_wheel="$HOME/src/mlx-main-b6d662a8/dist/mlx_omarchy-0.32.2.dev202609101916+b6d662a-cp314-cp314-linux_aarch64.whl"
echo "98821134f4bcf306ab1d64d6885ddf3d3e8e932311283bbd11f97aecf6a8e439  $base_wheel" \
  | sha256sum -c - || { echo "FATAL: base wheel mismatch"; exit 1; }

echo "== stage1 phase A: candidate wheel build (niced, -j4, OUTSIDE lock) =="
echo "${CAND_COMMIT:-unknown}" | tee "$R/m1-logs/cand-commit.txt"
nice -n 10 env DEV_RELEASE=1 CMAKE_BUILD_PARALLEL_LEVEL=4 \
  MLX_OMARCHY_SOURCE_COMMIT="${CAND_COMMIT:-local}" \
  scripts/build-wheel.sh 2>&1 | tail -5 | tee "$R/m1-logs/build-wheel-cand2.log"
cand_wheel=$(ls -t dist/mlx_omarchy-*.whl | head -1)
cp "$cand_wheel" "$R/wheels/candidate/"
sha256sum "$cand_wheel" | tee "$R/m1-logs/candidate-wheel.sha256"
realpath "$cand_wheel" | tee "$R/m1-logs/cand-wheel-path.txt"

echo "== stage1 phase B: fresh venvs (outside lock) =="
rm -rf .venv-prefill-base .venv-prefill-cand
python3.14 -m venv .venv-prefill-base
.venv-prefill-base/bin/pip install --quiet "$base_wheel" mlx-lm==0.31.3
python3.14 -m venv .venv-prefill-cand
.venv-prefill-cand/bin/pip install --quiet "$cand_wheel" mlx-lm==0.31.3

echo "== stage1 phase C: suite binaries, base AND cand shader (outside lock) =="
SHAD=.work/mlx/mlx/backend/omarchy/shaders/matmul_coopmat_bf16.comp
BASE_SHAD="$HOME/src/mlx-main-b6d662a8/overlay/mlx/backend/omarchy/shaders/matmul_coopmat_bf16.comp"
[[ -f "$BASE_SHAD" ]] || { echo "FATAL: base shader source missing"; exit 3; }
sha256sum "$SHAD" "$BASE_SHAD" | tee "$R/m1-logs/shader-hashes-b2.txt"
cmp -s "$SHAD" "$BASE_SHAD" && {
  echo "FATAL: cand shader == base shader; sync failed"; exit 3; }
cp "$SHAD" /tmp/cand-bf16.comp.keep
build_suites() {
  nice -n 10 cmake --build .work/build-prefill -j4 --target \
    omarchy_matmul_family_tests omarchy_runtime_tests mlx-omarchy-info \
    2>&1 | tail -2
}
cp "$BASE_SHAD" "$SHAD"
build_suites | tee "$R/m1-logs/build-suite-base.log"
cp .work/build-prefill/tests/omarchy/omarchy_matmul_family_tests /tmp/fam-base
cp .work/build-prefill/tests/omarchy/omarchy_runtime_tests /tmp/rt-base
cp /tmp/cand-bf16.comp.keep "$SHAD"
build_suites | tee "$R/m1-logs/build-suite-cand.log"
cp .work/build-prefill/tests/omarchy/omarchy_matmul_family_tests /tmp/fam-cand
cp .work/build-prefill/tests/omarchy/omarchy_runtime_tests /tmp/rt-cand
cmp -s /tmp/fam-base /tmp/fam-cand && {
  echo "FATAL: base and cand suite binaries identical"; exit 3; }
sha256sum /tmp/fam-base /tmp/fam-cand /tmp/rt-base /tmp/rt-cand \
  | tee "$R/m1-logs/suite-binaries.sha256"

echo "== stage1 phase D: driver gate + probe flags sweep (capped flock) =="
exec 9>/tmp/m1-gpu.lock
echo "$(date -Is) waiting for /tmp/m1-gpu.lock (cap 1800s)"
flock -w 1800 9 || { echo "FATAL: lock wait exceeded 1800s"; exit 4; }
echo "$(date -Is) lock acquired"
{
  g++ -std=c++17 -O2 -o /tmp/bf16-prefill-bench tools/bf16-prefill-bench/bench.cpp \
    || { echo "FATAL: probe build"; exit 3; }
  /tmp/bf16-prefill-bench --tiny 2>&1 | grep '"k":"dev"' \
    | tee "$R/m1-logs/driver-gate-b2.txt"
  grep -q '"coopmat":true' "$R/m1-logs/driver-gate-b2.txt" || {
    echo "FATAL: default driver lacks cooperative matrix; bailing."
    exit 3
  }
  echo "== stage1 phase E: probe flags sweep gate (144 cells) =="
  /tmp/bf16-prefill-bench > "$R/m1-logs/probe-flags-sweep.ndjson" \
    2> "$R/m1-logs/probe-flags-sweep.err"
  python3 - "$R/m1-logs/probe-flags-sweep.ndjson" <<'PY'
import json, sys
bad = 0; n = 0; flags = set()
for line in open(sys.argv[1]):
    try:
        r = json.loads(line)
    except Exception:
        continue
    if r.get("k") == "correct" and "cand_vs_base_mismatch" in r:
        n += 1; flags.add(r["flags"])
        if r["cand_vs_base_mismatch"] or r["dead_rows"]:
            bad += 1; print("BAD", r)
print(f"probe cells={n} flags={sorted(flags)} bad={bad}")
sys.exit(0 if (bad == 0 and n >= 36 and flags == {1}) else 1)
PY
  [[ $? -eq 0 ]] || { echo "FATAL: probe flags sweep failed"; exit 3; }
  echo "STAGE1-OK"
} > "$R/window-b2-stage1.log" 2>&1
rc=$?
echo "$(date -Is) stage 1 complete rc=$rc (lock released)"
exit $rc
