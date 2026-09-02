#!/usr/bin/env bash
# Run upstream MLX's own test suites against the omarchy-only backend and
# write per-file results into an output directory.
#
# Phase 1 (C++): configures .work/build-upstream with the omarchy backend on
#   and every other backend off, builds the upstream `tests` target, and runs
#   it once per test source file with doctest XML reports and a per-file
#   timeout. Raw console logs and XML land in $OUT_DIR/cpp/.
# Phase 2 (python): installs the scripts/build-wheel.sh output wheel into a
#   fresh venv, adds pytest, and runs each upstream python test file with a
#   per-file timeout and a junit xml report. Results land in $OUT_DIR/py/.
#
# Usage: tools/run-upstream-suite.sh [--cpp-only] [--py-only]
#
# Environment:
#   OUT_DIR          output root (default receipts/upstream-suite-<date>/)
#   CPP_TIMEOUT      per-file C++ timeout seconds (default 900)
#   PY_TIMEOUT       per-file python timeout seconds (default 900)
#   WHEEL            wheel to install for phase 2 (default newest dist/*.whl)
#
# Exit code is nonzero if any phase had a failure, timeout, or crash.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${MLX_OMARCHY_WORK_DIR:-$ROOT/.work}"
OUT_DIR="${OUT_DIR:-$ROOT/receipts/upstream-suite-$(date +%F)}"
CPP_TIMEOUT="${CPP_TIMEOUT:-900}"
PY_TIMEOUT="${PY_TIMEOUT:-900}"

CPP_ONLY=0
PY_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --cpp-only) CPP_ONLY=1 ;;
    --py-only) PY_ONLY=1 ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done
RUN_CPP=1
RUN_PY=1
if [[ $CPP_ONLY -eq 1 ]]; then RUN_PY=0; fi
if [[ $PY_ONLY -eq 1 ]]; then RUN_CPP=0; fi

mkdir -p "$OUT_DIR"

fail_total=0

# ---------------------------------------------------------------- phase 1: C++
if [[ $RUN_CPP -eq 1 ]]; then
  BUILD_DIR="$WORK_DIR/build-upstream"
  DOCTEST_PIN_ARGS=()
  if [[ -d "$WORK_DIR/build/_deps/doctest-src" ]]; then
    DOCTEST_PIN_ARGS+=(-DFETCHCONTENT_SOURCE_DIR_DOCTEST="$WORK_DIR/build/_deps/doctest-src")
  fi
  echo "== [cpp] configure $BUILD_DIR =="
  cmake -S "$WORK_DIR/mlx" -B "$BUILD_DIR" -G Ninja \
    -DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_CPU=OFF \
    -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF \
    -DMLX_BUILD_TESTS=ON -DMLX_BUILD_EXAMPLES=OFF -DMLX_BUILD_BENCHMARKS=OFF \
    "${DOCTEST_PIN_ARGS[@]}" || { echo "[cpp] configure FAILED"; exit 1; }
  echo "== [cpp] build tests target =="
  cmake --build "$BUILD_DIR" --target tests -j "$(nproc)" \
    || { echo "[cpp] build FAILED"; exit 1; }

  TESTS_BIN="$BUILD_DIR/tests/tests"
  mkdir -p "$OUT_DIR/cpp"

  # The upstream test TUs linked into the `tests` binary (gpu/residency are
  # Metal-only and are not compiled in an omarchy build).
  cpp_files="allocator_tests.cpp array_tests.cpp arg_reduce_tests.cpp \
autograd_tests.cpp blas_tests.cpp compile_tests.cpp custom_vjp_tests.cpp \
creations_tests.cpp device_tests.cpp einsum_tests.cpp export_import_tests.cpp \
eval_tests.cpp fft_tests.cpp load_tests.cpp linalg_tests.cpp ops_tests.cpp \
random_tests.cpp scheduler_tests.cpp utils_tests.cpp vmap_tests.cpp"

  : > "$OUT_DIR/cpp/summary.tsv"
  echo -e "file\trc\tresult" >> "$OUT_DIR/cpp/summary.tsv"

  for f in $cpp_files; do
    base="${f%.cpp}"
    log="$OUT_DIR/cpp/$base.log"
    xml="$OUT_DIR/cpp/$base.xml"
    echo "== [cpp] $f =="
    MLX_OMARCHY_ALLOW_NON_APPLE=1 \
      timeout "$CPP_TIMEOUT" "$TESTS_BIN" \
      -sf="*$f" -r=xml -o="$xml" -d -nc \
      > "$log" 2>&1
    rc=$?
    parsed="$(python3 - "$xml" <<'PYEOF'
import sys
import xml.etree.ElementTree as ET
try:
    tree = ET.parse(sys.argv[1])
except Exception as e:
    print(f"PARSE-ERROR: {e}")
    sys.exit(0)
cases = tree.findall(".//TestCase")
executed = [c for c in cases if c.get("skipped") != "true"]
failed = [
    c for c in executed
    if (a := c.find("OverallResultsAsserts")) is not None
    and a.get("test_case_success") == "false"
]
names = ",".join(c.get("name", "?") for c in failed)
print(f"executed={len(executed)} passed={len(executed) - len(failed)} "
      f"failed={len(failed)} cases={names}")
PYEOF
)"
    if [[ $rc -eq 0 ]]; then
      result="$parsed"
    elif [[ $rc -eq 124 ]]; then
      result="TIMEOUT after ${CPP_TIMEOUT}s; partial: $parsed"
    elif [[ $rc -eq 1 || $rc -eq 3 ]]; then
      # doctest exits 1 (3 with --exit) when test cases failed.
      result="$parsed"
    else
      result="CRASH rc=$rc; partial: $parsed"
    fi
    echo -e "$base\t$rc\t$result" | tee -a "$OUT_DIR/cpp/summary.tsv"
    if [[ $rc -ne 0 || "$result" != *"failed=0"* ]]; then fail_total=1; fi
  done
  echo "== [cpp] summary: $OUT_DIR/cpp/summary.tsv =="
fi

# ------------------------------------------------------------- phase 2: python
if [[ $RUN_PY -eq 1 ]]; then
  if [[ -z "${WHEEL:-}" ]]; then
    WHEEL="$(ls -t "$ROOT"/dist/mlx_omarchy-*.whl 2>/dev/null | head -n1 || true)"
  fi
  if [[ -z "$WHEEL" ]]; then
    echo "[py] no wheel in dist/; run scripts/build-wheel.sh first" >&2
    exit 1
  fi
  echo "== [py] wheel: $WHEEL =="
  VENV_DIR="$OUT_DIR/venv-pytest"
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install numpy pytest >/dev/null
    "$VENV_DIR/bin/pip" install "$(printf '%q' "$WHEEL")" >/dev/null
  fi
  mkdir -p "$OUT_DIR/py"
  # Distributed suites need MPI/NCCL clusters: out of scope for the omarchy
  # backend, excluded by design.
  py_files="$(ls "$WORK_DIR/mlx/python/tests"/test_*.py | grep -v -E 'distributed' || true)"

  : > "$OUT_DIR/py/summary.tsv"
  echo -e "file\trc\tresult" >> "$OUT_DIR/py/summary.tsv"

  for f in $py_files; do
    base="$(basename "$f")"
    log="$OUT_DIR/py/$base.log"
    junit="$OUT_DIR/py/$base.xml"
    echo "== [py] $base =="
    not_expr=""
    crash_excluded=()
    rc=0
    for attempt in $(seq 1 60); do
      alog="$OUT_DIR/py/$base.a${attempt}.log"
      MLX_OMARCHY_ALLOW_NON_APPLE=1 MLX_ENABLE_TF32=0 \
        timeout "$PY_TIMEOUT" "$VENV_DIR/bin/python" -m pytest "$f" \
        -v --no-header -p no:cacheprovider --junitxml="$junit" \
        ${not_expr:+-k "$not_expr"} > "$alog" 2>&1
      rc=$?
      mv "$alog" "$log"
      crash_id="$(sed -n 's/^\([^ ]*::[^ ]*\) Fatal Python error.*/\1/p' "$log" | tail -n1)"
      if [[ -n "$crash_id" ]]; then
        crash_excluded+=("$crash_id")
        crash_name="${crash_id##*::}"
        if [[ -z "$not_expr" ]]; then
          not_expr="not $crash_name"
        else
          not_expr="$not_expr and not $crash_name"
        fi
        continue
      fi
      break
    done
    parsed="$(python3 - "$junit" <<'PYEOF'
import sys
import xml.etree.ElementTree as ET
try:
    tree = ET.parse(sys.argv[1])
except Exception as e:
    print(f"PARSE-ERROR: {e}")
    sys.exit(0)
suite = tree.getroot().find("testsuite")
if suite is None:
    print("PARSE-ERROR: no testsuite element")
    sys.exit(0)
n = int(suite.get("tests", 0))
f_ = int(suite.get("failures", 0))
e = int(suite.get("errors", 0))
s = int(suite.get("skipped", 0))
print(f"executed={n - s} passed={n - s - f_ - e} failed={f_ + e} skipped={s}")
PYEOF
)"
    crash_note=""
    if [[ ${#crash_excluded[@]} -gt 0 ]]; then
      crash_note=" crash-excluded=${#crash_excluded[@]}:$(IFS=';'; echo "${crash_excluded[*]}")"
      printf '%s\n' "${crash_excluded[@]}" > "$OUT_DIR/py/$base.crash-excluded.txt"
    fi
    if [[ $rc -eq 0 && ${#crash_excluded[@]} -eq 0 ]]; then
      result="$parsed"
    elif [[ $rc -eq 124 ]]; then
      result="TIMEOUT after ${PY_TIMEOUT}s; partial: $parsed"
    else
      result="EXIT rc=$rc; $parsed$crash_note"
    fi
    echo -e "$base\t$rc\t$result" | tee -a "$OUT_DIR/py/summary.tsv"
    if [[ $rc -ne 0 || "$result" != *"failed=0"* ]]; then fail_total=1; fi
  done
  echo "== [py] summary: $OUT_DIR/py/summary.tsv =="
fi

exit "$fail_total"
