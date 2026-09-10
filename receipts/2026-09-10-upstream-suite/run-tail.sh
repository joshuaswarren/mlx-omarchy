#!/usr/bin/env bash
# Continue the baseline py suite after the fast_sdpa crash-stop.
# Same flags/format as tools/run-upstream-suite.sh; never stops early.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT_DIR="$SCRIPT_DIR/baseline-llvmpipe"
VENV_DIR="${PYTEST_VENV:-$ROOT/.work/venv-upstream-suite}"
export MLX_OMARCHY_ALLOW_NON_APPLE=1
export MLX_ENABLE_TF32=0
for base in "$@"; do
  f="$ROOT/.work/mlx/python/tests/$base"
  log="$OUT_DIR/py/$base.log"
  junit="$OUT_DIR/py/$base.xml"
  rm -f "$junit"
  echo "== [py-tail] $base =="
  timeout 3600 "$VENV_DIR/bin/python" -X faulthandler -m pytest "$f" \
    -v --no-header -p no:cacheprovider --junitxml="$junit" > "$log" 2>&1
  rc=$?
  parsed="$(python3 - "$junit" <<'PYEOF'
import sys
import xml.etree.ElementTree as ET
try:
    root = ET.parse(sys.argv[1]).getroot()
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    if not suites:
        raise ValueError("no testsuite elements")
    counts = [sum(int(suite.get(key, 0)) for suite in suites)
              for key in ("tests", "failures", "errors", "skipped")]
    n, failures, errors, skipped = counts
    if min(counts) < 0 or failures + errors + skipped > n:
        raise ValueError("inconsistent test counts")
    print(f"executed={n - skipped} passed={n - skipped - failures - errors} "
          f"failed={failures + errors} skipped={skipped}")
except (OSError, ET.ParseError, ValueError) as exc:
    print(f"INVALID-REPORT: {exc}")
PYEOF
)"
  printf '%s\t%s\t%s\n' "$base" "$rc" "$parsed" | tee -a "$OUT_DIR/py/summary.tsv"
done
