#!/usr/bin/env bash
set -euo pipefail

: "${1:?usage: $0 ROOT EXPECTED_COMMIT LABEL VENV}"
: "${2:?usage: $0 ROOT EXPECTED_COMMIT LABEL VENV}"
: "${3:?usage: $0 ROOT EXPECTED_COMMIT LABEL VENV}"
: "${4:?usage: $0 ROOT EXPECTED_COMMIT LABEL VENV}"
: "${DEADLINE_EPOCH:?set DEADLINE_EPOCH to no more than 1800 seconds from now}"
ROOT=$(readlink -f "$1")
EXPECTED_COMMIT=$2
LABEL=$3
VENV=$(readlink -m "$4")
WORK_ROOT="$ROOT/.work-longctx"
BUILD_ROOT="$WORK_ROOT/build"
TMP_ROOT="$WORK_ROOT/tmp"
STATUS="$WORK_ROOT/$LABEL-build.status"

[ "$(hostname -s)" = jwm1-linux ]
[ "$(git -C "$ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT" ]
[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=no)" ]
[[ "$LABEL" =~ ^[a-z]+$ ]]
[[ "$DEADLINE_EPOCH" =~ ^[0-9]+$ ]]
[ "$VENV" = "$WORK_ROOT/$LABEL-venv" ]
[ ! -e "$STATUS" ]
now=$(date +%s)
remaining=$((DEADLINE_EPOCH - now))
(( remaining >= 60 && remaining <= 1800 ))
available_kb=$(df -Pk "$HOME" | awk 'NR == 2 {print $4}')
(( available_kb >= 20971520 ))
mkdir -p "$BUILD_ROOT" "$TMP_ROOT"
printf 'build_start=true label=%s pid=%s pgid=%s deadline_epoch=%s source=%s commit=%s home_available_kb=%s\n' \
  "$LABEL" "$$" "$(ps -o pgid= -p $$ | tr -d ' ')" "$DEADLINE_EPOCH" "$ROOT" \
  "$EXPECTED_COMMIT" "$available_kb"
ulimit -c 0
set +e
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  TMPDIR="$TMP_ROOT" MLX_OMARCHY_WORK_DIR="$BUILD_ROOT" \
  DEV_RELEASE=1 CMAKE_BUILD_PARALLEL_LEVEL=1 \
  timeout --foreground --signal=TERM --kill-after=10s "${remaining}s" \
  "$ROOT/scripts/build-wheel.sh"
build_rc=$?
set -e
printf 'build_rc=%s finished_epoch=%s\n' "$build_rc" "$(date +%s)" | tee "$STATUS"
[ "$build_rc" -eq 0 ]
shopt -s nullglob
short=${EXPECTED_COMMIT:0:7}
wheels=("$ROOT"/dist/mlx_omarchy-*+"$short"-*.whl)
shopt -u nullglob
[ "${#wheels[@]}" -eq 1 ]
wheel=${wheels[0]}
python3 -m venv --system-site-packages "$VENV"
"$VENV/bin/python" -m pip install --force-reinstall --no-deps "$wheel"
"$VENV/bin/python" - "$wheel" "$EXPECTED_COMMIT" "$ROOT" "$VENV" <<'PY'
import hashlib
import importlib.metadata
import json
import pathlib
import sys

wheel = pathlib.Path(sys.argv[1]).resolve()
commit = sys.argv[2]
root = pathlib.Path(sys.argv[3]).resolve()
venv = pathlib.Path(sys.argv[4]).resolve()
version = importlib.metadata.version("mlx-omarchy")
assert commit[:7] in version, version
print(json.dumps({
    "commit": commit,
    "python": str(venv / "bin/python"),
    "source": str(root),
    "version": version,
    "wheel": str(wheel),
    "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
}, sort_keys=True))
PY
printf 'build_complete=true label=%s\n' "$LABEL"
