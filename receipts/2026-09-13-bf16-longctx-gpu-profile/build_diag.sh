#!/usr/bin/env bash
set -euo pipefail

BASE="$HOME/src/mlx-bf16-grouped-candidate-6b1ac029"
PROFILE="$HOME/src/mlx-bf16-grouped-profile-6b1ac029"
EXPECTED=6b1ac0296ba65a8e0075171ca9451e222ddff06b
DEADLINE_EPOCH=${DEADLINE_EPOCH:?set the parent-granted total deadline epoch}
LOG=${BUILD_LOG:-$HOME/LongContextCostAttribution-diag-build.log}

now=$(date +%s)
remaining=$((DEADLINE_EPOCH - now - 240))
[ "$remaining" -gt 0 ] || { echo "FATAL no build time remains before the reserved 240s measurement window"; exit 2; }
[ "$remaining" -le 600 ] || remaining=600
[ -e "$BASE/.git" ] || { echo "FATAL missing base worktree $BASE"; exit 2; }
[ "$(git -C "$BASE" rev-parse HEAD)" = "$EXPECTED" ] || { echo "FATAL base commit mismatch"; exit 3; }

if [ ! -e "$PROFILE/.git" ]; then
  git -C "$BASE" worktree add --detach "$PROFILE" "$EXPECTED"
fi
[ "$(git -C "$PROFILE" rev-parse HEAD)" = "$EXPECTED" ] || { echo "FATAL profile commit mismatch"; exit 3; }
[ -z "$(git -C "$PROFILE" status --porcelain)" ] || { echo "FATAL profile worktree is dirty"; exit 3; }

source "$PROFILE/mlx.lock"
mkdir -p "$PROFILE/.work"
archive="mlx-$MLX_VERSION-$MLX_COMMIT.tar.gz"
if [ ! -f "$PROFILE/.work/$archive" ]; then
  [ -f "$BASE/.work/$archive" ] || { echo "FATAL reusable MLX archive missing from $BASE/.work"; exit 2; }
  cp --reflink=auto "$BASE/.work/$archive" "$PROFILE/.work/$archive"
fi
printf '%s  %s\n' "$MLX_ARCHIVE_SHA256" "$PROFILE/.work/$archive" | sha256sum --check --status

echo "build_source=$EXPECTED"
echo "build_jobs=2"
echo "build_pid_pending=true deadline_epoch=$DEADLINE_EPOCH timeout_seconds=$remaining log=$LOG"
(
  cd "$PROFILE"
  timeout --foreground -k 10s "${remaining}s" env \
    CMAKE_BUILD_PARALLEL_LEVEL=2 \
    MLX_OMARCHY_WORK_DIR="$PROFILE/.work" \
    scripts/build-wheel.sh --diagnostics
) > "$LOG" 2>&1 &
build_pid=$!
echo "build_pid=$build_pid deadline_epoch=$DEADLINE_EPOCH log=$LOG"
set +e
wait "$build_pid"
rc=$?
set -e
echo "build_wait_rc=$rc"
[ "$rc" -eq 0 ] || { tail -n 80 "$LOG"; exit "$rc"; }

shopt -s nullglob
wheels=("$PROFILE"/dist/mlx_omarchy-*+diag.6b1ac02-*.whl)
shopt -u nullglob
[ "${#wheels[@]}" -eq 1 ] || { echo "FATAL expected one exact diagnostic wheel, got ${#wheels[@]}"; exit 4; }
wheel=${wheels[0]}
diag_site="$PROFILE/.work/diag-site"
rm -rf "$diag_site"
"$BASE/.work/venv-run/bin/python" -m pip install --no-deps --target "$diag_site" "$wheel"
PYTHONPATH="$diag_site" "$BASE/.work/venv-run/bin/python" - "$wheel" <<'PY'
import hashlib
import importlib.metadata
import json
import sys
import zipfile
from pathlib import Path
wheel = Path(sys.argv[1])
with zipfile.ZipFile(wheel) as archive:
    members = {name: hashlib.sha256(archive.read(name)).hexdigest()
               for name in archive.namelist() if name.endswith(".so")}
    profiling = any(b"MLX_OMARCHY_GPU_PROFILE" in archive.read(name)
                    for name in archive.namelist())
version = importlib.metadata.version("mlx-omarchy")
assert "+diag.6b1ac02" in version, version
assert profiling
print(json.dumps({
    "diagnostic_version": version,
    "diagnostic_wheel": str(wheel),
    "diagnostic_wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
    "profiling_literal_present": profiling,
    "shared_object_member_sha256": members,
}, sort_keys=True))
PY

echo "DIAGNOSTIC_BUILD_READY wheel=$wheel diag_site=$diag_site"
