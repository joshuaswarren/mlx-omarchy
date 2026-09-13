#!/usr/bin/env bash
set -euo pipefail

SOURCE_COMMIT=57cc36a2e8ece78dd979b5344752d04c76b4d49d
LIBANE_COMMIT=f261a6cb537aca62f267ad3d01beda0d6877544c
EXPECTED_VERSION=
WHEEL_NAME=
WHEEL_SHA_EXPECTED=
SOURCE_BUNDLE_SHA=aebef2154ed0b0c543d796c4d68f52ef3670b2826e17879237dc762f589bf7af
LIBANE_BUNDLE_SHA=a2e33cb7deadba974a5d89e669a8982ea0ed40483d9c87b18688d2c0ff599727
ADD_MANIFEST_SHA=f974e82eca49726a9eafb58f95a6578a458d423bab12efcdc3c3b780f1bbf680
CHAIN_MANIFEST_SHA=065f2acd0a0668c0911fcc728bc8bfef56812e0e64c0064de6287ac29dc98619
ROOT=/var/tmp/AneRuntimeBridge-57cc36a2
VULKANINFO=/home/joshuawarren/.local/bin/vulkaninfo
SOURCE="$ROOT/source"
LIBANE="$ROOT/libane"
WHEEL=
VENV="$ROOT/install/venv"
ADD_BUNDLE="$ROOT/add"
CHAIN_BUNDLE="$ROOT/add-mul"
LEASE=/tmp/m1-gpu.lease.AneRuntimeBridge
QUARANTINE=/run/lock/mlx-omarchy-ane/quarantine
DEVICE_LOCK=/run/lock/mlx-omarchy-ane/device.lock
IDENTITY_FILE="$ROOT/installed-identity.tsv"
WHEEL_PATH_FILE="$ROOT/wheel-path.txt"
: "${ACQUIRE_CUTOFF_EPOCH:?ACQUIRE_CUTOFF_EPOCH is required}"
: "${OUTER_DEADLINE_EPOCH:?OUTER_DEADLINE_EPOCH is required}"
CLEANUP_RESERVE_SECONDS=60
SESSION_DEADLINE_EPOCH=0
DEADLINE_EPOCH=0
PGID=""
GUARDIAN_PID=""
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}

remaining_seconds() {
  echo $((DEADLINE_EPOCH - $(date +%s)))
}

require_time() {
  local phase=$1 required=$2 remaining
  remaining=$(remaining_seconds)
  if (( remaining < required )); then
    echo "deadline_guard=failed phase=$phase remaining_seconds=$remaining required_seconds=$required"
    exit 124
  fi
  echo "deadline_guard=pass phase=$phase remaining_seconds=$remaining required_seconds=$required"
}

run_with_window() {
  local remaining
  remaining=$(remaining_seconds)
  (( remaining > 0 )) || { echo "absolute_deadline_expired=true"; exit 124; }
  (( remaining <= 90 )) || remaining=90
  timeout --signal=TERM --kill-after=10s "${remaining}s" "$@"
}

run_as_user() {
  local remaining
  remaining=$(remaining_seconds)
  (( remaining > 0 )) || { echo "absolute_deadline_expired=true"; exit 124; }
  (( remaining <= 90 )) || remaining=90
  timeout --signal=TERM --kill-after=10s "${remaining}s" \
    sudo -n -u "$INSTALL_USER" -- "$@"
}

run_long_with_window() {
  local remaining
  remaining=$(remaining_seconds)
  (( remaining > 0 )) || { echo "absolute_deadline_expired=true"; exit 124; }
  timeout --signal=TERM --kill-after=10s "${remaining}s" "$@"
}

workers() {
  local rc
  if pgrep -af '^.*/mlx-omarchy-ane-worker([[:space:]]|$)'; then
    return 0
  else
    rc=$?
    (( rc == 1 )) && return 0
    return "$rc"
  fi
}

capture_guardian() {
  local guardian_comm guardian_pgid
  GUARDIAN_PID=$PPID
  [[ "$GUARDIAN_PID" =~ ^[0-9]+$ ]]
  IFS= read -r guardian_comm <"/proc/$GUARDIAN_PID/comm"
  [[ "$guardian_comm" == timeout ]] || {
    echo "guardian_identity=false pid=$GUARDIAN_PID comm=$guardian_comm expected=timeout"
    exit 77
  }
  PGID=$(ps -o pgid= -p $$ | tr -d ' ')
  guardian_pgid=$(ps -o pgid= -p "$GUARDIAN_PID" | tr -d ' ')
  [[ -n "$PGID" && "$guardian_pgid" == "$PGID" ]] || {
    echo "guardian_identity=false pid=$GUARDIAN_PID guardian_pgid=$guardian_pgid runner_pgid=$PGID"
    exit 77
  }
  echo "guardian_identity=true guardian_pid=$GUARDIAN_PID comm=$guardian_comm pgid=$PGID"
}

process_group_members() {
  [[ -n "$PGID" && -n "$GUARDIAN_PID" ]] || return 0
  python3 - "$PGID" "$$" "$GUARDIAN_PID" <<'PY'
import glob
import os
import sys
wanted = int(sys.argv[1])
script = int(sys.argv[2])
guardian = int(sys.argv[3])
me = os.getpid()
scanner_parent = os.getppid()
for path in glob.glob('/proc/[0-9]*/stat'):
    try:
        text = open(path).read()
        close = text.rfind(')')
        pid = int(text[:text.find(' ')])
        fields = text[close + 2:].split()
        if int(fields[2]) == wanted and pid not in (script, guardian, me, scanner_parent):
            print(f"pid={pid} ppid={fields[1]} pgid={fields[2]} state={fields[0]}")
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, IndexError):
        pass
PY
}

runtime_idle() {
  local found
  found=$(workers)
  [[ -z "$found" ]] || {
    echo "worker_clearance=false"
    printf '%s\n' "$found"
    return 1
  }
  sudo -n test -f "$QUARANTINE" || {
    echo "runtime_quarantine=missing_or_wrong_type"
    return 1
  }
  if ! sudo -n test ! -s "$QUARANTINE"; then
    echo "runtime_quarantine=nonempty"
    sudo -n cat "$QUARANTINE"
    return 1
  fi
  echo "worker_clearance=true runtime_quarantine=empty"
}

finish() {
  local rc=$?
  trap - EXIT INT TERM
  set +e
  local group_members worker_members lock_state quarantine_state group_scan worker_scan finished_epoch deadline_state
  if group_members=$(process_group_members); then
    group_scan=pass
  else
    group_scan=failed
  fi
  if worker_members=$(workers); then
    worker_scan=pass
  else
    worker_scan=failed
  fi
  if sudo -n test -f "$QUARANTINE" && sudo -n test ! -s "$QUARANTINE"; then
    quarantine_state=empty
  elif sudo -n test -e "$QUARANTINE"; then
    quarantine_state=unsafe
  else
    quarantine_state=missing_or_unreadable
  fi
  flock -u 9
  exec 9>&-
  exec 8>/tmp/m1-gpu.lock
  if flock -n 8; then
    lock_state=free
    flock -u 8
  else
    lock_state=busy
  fi
  exec 8>&-
  finished_epoch=$(date +%s)
  if (( finished_epoch <= SESSION_DEADLINE_EPOCH )); then
    deadline_state=within
  else
    deadline_state=missed
  fi
  if [[ "$group_scan" == pass && "$worker_scan" == pass && -z "$group_members" && -z "$worker_members" && "$lock_state" == free && "$quarantine_state" == empty && "$deadline_state" == within ]] && rm -f "$LEASE" && [[ ! -e "$LEASE" ]]; then
    echo "clearance=true outer_lock=free process_group_scan=pass process_group=empty worker_scan=pass workers=absent quarantine=empty lease=removed finished_epoch=$finished_epoch session_deadline_epoch=$SESSION_DEADLINE_EPOCH cleanup_deadline=$deadline_state exit_rc=$rc"
  else
    {
      printf 'quarantine=true exit_rc=%s observed=%s outer_lock=%s runtime_quarantine=%s cleanup_deadline=%s finished_epoch=%s session_deadline_epoch=%s\n' "$rc" "$(date -Is)" "$lock_state" "$quarantine_state" "$deadline_state" "$finished_epoch" "$SESSION_DEADLINE_EPOCH"
      printf 'group_scan=%s group_members=%s\nworker_scan=%s worker_members=%s\n' "$group_scan" "$group_members" "$worker_scan" "$worker_members"
    } >>"$LEASE"
    echo "clearance=false outer_lock=$lock_state process_group_scan=$group_scan process_group=${group_members:-empty} worker_scan=$worker_scan workers=${worker_members:-absent} quarantine=$quarantine_state finished_epoch=$finished_epoch session_deadline_epoch=$SESSION_DEADLINE_EPOCH cleanup_deadline=$deadline_state exit_rc=$rc"
    (( rc != 0 )) || rc=98
  fi
  exit "$rc"
}

validate_runtime_state() {
  sudo -n python3 - "$RENDER_GID" <<'PY'
import os
import stat
import sys
render_gid = int(sys.argv[1])
expected = (
    ("/run/lock/mlx-omarchy-ane", stat.S_ISDIR, 0o750, False),
    ("/run/lock/mlx-omarchy-ane/device.lock", stat.S_ISREG, 0o660, True),
    ("/run/lock/mlx-omarchy-ane/quarantine", stat.S_ISREG, 0o660, True),
)
for path, kind, mode, single_link in expected:
    value = os.stat(path, follow_symlinks=False)
    assert kind(value.st_mode), path
    assert value.st_uid == 0 and value.st_gid == render_gid, path
    assert stat.S_IMODE(value.st_mode) == mode, path
    assert not single_link or value.st_nlink == 1, path
    print(f"ownership_state path={path} inode={value.st_ino} uid={value.st_uid} gid={value.st_gid} mode={mode:o} links={value.st_nlink}")
PY
}

capture_guardian
test "$(hostname)" = jwm1-linux
test "$(uname -m)" = aarch64
test -c /dev/accel/accel0
for command in sudo c++ git sha256sum python3 timeout flock ldd getent find wc grep stat cat cut dirname pgrep ps rm cmake install readelf; do
  command -v "$command" >/dev/null
done
test -x "$VULKANINFO"
test "$(readlink -f "$VULKANINFO")" = /home/joshuawarren/.local/apple-hardware-sdk/usr/bin/vulkaninfo
sudo -n true
INSTALL_USER=$(id -un)
CALLER_UID=$(id -u)
test "$(sudo -n -u "$INSTALL_USER" -- id -u)" = "$CALLER_UID"
RENDER_GID=$(getent group render | cut -d: -f3)
test -n "$RENDER_GID"
FRESH_GROUPS=$(sudo -n -u "$INSTALL_USER" -- id -G)
case " $FRESH_GROUPS " in *" $RENDER_GID "*) ;; *) exit 97 ;; esac
validate_runtime_state
runtime_idle
LEASE_CONFLICT=$(find /tmp -maxdepth 1 -type f -name 'm1-gpu.lease.*' ! -name 'm1-gpu.lease.AneRuntimeBridge' -print -quit)
[[ -z "$LEASE_CONFLICT" ]] || {
  echo "lease_clearance=false conflict=$LEASE_CONFLICT"
  exit 75
}
test ! -e "$LEASE"
printf '%s  %s\n' "$SOURCE_BUNDLE_SHA" "$ROOT/source.bundle" | sha256sum --check --status
printf '%s  %s\n' "$LIBANE_BUNDLE_SHA" "$ROOT/libane.bundle" | sha256sum --check --status
printf '%s  %s\n' "$ADD_MANIFEST_SHA" "$ADD_BUNDLE/manifest.json" | sha256sum --check --status
printf '%s  %s\n' "$CHAIN_MANIFEST_SHA" "$CHAIN_BUNDLE/manifest.json" | sha256sum --check --status
test ! -e "$SOURCE"
test ! -e "$LIBANE"
test ! -e "$ROOT/wheel-work"
test ! -e "$ROOT/release"
test ! -e "$ROOT/install"
echo "preflight_identity host=$(hostname) machine=$(uname -m) user=$INSTALL_USER uid=$CALLER_UID groups=$FRESH_GROUPS render_gid=$RENDER_GID source=$SOURCE_COMMIT libane=$LIBANE_COMMIT source_bundle_sha256=$SOURCE_BUNDLE_SHA"
if (( PREFLIGHT_ONLY )); then
  echo "preflight=pass acquisition=not-attempted"
  exit 0
fi

NOW_EPOCH=$(date +%s)
(( NOW_EPOCH <= ACQUIRE_CUTOFF_EPOCH )) || {
  echo "acquire_cutoff_missed=true now=$NOW_EPOCH cutoff=$ACQUIRE_CUTOFF_EPOCH"
  exit 76
}
(( OUTER_DEADLINE_EPOCH - NOW_EPOCH >= 820 )) || {
  echo "outer_deadline_too_short=true remaining=$((OUTER_DEADLINE_EPOCH - NOW_EPOCH)) required=820 cleanup_reserve=$CLEANUP_RESERVE_SECONDS"
  exit 124
}
SESSION_DEADLINE_EPOCH=$((NOW_EPOCH + 900))
(( SESSION_DEADLINE_EPOCH > OUTER_DEADLINE_EPOCH )) && SESSION_DEADLINE_EPOCH=$OUTER_DEADLINE_EPOCH
DEADLINE_EPOCH=$((SESSION_DEADLINE_EPOCH - CLEANUP_RESERVE_SECONDS))
exec 9>/tmp/m1-gpu.lock
if ! flock -n 9; then
  echo "lock_acquired=false time=$(date -Is)"
  exit 75
fi
test "$(ps -o pgid= -p $$ | tr -d ' ')" = "$PGID"
test "$PPID" = "$GUARDIAN_PID"
test "$(cat "/proc/$GUARDIAN_PID/comm")" = timeout
printf 'agent=AneRuntimeBridge pid=%s pgid=%s guardian_pid=%s source=%s libane=%s started=%s work_deadline_epoch=%s outer_deadline_epoch=%s cleanup_reserve_seconds=%s scope=source-build-install-ane-runtime-acceptance\n' \
  "$$" "$PGID" "$GUARDIAN_PID" "$SOURCE_COMMIT" "$LIBANE_COMMIT" "$(date -Is)" "$DEADLINE_EPOCH" "$SESSION_DEADLINE_EPOCH" "$CLEANUP_RESERVE_SECONDS" >"$LEASE"
echo "lock_acquired=true time=$(date -Is) pid=$$ pgid=$PGID guardian_pid=$GUARDIAN_PID lease=$LEASE source=$SOURCE_COMMIT work_deadline_epoch=$DEADLINE_EPOCH outer_deadline_epoch=$SESSION_DEADLINE_EPOCH cleanup_reserve_seconds=$CLEANUP_RESERVE_SECONDS source_build=true install=true"
trap finish EXIT
trap 'exit 124' INT TERM
require_time acquired 760
runtime_idle

git clone --quiet "$ROOT/source.bundle" "$SOURCE"
git -C "$SOURCE" checkout --quiet --detach "$SOURCE_COMMIT"
git clone --quiet "$ROOT/libane.bundle" "$LIBANE"
git -C "$LIBANE" checkout --quiet --detach "$LIBANE_COMMIT"
test "$(git -C "$SOURCE" rev-parse HEAD)" = "$SOURCE_COMMIT"
test "$(git -C "$LIBANE" rev-parse HEAD)" = "$LIBANE_COMMIT"
git -C "$SOURCE" diff-index --quiet HEAD --
git -C "$LIBANE" diff-index --quiet HEAD --
echo "source_commit=$SOURCE_COMMIT libane_commit=$LIBANE_COMMIT host=$(hostname) kernel=$(uname -r) machine=$(uname -m)"

require_time wheel_build 700
run_long_with_window env \
  DEV_RELEASE=1 \
  MLX_OMARCHY_SOURCE_COMMIT="$SOURCE_COMMIT" \
  MLX_OMARCHY_ANE_SOURCE_DIR="$LIBANE" \
  MLX_OMARCHY_WORK_DIR="$ROOT/wheel-work" \
  CMAKE_BUILD_PARALLEL_LEVEL=1 \
  "$SOURCE/scripts/build-wheel.sh"
rm -f "$WHEEL_PATH_FILE" "$WHEEL_PATH_FILE.tmp"
python3 - "$SOURCE/dist" "$WHEEL_PATH_FILE" <<'PY'
import os
from pathlib import Path
import sys
root = Path(sys.argv[1]).resolve(strict=True)
output = Path(sys.argv[2])
wheels = list(root.glob('mlx_omarchy-*.whl'))
if len(wheels) != 1:
    raise SystemExit(f"expected exactly one wheel, found {wheels}")
temporary = output.with_suffix(output.suffix + '.tmp')
temporary.write_text(str(wheels[0].resolve(strict=True)) + '\n', encoding='utf-8')
os.replace(temporary, output)
PY
WHEEL=''
IFS= read -r WHEEL <"$WHEEL_PATH_FILE"
test -f "$WHEEL"
WHEEL_NAME=${WHEEL##*/}
WHEEL_SHA_EXPECTED=$(sha256sum "$WHEEL" | cut -d' ' -f1)
EXPECTED_VERSION=$(python3 - "$WHEEL" <<'PY'
import email
import sys
import zipfile
with zipfile.ZipFile(sys.argv[1]) as archive:
    names = [name for name in archive.namelist() if name.endswith('.dist-info/METADATA')]
    if len(names) != 1:
        raise SystemExit('wheel METADATA is not unique')
    version = email.message_from_bytes(archive.read(names[0]))['Version']
    if not version:
        raise SystemExit('wheel Version is missing')
    print(version)
PY
)
[[ "$EXPECTED_VERSION" == *"$SOURCE_COMMIT"* ]]
mkdir -p "$ROOT/release" "$ROOT/home"
install -m 0644 "$WHEEL" "$ROOT/release/$WHEEL_NAME"
(cd "$ROOT/release" && sha256sum "$WHEEL_NAME" >SHA256SUMS)
WHEEL="$ROOT/release/$WHEEL_NAME"
printf '%s  %s\n' "$WHEEL_SHA_EXPECTED" "$WHEEL" | sha256sum --check --status
echo "wheel=$WHEEL_NAME version=$EXPECTED_VERSION wheel_sha256=$WHEEL_SHA_EXPECTED"

require_time installer 220
run_long_with_window env \
  HOME="$ROOT/home" \
  MLX_OMARCHY_HOME="$ROOT/install" \
  MLX_OMARCHY_RELEASE_BASE="file://$ROOT/release" \
  MLX_OMARCHY_VERSION=v0.4.2 \
  "$SOURCE/install.sh" --ane
VENV="$ROOT/install/venv"
test -x "$VENV/bin/python"
FRESH_GROUPS=$(sudo -n -u "$INSTALL_USER" -- id -G)
case " $FRESH_GROUPS " in *" $RENDER_GID "*) ;; *) exit 97 ;; esac
validate_runtime_state
runtime_idle

rm -f "$IDENTITY_FILE" "$IDENTITY_FILE.tmp"
run_as_user env HOME="$ROOT/home" "$VENV/bin/python" -I - \
  "$WHEEL" "$VENV" "$IDENTITY_FILE" "$EXPECTED_VERSION" "$SOURCE_COMMIT" <<'PY'
import base64
import csv
import hashlib
import hmac
import importlib.metadata
import os
from pathlib import Path
import stat
import sys
import zipfile

wheel = Path(sys.argv[1]).resolve(strict=True)
expected_venv = Path(sys.argv[2]).resolve(strict=True)
output = Path(sys.argv[3])
expected_version = sys.argv[4]
source_commit = sys.argv[5]

import mlx.core

def mapped_files():
    paths = set()
    with open('/proc/self/maps', encoding='utf-8') as maps:
        for line in maps:
            fields = line.rstrip().split(maxsplit=5)
            if len(fields) != 6 or not fields[5].startswith('/'):
                continue
            if fields[5].endswith(' (deleted)'):
                raise SystemExit(f"mapped file was deleted: {fields[5]}")
            paths.add(Path(fields[5]).resolve(strict=True))
    return paths

def checked_regular(path, executable=False):
    value = path.lstat()
    if not stat.S_ISREG(value.st_mode) or path.is_symlink():
        raise SystemExit(f"identity path is not a regular non-symlink: {path}")
    if executable and not os.access(path, os.X_OK):
        raise SystemExit(f"worker is not executable: {path}")

def digest(path):
    return hashlib.sha256(path.read_bytes()).digest()

version = importlib.metadata.version('mlx-omarchy')
if version != expected_version or source_commit not in version:
    raise SystemExit(f"installed version mismatch: {version}")
distribution = importlib.metadata.distribution('mlx-omarchy')
dist_root = Path(distribution.locate_file('')).resolve(strict=True)
if not dist_root.is_relative_to(expected_venv):
    raise SystemExit(f"distribution escaped dedicated venv: {dist_root}")
core_candidate = Path(mlx.core.__spec__.origin)
lib_candidate = dist_root / 'mlx/lib/libmlx.so'
worker_candidate = dist_root / 'mlx/bin/mlx-omarchy-ane-worker'
for path in (core_candidate, lib_candidate):
    checked_regular(path)
checked_regular(worker_candidate, executable=True)
core = core_candidate.resolve(strict=True)
installed_lib = lib_candidate.resolve(strict=True)
worker = worker_candidate.resolve(strict=True)

mapped = mapped_files()
if core not in mapped:
    raise SystemExit(f"imported mlx.core is not mapped: {core}")
mapped_libs = sorted(path for path in mapped if path.name == 'libmlx.so')
if mapped_libs != [installed_lib]:
    raise SystemExit(f"expected only installed libmlx.so to be mapped, found {mapped_libs}")

with zipfile.ZipFile(wheel) as archive:
    names = archive.namelist()
    record_names = [name for name in names if name.endswith('.dist-info/RECORD')]
    metadata_names = [name for name in names if name.endswith('.dist-info/METADATA')]
    if len(record_names) != 1 or len(metadata_names) != 1:
        raise SystemExit('wheel metadata members are not unique')
    metadata = archive.read(metadata_names[0]).decode('utf-8')
    if f'Version: {version}\n' not in metadata.replace('\r\n', '\n'):
        raise SystemExit('wheel and installed versions differ')
    rows = list(csv.reader(archive.read(record_names[0]).decode('utf-8').splitlines()))
    records = {row[0]: row[1] for row in rows}
    for path in (core, installed_lib, worker):
        relative = path.relative_to(dist_root).as_posix()
        encoded = records.get(relative, '')
        if not encoded.startswith('sha256='):
            raise SystemExit(f"missing sha256 RECORD identity for {relative}")
        expected = base64.urlsafe_b64decode(encoded.removeprefix('sha256=') + '==')
        archived = hashlib.sha256(archive.read(relative)).digest()
        installed = digest(path)
        if not hmac.compare_digest(expected, archived):
            raise SystemExit(f"wheel RECORD does not match archived member: {relative}")
        if not hmac.compare_digest(expected, installed):
            raise SystemExit(f"installed file does not match wheel RECORD: {relative}")

values = (
    version,
    str(dist_root),
    str(core),
    digest(core).hex(),
    str(installed_lib),
    digest(installed_lib).hex(),
    str(worker),
    digest(worker).hex(),
    digest(wheel).hex(),
)
if any('\t' in value or '\n' in value or not value for value in values):
    raise SystemExit('identity fields are not shell-safe')
temporary = output.with_suffix(output.suffix + '.tmp')
temporary.write_text('\t'.join(values) + '\n', encoding='utf-8')
os.replace(temporary, output)
PY

test -s "$IDENTITY_FILE"
test "$(wc -l <"$IDENTITY_FILE")" -eq 1
MLX_VERSION=''
DIST_ROOT=''
CORE_FILE=''
CORE_SHA=''
INSTALLED_LIB=''
INSTALLED_LIB_SHA=''
INSTALLED_WORKER=''
INSTALLED_WORKER_SHA=''
WHEEL_SHA=''
IFS=$'\t' read -r MLX_VERSION DIST_ROOT CORE_FILE CORE_SHA INSTALLED_LIB INSTALLED_LIB_SHA INSTALLED_WORKER INSTALLED_WORKER_SHA WHEEL_SHA <"$IDENTITY_FILE"
for value in "$MLX_VERSION" "$DIST_ROOT" "$CORE_FILE" "$CORE_SHA" "$INSTALLED_LIB" "$INSTALLED_LIB_SHA" "$INSTALLED_WORKER" "$INSTALLED_WORKER_SHA" "$WHEEL_SHA"; do
  test -n "$value"
done
test "$MLX_VERSION" = "$EXPECTED_VERSION"
test "$WHEEL_SHA" = "$WHEEL_SHA_EXPECTED"
echo "wheel_record_identity=pass installed_version=$MLX_VERSION dist_root=$DIST_ROOT mapped_core=$CORE_FILE mapped_core_sha256=$CORE_SHA mapped_libmlx=$INSTALLED_LIB mapped_libmlx_sha256=$INSTALLED_LIB_SHA installed_worker=$INSTALLED_WORKER installed_worker_sha256=$INSTALLED_WORKER_SHA wheel_sha256=$WHEEL_SHA"
stat -c 'installed_core_identity inode=%i mode=%a uid=%u gid=%g bytes=%s' "$CORE_FILE"
stat -c 'installed_libmlx_identity inode=%i mode=%a uid=%u gid=%g bytes=%s' "$INSTALLED_LIB"
stat -c 'installed_worker_identity inode=%i mode=%a uid=%u gid=%g bytes=%s' "$INSTALLED_WORKER"
readelf -d "$INSTALLED_WORKER" | grep -F 'Library runpath: [$ORIGIN/../lib]'

QUARANTINE_INODE=$(sudo -n stat -c '%i' "$QUARANTINE")
test -n "$QUARANTINE_INODE"
runtime_idle

require_time smoke_harness_compile 300
INSTALLED_SMOKE="$ROOT/installed-ane-smoke"
LIBDIR=$(dirname "$INSTALLED_LIB")
run_with_window c++ -std=c++20 -O2 \
  -I"$ROOT/wheel-work/mlx" \
  "$SOURCE/overlay/tests/omarchy/ane/runtime_hardware_smoke.cpp" \
  "$INSTALLED_LIB" -Wl,-rpath,"$LIBDIR" -ldl -pthread -o "$INSTALLED_SMOKE"
LD_RESOLUTION=$(ldd "$INSTALLED_SMOKE")
printf '%s\n' "$LD_RESOLUTION"
[[ "$LD_RESOLUTION" == *"$INSTALLED_LIB"* ]]

audit_ld_debug() {
  local prefix=$1
  python3 - "$prefix" "$INSTALLED_LIB" <<'PY'
import glob
import sys
prefix, installed = sys.argv[1:]
files = glob.glob(prefix + '.*')
if not files:
    raise SystemExit(f"no LD_DEBUG files for {prefix}")
matched = []
for path in files:
    with open(path, errors='replace') as source:
        if installed in source.read():
            matched.append(path)
if not matched:
    raise SystemExit(f"installed libmlx was absent from LD_DEBUG files: {files}")
print(f"ld_debug_identity=pass installed_libmlx={installed} files={','.join(sorted(matched))}")
PY
}

run_smoke() {
  local label=$1 bundle=$2 operation=$3 released=$4
  local diagnostic output_file ld_prefix
  require_time "$label" 55
  diagnostic="$ROOT/${label}.diagnostic"
  output_file="$ROOT/${label}.output"
  ld_prefix="$ROOT/${label}.lddebug"
  rm -f "$diagnostic" "$output_file" "$ld_prefix".*
  if ! run_as_user env -u LD_LIBRARY_PATH \
      EXPECTED_UID="$CALLER_UID" \
      EXPECTED_RENDER_GID="$RENDER_GID" \
      bash -c '
        test "$(id -u)" = "$EXPECTED_UID"
        case " $(id -G) " in *" $EXPECTED_RENDER_GID "*) ;; *) exit 97 ;; esac
        echo "smoke_effective_identity uid=$(id -u) groups=$(id -G) render_gid=$EXPECTED_RENDER_GID"
        ldout=$1
        shift
        export LD_DEBUG=libs,files LD_DEBUG_OUTPUT="$ldout"
        exec "$@"
      ' bash "$ld_prefix" \
      "$INSTALLED_SMOKE" "$bundle" "$operation" 10000 "$diagnostic" >"$output_file" 2>&1; then
    cat "$output_file"
    if [[ -f "$diagnostic" ]]; then
      echo diagnostic_begin
      cat "$diagnostic"
      echo diagnostic_end
    fi
    return 1
  fi
  cat "$output_file"
  grep -F 'runtime_identity ' "$output_file"
  grep -F "worker_executable \"$INSTALLED_WORKER\"" "$output_file" >/dev/null || \
    grep -F "worker_executable $INSTALLED_WORKER" "$output_file" >/dev/null
  grep -F 'iteration 0 exact_fp16=PASS bytes=128' "$output_file"
  grep -F 'iteration 1 exact_fp16=PASS bytes=128' "$output_file"
  grep -F "released_programs=$released process_released=true" "$output_file"
  test -f "$diagnostic"
  grep -q '^runtime_load_graph_hash=[0-9a-f]\{64\}$' "$diagnostic"
  if grep -q -E 'reason=|submitted=|recovery=' "$diagnostic"; then
    echo "unexpected_failure_diagnostic=$diagnostic"
    return 1
  fi
  audit_ld_debug "$ld_prefix"
  runtime_idle
  echo "smoke=$label status=pass actual_loaded_libmlx=$INSTALLED_LIB actual_worker=$INSTALLED_WORKER exact_fp16_iterations=2 released_programs=$released process_released=true"
}

run_smoke add "$ADD_BUNDLE" add 1
run_smoke add-mul "$CHAIN_BUNDLE" add-mul 2
test "$(sudo -n stat -c '%i' "$QUARANTINE")" = "$QUARANTINE_INODE"
sudo -n test ! -s "$QUARANTINE"
run_as_user flock -n "$DEVICE_LOCK" -c true
require_time gpu_postcheck 20
run_as_user "$VENV/bin/python" - <<'PY'
import mlx.core as mx
mx.set_default_device(mx.gpu)
if mx.default_device() != mx.gpu:
    raise SystemExit(f"gpu_default=false actual={mx.default_device()}")
a = mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float32)
b = mx.array([[5.0, 6.0], [7.0, 8.0]], dtype=mx.float32)
out = mx.matmul(a, b) + mx.array(1.0, dtype=mx.float32)
mx.eval(out)
mx.synchronize()
actual = out.tolist()
expected = [[20.0, 23.0], [44.0, 51.0]]
if actual != expected:
    raise SystemExit(f"numerical_match=false actual={actual!r} expected={expected!r}")
info = mx.device_info()
if info.get("device_name") != "Apple M1 (G13G B1)" or info.get("architecture") != "honeykrisp":
    raise SystemExit(f"gpu_identity=false info={info!r}")
print(f"gpu_smoke=pass default_device={mx.default_device()} numerical_match=true result={actual!r} device_info={info!r}")
PY
run_as_user "$VULKANINFO" --summary
echo "vulkan_device_reopen=true path=$VULKANINFO"
runtime_idle
test -z "$(process_group_members)"
echo "installed_ane_runtime_acceptance=pass source=$SOURCE_COMMIT libane=$LIBANE_COMMIT version=$MLX_VERSION wheel_sha256=$WHEEL_SHA loaded_core=$CORE_FILE loaded_core_sha256=$CORE_SHA loaded_libmlx=$INSTALLED_LIB loaded_libmlx_sha256=$INSTALLED_LIB_SHA worker=$INSTALLED_WORKER worker_sha256=$INSTALLED_WORKER_SHA smokes=2 executions=4 gpu_postcheck=true vulkan_reopen=true quarantine_inode_preserved=$QUARANTINE_INODE source_build=true install=true"
