#!/usr/bin/env bash
set -euo pipefail

SOURCE_COMMIT=57cc36a2e8ece78dd979b5344752d04c76b4d49d
LIBANE_COMMIT=f261a6cb537aca62f267ad3d01beda0d6877544c
EXPECTED_VERSION=0.32.2.dev202609130952+57cc36a2e8ece78dd979b5344752d04c76b4d49d
EXPECTED_WHEEL_SHA=69e381c7ea91ca228a0d4ffa23e9409b3cd1b9026ed102179d05ee1043ab42d3
EXPECTED_CORE_SHA=4ad850da16c300ea9f60aa7247f034f358aa16d7ba2fd146979a01c867e3cfa5
EXPECTED_LIB_SHA=402d4da86cd09482344c9cf624eb085323a73d6a38790f60b9e96a5210a0a3f0
EXPECTED_WORKER_SHA=52fa2959b0d2cfb7e7a0af877deaac605b7141b033879da1f8f5b908575e2198
SOURCE_BUNDLE_SHA=aebef2154ed0b0c543d796c4d68f52ef3670b2826e17879237dc762f589bf7af
ADD_MANIFEST_SHA=f974e82eca49726a9eafb58f95a6578a458d423bab12efcdc3c3b780f1bbf680
CHAIN_MANIFEST_SHA=065f2acd0a0668c0911fcc728bc8bfef56812e0e64c0064de6287ac29dc98619
FIXTURE_TAR_SHA=194ee8551491b8c00b57ebc8c3ca16212d69c0588542c55c4c653bf1c0fb145f
VULKAN_MEL_SHA=4f61cd5cd1ebabb2a96d3270e347d3a6607be9b9c186f56663c5f9d14ba095c0
COMPARATOR_SHA=210b38995c1491890fd71a7ae6c5c79b79c6d8d583289e3b3831a86b7066b7f8
CONSTANTS_SHA=caad0ac74d60a4d7378326d5c6d2e0f0fa7d1fd807157f9d203e7348874d258f
LOCK_SHA=63415836246175d34748c1c44f2c8d00c79affc5be0f4d7da058eb7d5328f8d1
ROOT=/var/tmp/AneRuntimeBridge-57cc36a2
SOURCE="$ROOT/source"
VENV="$ROOT/install/venv"
IDENTITY_FILE="$ROOT/installed-identity.tsv"
WHEEL_PATH_FILE="$ROOT/wheel-path.txt"
ADD_BUNDLE="$ROOT/add"
CHAIN_BUNDLE="$ROOT/add-mul"
FIXTURE_TAR="$ROOT/parakeet-mel-fixtures-18.tar"
FIXTURE_ROOT="$ROOT/parakeet-mel-fixtures-18"
CAPTURE="$FIXTURE_ROOT/20260912T154759Z-librispeech/ane"
STAGES="$FIXTURE_ROOT/mel-stage-probes/stage-capture"
REPORT="$ROOT/installed-57cc-parakeet-mel-18.json"
LEASE=/tmp/m1-gpu.lease.AneRuntimeBridge
QUARANTINE=/run/lock/mlx-omarchy-ane/quarantine
DEVICE_LOCK=/run/lock/mlx-omarchy-ane/device.lock
VULKANINFO=/home/joshuawarren/.local/bin/vulkaninfo
CLEANUP_RESERVE_SECONDS=60
: "${ACQUIRE_CUTOFF_EPOCH:?ACQUIRE_CUTOFF_EPOCH is required}"
: "${OUTER_DEADLINE_EPOCH:?OUTER_DEADLINE_EPOCH is required}"
SESSION_DEADLINE_EPOCH=0
DEADLINE_EPOCH=0
PGID=""
GUARDIAN_PID=""
LEASE_OWNED=0

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
  if [[ "$group_scan" == pass && "$worker_scan" == pass && -z "$group_members" && -z "$worker_members" && "$lock_state" == free && "$quarantine_state" == empty && "$deadline_state" == within ]] && (( LEASE_OWNED )) && [[ -e "$LEASE" ]] && rm -f "$LEASE" && [[ ! -e "$LEASE" ]]; then
    echo "clearance=true outer_lock=free process_group_scan=pass process_group=empty worker_scan=pass workers=absent quarantine=empty lease=removed finished_epoch=$finished_epoch session_deadline_epoch=$SESSION_DEADLINE_EPOCH cleanup_deadline=$deadline_state exit_rc=$rc"
  else
    if (( LEASE_OWNED )); then
      {
        printf 'quarantine=true exit_rc=%s observed=%s outer_lock=%s runtime_quarantine=%s cleanup_deadline=%s finished_epoch=%s session_deadline_epoch=%s\n' "$rc" "$(date -Is)" "$lock_state" "$quarantine_state" "$deadline_state" "$finished_epoch" "$SESSION_DEADLINE_EPOCH"
        printf 'group_scan=%s group_members=%s\nworker_scan=%s worker_members=%s\n' "$group_scan" "$group_members" "$worker_scan" "$worker_members"
      } >>"$LEASE"
    fi
    echo "clearance=false outer_lock=$lock_state process_group_scan=$group_scan process_group=${group_members:-empty} worker_scan=$worker_scan workers=${worker_members:-absent} quarantine=$quarantine_state lease=retained finished_epoch=$finished_epoch session_deadline_epoch=$SESSION_DEADLINE_EPOCH cleanup_deadline=$deadline_state exit_rc=$rc"
    (( rc != 0 )) || rc=98
  fi
  exit "$rc"
}

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
  (( remaining > 0 )) || { echo "absolute_work_deadline_expired=true"; exit 124; }
  (( remaining <= 90 )) || remaining=90
  timeout --signal=TERM --kill-after=10s "${remaining}s" "$@"
}

run_as_user() {
  local remaining
  remaining=$(remaining_seconds)
  (( remaining > 0 )) || { echo "absolute_work_deadline_expired=true"; exit 124; }
  (( remaining <= 90 )) || remaining=90
  timeout --signal=TERM --kill-after=10s "${remaining}s" sudo -n -u "$INSTALL_USER" -- "$@"
}

runtime_idle() {
  local found
  found=$(workers)
  [[ -z "$found" ]] || { echo "worker_clearance=false"; printf '%s\n' "$found"; return 1; }
  sudo -n test -f "$QUARANTINE"
  sudo -n test ! -s "$QUARANTINE"
  echo "worker_clearance=true runtime_quarantine=empty"
}

capture_guardian
now=$(date +%s)
[[ "$ACQUIRE_CUTOFF_EPOCH" =~ ^[0-9]+$ && "$OUTER_DEADLINE_EPOCH" =~ ^[0-9]+$ ]]
(( now <= ACQUIRE_CUTOFF_EPOCH ))
SESSION_DEADLINE_EPOCH=$((now + 660))
(( SESSION_DEADLINE_EPOCH > OUTER_DEADLINE_EPOCH )) && SESSION_DEADLINE_EPOCH=$OUTER_DEADLINE_EPOCH
DEADLINE_EPOCH=$((SESSION_DEADLINE_EPOCH - CLEANUP_RESERVE_SECONDS))
(( DEADLINE_EPOCH - now >= 600 ))
exec 9>/tmp/m1-gpu.lock
flock -n 9 || { echo "lock_acquired=false"; exit 75; }
now=$(date +%s)
(( now <= ACQUIRE_CUTOFF_EPOCH )) || { echo "acquisition_cutoff_missed=true now=$now cutoff=$ACQUIRE_CUTOFF_EPOCH"; exit 124; }
test ! -e "$LEASE"
printf 'owner=AneRuntimeBridge\nsource=%s\npid=%s\npgid=%s\ndeadline=%s\n' "$SOURCE_COMMIT" "$$" "$PGID" "$SESSION_DEADLINE_EPOCH" >"$LEASE"
LEASE_OWNED=1
trap finish EXIT
trap 'exit 124' INT TERM
printf 'lock_acquired=true pid=%s pgid=%s guardian_pid=%s work_deadline_epoch=%s session_deadline_epoch=%s cleanup_reserve_seconds=%s source_build=false install=false\n' "$$" "$PGID" "$GUARDIAN_PID" "$DEADLINE_EPOCH" "$SESSION_DEADLINE_EPOCH" "$CLEANUP_RESERVE_SECONDS"

INSTALL_USER=$(id -un)
CALLER_UID=$(id -u)
RENDER_GID=$(stat -c '%g' /dev/dri/renderD128)
test "$(git -C "$SOURCE" rev-parse HEAD)" = "$SOURCE_COMMIT"
printf '%s  %s\n' "$SOURCE_BUNDLE_SHA" "$ROOT/source.bundle" | sha256sum --check --status
printf '%s  %s\n' "$ADD_MANIFEST_SHA" "$ADD_BUNDLE/manifest.json" | sha256sum --check --status
printf '%s  %s\n' "$CHAIN_MANIFEST_SHA" "$CHAIN_BUNDLE/manifest.json" | sha256sum --check --status
printf '%s  %s\n' "$FIXTURE_TAR_SHA" "$FIXTURE_TAR" | sha256sum --check --status
printf '%s  %s\n' "$VULKAN_MEL_SHA" "$SOURCE/overlay/tools/coreml/vulkan_mel.py" | sha256sum --check --status
printf '%s  %s\n' "$COMPARATOR_SHA" "$SOURCE/overlay/tools/coreml/mel_stage_compare.py" | sha256sum --check --status
printf '%s  %s\n' "$CONSTANTS_SHA" "$SOURCE/overlay/tools/coreml/vulkan_mel_constants.py" | sha256sum --check --status
printf '%s  %s\n' "$LOCK_SHA" "$SOURCE/overlay/tools/coreml/parakeet-reference.lock" | sha256sum --check --status
test -x "$VENV/bin/python"
test -s "$IDENTITY_FILE"
test -s "$WHEEL_PATH_FILE"
sudo -n python3 - "$RENDER_GID" <<'PY'
import os
import stat
import sys
render_gid = int(sys.argv[1])
device = os.stat('/dev/accel/accel0', follow_symlinks=False)
if not stat.S_ISCHR(device.st_mode) or device.st_gid != render_gid:
    raise SystemExit('ANE device identity or render group changed')
expected = (
    ('/run/lock/mlx-omarchy-ane', stat.S_ISDIR, 0o750, False),
    ('/run/lock/mlx-omarchy-ane/device.lock', stat.S_ISREG, 0o660, True),
    ('/run/lock/mlx-omarchy-ane/quarantine', stat.S_ISREG, 0o660, True),
)
for path, kind, mode, single_link in expected:
    value = os.stat(path, follow_symlinks=False)
    valid = kind(value.st_mode) and value.st_uid == 0 and value.st_gid == render_gid and stat.S_IMODE(value.st_mode) == mode and (not single_link or value.st_nlink == 1)
    if not valid:
        raise SystemExit(f'invalid ANE ownership path: {path}')
    print(f'ownership_state path={path} inode={value.st_ino} uid={value.st_uid} gid={value.st_gid} mode={stat.S_IMODE(value.st_mode):o} links={value.st_nlink}')
PY
runtime_idle

run_as_user env HOME="$ROOT/home" "$VENV/bin/python" -I - \
  "$IDENTITY_FILE" "$WHEEL_PATH_FILE" "$EXPECTED_VERSION" "$EXPECTED_WHEEL_SHA" \
  "$EXPECTED_CORE_SHA" "$EXPECTED_LIB_SHA" "$EXPECTED_WORKER_SHA" <<'PY'
import base64
import csv
import hashlib
import hmac
import importlib.metadata
import os
from pathlib import Path
import sys
import zipfile
identity_path, wheel_path_file, expected_version, expected_wheel, expected_core, expected_lib, expected_worker = sys.argv[1:]
fields = Path(identity_path).read_text(encoding='utf-8').rstrip('\n').split('\t')
if len(fields) != 9:
    raise SystemExit(f'installed identity field count is {len(fields)}, expected 9')
version, dist_root_text, core_text, core_sha, lib_text, lib_sha, worker_text, worker_sha, wheel_sha = fields
expected = (expected_version, expected_core, expected_lib, expected_worker, expected_wheel)
actual = (version, core_sha, lib_sha, worker_sha, wheel_sha)
if actual != expected:
    raise SystemExit(f'persisted identity differs: {actual!r} != {expected!r}')
dist_root = Path(dist_root_text).resolve(strict=True)
core = Path(core_text).resolve(strict=True)
lib = Path(lib_text).resolve(strict=True)
worker = Path(worker_text).resolve(strict=True)
wheel = Path(Path(wheel_path_file).read_text(encoding='utf-8').strip()).resolve(strict=True)
def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).digest()
if digest(wheel).hex() != expected_wheel:
    raise SystemExit('wheel bytes changed')
with zipfile.ZipFile(wheel) as archive:
    records = [name for name in archive.namelist() if name.endswith('.dist-info/RECORD')]
    metadata = [name for name in archive.namelist() if name.endswith('.dist-info/METADATA')]
    if len(records) != 1 or len(metadata) != 1:
        raise SystemExit('wheel metadata members are not unique')
    if f'Version: {version}\n' not in archive.read(metadata[0]).decode().replace('\r\n', '\n'):
        raise SystemExit('wheel metadata version differs')
    rows = csv.reader(archive.read(records[0]).decode().splitlines())
    record = {row[0]: row[1] for row in rows}
    for path in (core, lib, worker):
        relative = path.relative_to(dist_root).as_posix()
        encoded = record.get(relative, '')
        if not encoded.startswith('sha256='):
            raise SystemExit(f'missing RECORD hash for {relative}')
        wanted = base64.urlsafe_b64decode(encoded.removeprefix('sha256=') + '==')
        if not hmac.compare_digest(wanted, hashlib.sha256(archive.read(relative)).digest()):
            raise SystemExit(f'archived member differs from RECORD: {relative}')
        if not hmac.compare_digest(wanted, digest(path)):
            raise SystemExit(f'installed member differs from RECORD: {relative}')
import mlx.core as mx
if Path(mx.__file__).resolve() != core:
    raise SystemExit(f'imported core differs: {mx.__file__} != {core}')
if importlib.metadata.version('mlx-omarchy') != version:
    raise SystemExit('installed distribution version differs')
mapped = set()
with open('/proc/self/maps', encoding='utf-8') as maps:
    for line in maps:
        path = line.rstrip().split(maxsplit=5)
        if len(path) == 6 and path[5].startswith('/'):
            mapped.add(Path(os.path.realpath(path[5])))
if core not in mapped:
    raise SystemExit(f'imported core is not mapped: {core}')
mapped_libs = sorted(path for path in mapped if path.name == 'libmlx.so')
if mapped_libs != [lib]:
    raise SystemExit(f'expected only installed libmlx.so, found {mapped_libs}')
print(f'wheel_record_identity=pass installed_version={version} mapped_core={core} mapped_core_sha256={core_sha} mapped_libmlx={lib} mapped_libmlx_sha256={lib_sha} installed_worker={worker} installed_worker_sha256={worker_sha} wheel_sha256={wheel_sha}')
PY

IFS=$'\t' read -r MLX_VERSION DIST_ROOT CORE_FILE CORE_SHA INSTALLED_LIB INSTALLED_LIB_SHA INSTALLED_WORKER INSTALLED_WORKER_SHA WHEEL_SHA <"$IDENTITY_FILE"
for value in "$MLX_VERSION" "$DIST_ROOT" "$CORE_FILE" "$CORE_SHA" "$INSTALLED_LIB" "$INSTALLED_LIB_SHA" "$INSTALLED_WORKER" "$INSTALLED_WORKER_SHA" "$WHEEL_SHA"; do test -n "$value"; done
readelf -d "$INSTALLED_WORKER" | grep -F 'Library runpath: [$ORIGIN/../lib]'
QUARANTINE_INODE=$(sudo -n stat -c '%i' "$QUARANTINE")
require_time smoke_harness_compile 300
INSTALLED_SMOKE="$ROOT/installed-ane-smoke"
LIBDIR=$(dirname "$INSTALLED_LIB")
run_with_window c++ -std=c++20 -O2 -I"$ROOT/wheel-work/mlx" \
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
    raise SystemExit(f'no LD_DEBUG files for {prefix}')
matched = []
for path in files:
    with open(path, errors='replace') as source:
        if installed in source.read():
            matched.append(path)
if not matched:
    raise SystemExit(f'installed libmlx absent from LD_DEBUG files: {files}')
print(f"ld_debug_identity=pass installed_libmlx={installed} files={','.join(sorted(matched))}")
PY
}

run_smoke() {
  local label=$1 bundle=$2 operation=$3 released=$4
  local diagnostic output_file ld_prefix
  require_time "$label" 55
  diagnostic="$ROOT/combined-${label}.diagnostic"
  output_file="$ROOT/combined-${label}.output"
  ld_prefix="$ROOT/combined-${label}.lddebug"
  rm -f "$diagnostic" "$output_file" "$ld_prefix".*
  if ! run_as_user env -u LD_LIBRARY_PATH EXPECTED_UID="$CALLER_UID" EXPECTED_RENDER_GID="$RENDER_GID" bash -c '
    test "$(id -u)" = "$EXPECTED_UID"
    case " $(id -G) " in *" $EXPECTED_RENDER_GID "*) ;; *) exit 97 ;; esac
    echo "smoke_effective_identity uid=$(id -u) groups=$(id -G) render_gid=$EXPECTED_RENDER_GID"
    ldout=$1
    shift
    export LD_DEBUG=libs,files LD_DEBUG_OUTPUT="$ldout"
    exec "$@"
  ' bash "$ld_prefix" "$INSTALLED_SMOKE" "$bundle" "$operation" 10000 "$diagnostic" >"$output_file" 2>&1; then
    cat "$output_file"
    if [[ -f "$diagnostic" ]]; then echo diagnostic_begin; cat "$diagnostic"; echo diagnostic_end; fi
    return 1
  fi
  cat "$output_file"
  grep -F 'runtime_identity ' "$output_file"
  grep -F "worker_executable \"$INSTALLED_WORKER\"" "$output_file" >/dev/null || grep -F "worker_executable $INSTALLED_WORKER" "$output_file" >/dev/null
  grep -F 'iteration 0 exact_fp16=PASS bytes=128' "$output_file"
  grep -F 'iteration 1 exact_fp16=PASS bytes=128' "$output_file"
  grep -F "released_programs=$released process_released=true" "$output_file"
  test -f "$diagnostic"
  grep -q '^runtime_load_graph_hash=[0-9a-f]\{64\}$' "$diagnostic"
  if grep -q -E 'reason=|submitted=|recovery=' "$diagnostic"; then echo "unexpected_failure_diagnostic=$diagnostic"; return 1; fi
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
    raise SystemExit(f'gpu_default=false actual={mx.default_device()}')
a = mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float32)
b = mx.array([[5.0, 6.0], [7.0, 8.0]], dtype=mx.float32)
out = mx.matmul(a, b) + mx.array(1.0, dtype=mx.float32)
mx.eval(out)
mx.synchronize()
actual = out.tolist()
expected = [[20.0, 23.0], [44.0, 51.0]]
if actual != expected:
    raise SystemExit(f'numerical_match=false actual={actual!r} expected={expected!r}')
info = mx.device_info()
if info.get('device_name') != 'Apple M1 (G13G B1)' or info.get('architecture') != 'honeykrisp':
    raise SystemExit(f'gpu_identity=false info={info!r}')
print(f'gpu_smoke=pass default_device={mx.default_device()} numerical_match=true result={actual!r} device_info={info!r}')
PY

require_time mel_frontend 180
rm -f "$REPORT"
run_as_user env -u PYTHONPATH -u LD_LIBRARY_PATH HOME="$ROOT/home" MLX_OMARCHY_ALLOW_NON_APPLE=1 \
  "$VENV/bin/python" -I - "$CORE_FILE" "$CORE_SHA" "$WHEEL_SHA" \
  "$SOURCE/overlay/tools/coreml/vulkan_mel.py" "$CAPTURE" "$STAGES" "$REPORT" <<'PY'
from pathlib import Path
import runpy
import sys
core_file, core_sha, wheel_sha, script, capture, stages, report = sys.argv[1:]
import mlx.core as mx
if Path(mx.__file__).resolve() != Path(core_file).resolve():
    raise SystemExit(f'imported extension mismatch: {mx.__file__} != {core_file}')
mx.set_default_device(mx.gpu)
print(f'installed_extension_identity=pass core={core_file} core_sha256={core_sha} wheel_sha256={wheel_sha} device={mx.default_device()}')
sys.path.insert(0, str(Path(script).resolve().parent))
sys.argv = [script, capture, stages, '--json-out', report]
runpy.run_path(script, run_name='__main__')
PY

python3 - "$REPORT" <<'PY'
import json
import sys
report = json.load(open(sys.argv[1], encoding='utf-8'))
expected = {'preemph', 'hann', 'mel_fb', 'frames', 'dft_real', 'dft_imag', 'power', 'melproj', 'logmel', 'mean', 'std', 'mel_stepwise', 'mel_pinned', 'mel_mask', 'mel', 'mask', 'encoder_features', 'encoder_mask'}
comparisons = report.get('comparisons', {})
if set(comparisons) != expected or len(comparisons) != 18:
    raise SystemExit(f'comparison set mismatch: {sorted(comparisons)}')
if not report.get('qualified') or not report.get('all_bit_exact') or not report.get('stage_set_exact'):
    raise SystemExit('qualification booleans are not all true')
if any(not value.get('bit_exact') for value in comparisons.values()):
    raise SystemExit('at least one comparison is not bit-exact')
if report.get('device') != 'Device(gpu, 0)':
    raise SystemExit(f"unexpected device: {report.get('device')}")
trace = report.get('trace_delta', {})
for key in ('gpu_primitive_dispatches', 'vk_compute_dispatches', 'vk_submissions'):
    if not isinstance(trace.get(key), int) or trace[key] <= 0:
        raise SystemExit(f'{key} is not positive: {trace.get(key)!r}')
if trace.get('vk_compute_dispatches') != 8:
    raise SystemExit(f"vk_compute_dispatches={trace.get('vk_compute_dispatches')} expected=8")
print(f"installed_57cc_mel_acceptance=pass comparisons=18 device={report['device']} trace={json.dumps(trace, sort_keys=True, separators=(',', ':'))}")
PY

run_as_user "$VULKANINFO" --summary
echo "vulkan_device_reopen=true path=$VULKANINFO"
runtime_idle
test -z "$(process_group_members)"
printf 'installed_combined_acceptance=pass source=%s libane=%s version=%s wheel_sha256=%s core_sha256=%s libmlx_sha256=%s worker_sha256=%s ane_smokes=2 ane_executions=4 gpu_postcheck=true mel_comparisons=18 vulkan_reopen=true mel_report_sha256=%s source_build=false install=false\n' "$SOURCE_COMMIT" "$LIBANE_COMMIT" "$MLX_VERSION" "$WHEEL_SHA" "$CORE_SHA" "$INSTALLED_LIB_SHA" "$INSTALLED_WORKER_SHA" "$(sha256sum "$REPORT" | cut -d' ' -f1)"
