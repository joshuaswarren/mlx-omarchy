#!/usr/bin/env bash
set -euo pipefail

: "${1:?usage: $0 ACQUIRE_CUTOFF_EPOCH OUTER_DEADLINE_EPOCH LOG_PATH}"
: "${2:?usage: $0 ACQUIRE_CUTOFF_EPOCH OUTER_DEADLINE_EPOCH LOG_PATH}"
: "${3:?usage: $0 ACQUIRE_CUTOFF_EPOCH OUTER_DEADLINE_EPOCH LOG_PATH}"
ACQUIRE_CUTOFF_EPOCH=$1
OUTER_DEADLINE_EPOCH=$2
LOG_PATH=$3
HOST=jwm1-linux
ROOT=/var/tmp/AneRuntimeBridge-57cc36a2
SOURCE_BUNDLE_SHA=aebef2154ed0b0c543d796c4d68f52ef3670b2826e17879237dc762f589bf7af
FIXTURE_TAR=/tmp/AneRuntimeBridge-parakeet-mel-fixtures-18.tar
FIXTURE_TAR_SHA=194ee8551491b8c00b57ebc8c3ca16212d69c0588542c55c4c653bf1c0fb145f
RUNNER=/tmp/AneRuntimeBridge-57cc36a2-installed-combined.sh
RUNNER_SHA=d6eca2e9ad946873d73ac64441579add481bdf8ea77c179aa32370e6a0a176ea
REMOTE_FIXTURE_TAR="$ROOT/parakeet-mel-fixtures-18.tar"
REMOTE_FIXTURE_ROOT="$ROOT/parakeet-mel-fixtures-18"
REMOTE_RUNNER="$ROOT/installed-combined-acceptance.sh"
REPORT="$ROOT/installed-57cc-parakeet-mel-18.json"
LEASE=/tmp/m1-gpu.lease.AneRuntimeBridge
QUARANTINE=/run/lock/mlx-omarchy-ane/quarantine
SSH=(ssh -o ConnectTimeout=8 -o BatchMode=yes "$HOST")
SCP=(scp -q -o ConnectTimeout=8 -o BatchMode=yes)

printf '%s  %s\n' "$FIXTURE_TAR_SHA" "$FIXTURE_TAR" | sha256sum --check --status
printf '%s  %s\n' "$RUNNER_SHA" "$RUNNER" | sha256sum --check --status
[[ "$ACQUIRE_CUTOFF_EPOCH" =~ ^[0-9]+$ ]]
[[ "$OUTER_DEADLINE_EPOCH" =~ ^[0-9]+$ ]]
now=$(date +%s)
(( now <= ACQUIRE_CUTOFF_EPOCH ))
(( OUTER_DEADLINE_EPOCH - now >= 665 ))
mkdir -p "$(dirname "$LOG_PATH")"
test ! -e "$LOG_PATH"

REMOTE_BASE_CHECK="set -e; test \"\$(git -C '$ROOT/source' rev-parse HEAD)\" = 57cc36a2e8ece78dd979b5344752d04c76b4d49d; printf '%s  %s\n' '$SOURCE_BUNDLE_SHA' '$ROOT/source.bundle' | sha256sum --check --status; test -x '$ROOT/install/venv/bin/python'; test -s '$ROOT/installed-identity.tsv'; test -s '$ROOT/wheel-path.txt'; test \"\$(cut -f1 '$ROOT/installed-identity.tsv')\" = 0.32.2.dev202609130952+57cc36a2e8ece78dd979b5344752d04c76b4d49d; test \"\$(cut -f4 '$ROOT/installed-identity.tsv')\" = 4ad850da16c300ea9f60aa7247f034f358aa16d7ba2fd146979a01c867e3cfa5; test \"\$(cut -f6 '$ROOT/installed-identity.tsv')\" = 402d4da86cd09482344c9cf624eb085323a73d6a38790f60b9e96a5210a0a3f0; test \"\$(cut -f8 '$ROOT/installed-identity.tsv')\" = 52fa2959b0d2cfb7e7a0af877deaac605b7141b033879da1f8f5b908575e2198; test \"\$(cut -f9 '$ROOT/installed-identity.tsv')\" = 69e381c7ea91ca228a0d4ffa23e9409b3cd1b9026ed102179d05ee1043ab42d3; test \"\$(sha256sum \"\$(cat '$ROOT/wheel-path.txt')\" | cut -d' ' -f1)\" = 69e381c7ea91ca228a0d4ffa23e9409b3cd1b9026ed102179d05ee1043ab42d3; test \"\$(sha256sum '$ROOT/add/manifest.json' | cut -d' ' -f1)\" = f974e82eca49726a9eafb58f95a6578a458d423bab12efcdc3c3b780f1bbf680; test \"\$(sha256sum '$ROOT/add-mul/manifest.json' | cut -d' ' -f1)\" = 065f2acd0a0668c0911fcc728bc8bfef56812e0e64c0064de6287ac29dc98619; test ! -e '$REPORT'; printf '%s  %s\n' 4f61cd5cd1ebabb2a96d3270e347d3a6607be9b9c186f56663c5f9d14ba095c0 '$ROOT/source/overlay/tools/coreml/vulkan_mel.py' | sha256sum --check --status; printf '%s  %s\n' 210b38995c1491890fd71a7ae6c5c79b79c6d8d583289e3b3831a86b7066b7f8 '$ROOT/source/overlay/tools/coreml/mel_stage_compare.py' | sha256sum --check --status; printf '%s  %s\n' caad0ac74d60a4d7378326d5c6d2e0f0fa7d1fd807157f9d203e7348874d258f '$ROOT/source/overlay/tools/coreml/vulkan_mel_constants.py' | sha256sum --check --status; printf '%s  %s\n' 63415836246175d34748c1c44f2c8d00c79affc5be0f4d7da058eb7d5328f8d1 '$ROOT/source/overlay/tools/coreml/parakeet-reference.lock' | sha256sum --check --status"
REMOTE_FULL_CHECK="$REMOTE_BASE_CHECK; printf '%s  %s\n' '$FIXTURE_TAR_SHA' '$REMOTE_FIXTURE_TAR' | sha256sum --check --status; printf '%s  %s\n' '$RUNNER_SHA' '$REMOTE_RUNNER' | sha256sum --check --status; test -f '$REMOTE_FIXTURE_ROOT/20260912T154759Z-librispeech/ane/manifest.sha256'; test -f '$REMOTE_FIXTURE_ROOT/mel-stage-probes/stage-capture/manifest.json'"
if "${SSH[@]}" "$REMOTE_FULL_CHECK"; then
  echo "combined_staging=reused root=$REMOTE_FIXTURE_ROOT"
else
  "${SSH[@]}" "$REMOTE_BASE_CHECK; test ! -e '$REMOTE_FIXTURE_TAR'; test ! -e '$REMOTE_RUNNER'; test ! -e '$REMOTE_FIXTURE_ROOT'; test ! -e '$REMOTE_FIXTURE_ROOT.tmp'"
  "${SCP[@]}" "$FIXTURE_TAR" "$HOST:$REMOTE_FIXTURE_TAR.tmp"
  "${SCP[@]}" "$RUNNER" "$HOST:$REMOTE_RUNNER.tmp"
  "${SSH[@]}" "set -e; printf '%s  %s\n' '$FIXTURE_TAR_SHA' '$REMOTE_FIXTURE_TAR.tmp' | sha256sum --check --status; printf '%s  %s\n' '$RUNNER_SHA' '$REMOTE_RUNNER.tmp' | sha256sum --check --status; mv '$REMOTE_FIXTURE_TAR.tmp' '$REMOTE_FIXTURE_TAR'; mv '$REMOTE_RUNNER.tmp' '$REMOTE_RUNNER'; chmod 700 '$REMOTE_RUNNER'; install -d -m 700 '$REMOTE_FIXTURE_ROOT.tmp'; tar -xf '$REMOTE_FIXTURE_TAR' -C '$REMOTE_FIXTURE_ROOT.tmp'; mv '$REMOTE_FIXTURE_ROOT.tmp' '$REMOTE_FIXTURE_ROOT'; $REMOTE_FULL_CHECK"
  echo "combined_staging=created root=$REMOTE_FIXTURE_ROOT"
fi

remaining=$((OUTER_DEADLINE_EPOCH - $(date +%s)))
(( remaining >= 660 ))
set -o pipefail
set +e
timeout --signal=TERM --kill-after=5s "$((remaining + 5))s" "${SSH[@]}" \
  "set -e; ACQUIRE_CUTOFF_EPOCH=$ACQUIRE_CUTOFF_EPOCH; OUTER_DEADLINE_EPOCH=$OUTER_DEADLINE_EPOCH; $REMOTE_FULL_CHECK; remote_now=\$(date +%s); remote_session_deadline=\$((remote_now + 660)); (( remote_session_deadline > OUTER_DEADLINE_EPOCH )) && remote_session_deadline=\$OUTER_DEADLINE_EPOCH; remote_limit=\$((remote_session_deadline - remote_now - 15)); (( remote_limit >= 640 )); exec setsid --wait timeout --signal=TERM --kill-after=10s \"\$remote_limit\"s env ACQUIRE_CUTOFF_EPOCH=\$ACQUIRE_CUTOFF_EPOCH OUTER_DEADLINE_EPOCH=\$OUTER_DEADLINE_EPOCH '$REMOTE_RUNNER'" \
  2>&1 | tee "$LOG_PATH"
run_rc=${PIPESTATUS[0]}
set -e

read -r guardian_pid pgid < <(python3 - "$LOG_PATH" <<'PY'
import re
import sys
text = open(sys.argv[1], encoding='utf-8').read()
matches = re.findall(r'^guardian_identity=true guardian_pid=(\d+) comm=timeout pgid=(\d+)$', text, re.MULTILINE)
if len(matches) != 1:
    raise SystemExit(f'expected one guardian identity, found {matches}')
print(*matches[0])
PY
)

"${SSH[@]}" python3 - "$guardian_pid" "$pgid" "$LEASE" "$QUARANTINE" <<'PY'
import fcntl
import glob
import os
import stat
import sys
guardian = int(sys.argv[1])
pgid = int(sys.argv[2])
lease = sys.argv[3]
quarantine = sys.argv[4]
if os.path.exists(f'/proc/{guardian}'):
    raise SystemExit(f'guardian_residual={guardian}')
members = []
workers = []
for path in glob.glob('/proc/[0-9]*/stat'):
    try:
        text = open(path).read()
        close = text.rfind(')')
        pid = int(text[:text.find(' ')])
        fields = text[close + 2:].split()
        if int(fields[2]) == pgid:
            members.append(pid)
        cmdline = open(path.rsplit('/', 1)[0] + '/cmdline', 'rb').read().split(b'\0', 1)[0]
        if os.path.basename(os.fsdecode(cmdline)) == 'mlx-omarchy-ane-worker':
            workers.append(pid)
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, IndexError):
        pass
if members:
    raise SystemExit(f'process_group_residual={members}')
if workers:
    raise SystemExit(f'worker_residual={workers}')
if os.path.exists(lease):
    raise SystemExit(f'lease_residual={lease}')
value = os.stat(quarantine, follow_symlinks=False)
if not stat.S_ISREG(value.st_mode) or value.st_size != 0:
    raise SystemExit(f'quarantine_not_empty mode={oct(value.st_mode)} size={value.st_size}')
fd = os.open('/tmp/m1-gpu.lock', os.O_WRONLY)
try:
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    fcntl.flock(fd, fcntl.LOCK_UN)
finally:
    os.close(fd)
print(f'independent_clearance=pass guardian_absent={guardian} pgid_empty={pgid} workers_absent=true lease_absent=true quarantine_empty=true lock_free=true')
PY

exit "$run_rc"
