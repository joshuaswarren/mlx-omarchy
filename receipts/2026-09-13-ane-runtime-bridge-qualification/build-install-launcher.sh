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
SOURCE_BUNDLE=/tmp/AneRuntimeBridge-57cc36a2.bundle
LIBANE_BUNDLE=/tmp/AneRuntimeBridge-libane-f261.bundle
FIXTURES=/tmp/AneRuntimeBridge-3172-fixtures.tar
RUNNER=/tmp/AneRuntimeBridge-57cc36a2-acceptance.sh
SOURCE_SHA=aebef2154ed0b0c543d796c4d68f52ef3670b2826e17879237dc762f589bf7af
LIBANE_SHA=a2e33cb7deadba974a5d89e669a8982ea0ed40483d9c87b18688d2c0ff599727
FIXTURES_SHA=88a3f5ffe800af52ca270311e33b6153efdd7860f0332bec4d37ddd8fc956c76
RUNNER_SHA=1ab7d3f2072f2191e0c85671f23b29667c2d6e128e6abbde1658ebe940df3c31
SSH=(ssh -o ConnectTimeout=8 -o BatchMode=yes "$HOST")
SCP=(scp -q -o ConnectTimeout=8 -o BatchMode=yes)
LEASE=/tmp/m1-gpu.lease.AneRuntimeBridge
QUARANTINE=/run/lock/mlx-omarchy-ane/quarantine

printf '%s  %s\n' "$SOURCE_SHA" "$SOURCE_BUNDLE" | sha256sum --check --status
printf '%s  %s\n' "$LIBANE_SHA" "$LIBANE_BUNDLE" | sha256sum --check --status
printf '%s  %s\n' "$FIXTURES_SHA" "$FIXTURES" | sha256sum --check --status
printf '%s  %s\n' "$RUNNER_SHA" "$RUNNER" | sha256sum --check --status
[[ "$ACQUIRE_CUTOFF_EPOCH" =~ ^[0-9]+$ ]]
[[ "$OUTER_DEADLINE_EPOCH" =~ ^[0-9]+$ ]]
now=$(date +%s)
(( now <= ACQUIRE_CUTOFF_EPOCH ))
(( OUTER_DEADLINE_EPOCH - now >= 840 ))
mkdir -p "$(dirname "$LOG_PATH")"
test ! -e "$LOG_PATH"

REMOTE_STAGE_CHECK="set -e; printf '%s  %s\n' '$SOURCE_SHA' '$ROOT/source.bundle' | sha256sum --check --status; printf '%s  %s\n' '$LIBANE_SHA' '$ROOT/libane.bundle' | sha256sum --check --status; printf '%s  %s\n' '$FIXTURES_SHA' '$ROOT/fixtures.tar' | sha256sum --check --status; printf '%s  %s\n' '$RUNNER_SHA' '$ROOT/acceptance.sh' | sha256sum --check --status; test \"\$(sha256sum '$ROOT/add/manifest.json' | cut -d' ' -f1)\" = f974e82eca49726a9eafb58f95a6578a458d423bab12efcdc3c3b780f1bbf680; test \"\$(sha256sum '$ROOT/add-mul/manifest.json' | cut -d' ' -f1)\" = 065f2acd0a0668c0911fcc728bc8bfef56812e0e64c0064de6287ac29dc98619; test ! -e '$ROOT/source'; test ! -e '$ROOT/libane'; test ! -e '$ROOT/wheel-work'; test ! -e '$ROOT/release'; test ! -e '$ROOT/install'"
if "${SSH[@]}" "$REMOTE_STAGE_CHECK"; then
  echo "immutable_staging=reused root=$ROOT"
else
  "${SSH[@]}" "test ! -e '$ROOT' && install -d -m 700 '$ROOT'"
  "${SCP[@]}" "$SOURCE_BUNDLE" "$HOST:$ROOT/source.bundle.tmp"
  "${SCP[@]}" "$LIBANE_BUNDLE" "$HOST:$ROOT/libane.bundle.tmp"
  "${SCP[@]}" "$FIXTURES" "$HOST:$ROOT/fixtures.tar.tmp"
  "${SCP[@]}" "$RUNNER" "$HOST:$ROOT/acceptance.sh.tmp"
  "${SSH[@]}" "set -e; printf '%s  %s\n' '$SOURCE_SHA' '$ROOT/source.bundle.tmp' | sha256sum --check --status; printf '%s  %s\n' '$LIBANE_SHA' '$ROOT/libane.bundle.tmp' | sha256sum --check --status; printf '%s  %s\n' '$FIXTURES_SHA' '$ROOT/fixtures.tar.tmp' | sha256sum --check --status; printf '%s  %s\n' '$RUNNER_SHA' '$ROOT/acceptance.sh.tmp' | sha256sum --check --status; mv '$ROOT/source.bundle.tmp' '$ROOT/source.bundle'; mv '$ROOT/libane.bundle.tmp' '$ROOT/libane.bundle'; mv '$ROOT/fixtures.tar.tmp' '$ROOT/fixtures.tar'; mv '$ROOT/acceptance.sh.tmp' '$ROOT/acceptance.sh'; chmod 700 '$ROOT/acceptance.sh'; tar -xf '$ROOT/fixtures.tar' -C '$ROOT'; $REMOTE_STAGE_CHECK"
  echo "immutable_staging=created root=$ROOT"
fi

remaining=$((OUTER_DEADLINE_EPOCH - $(date +%s)))
(( remaining >= 820 ))
set -o pipefail
set +e
timeout --signal=TERM --kill-after=5s "$((remaining + 5))s" "${SSH[@]}" \
  "set -e; ACQUIRE_CUTOFF_EPOCH=$ACQUIRE_CUTOFF_EPOCH; OUTER_DEADLINE_EPOCH=$OUTER_DEADLINE_EPOCH; $REMOTE_STAGE_CHECK; remote_now=\$(date +%s); remote_session_deadline=\$((remote_now + 900)); (( remote_session_deadline > OUTER_DEADLINE_EPOCH )) && remote_session_deadline=\$OUTER_DEADLINE_EPOCH; remote_limit=\$((remote_session_deadline - remote_now - 15)); (( remote_limit >= 805 )); exec setsid --wait timeout --signal=TERM --kill-after=10s \"\$remote_limit\"s env ACQUIRE_CUTOFF_EPOCH=\$ACQUIRE_CUTOFF_EPOCH OUTER_DEADLINE_EPOCH=\$OUTER_DEADLINE_EPOCH '$ROOT/acceptance.sh'" \
  2>&1 | tee "$LOG_PATH"
run_rc=${PIPESTATUS[0]}
set -e

read -r guardian_pid pgid < <(python3 - "$LOG_PATH" <<'PY'
import re
import sys
text = open(sys.argv[1], encoding="utf-8").read()
matches = re.findall(r"^guardian_identity=true guardian_pid=(\d+) comm=timeout pgid=(\d+)$", text, re.MULTILINE)
if len(matches) != 1:
    raise SystemExit(f"expected one guardian identity, found {matches}")
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
if os.path.exists(f"/proc/{guardian}"):
    raise SystemExit(f"guardian_residual={guardian}")
members = []
workers = []
for path in glob.glob("/proc/[0-9]*/stat"):
    try:
        text = open(path).read()
        close = text.rfind(")")
        pid = int(text[:text.find(" ")])
        fields = text[close + 2:].split()
        if int(fields[2]) == pgid:
            members.append(pid)
        cmdline = open(path.rsplit("/", 1)[0] + "/cmdline", "rb").read().split(b"\0", 1)[0]
        if os.path.basename(os.fsdecode(cmdline)) == "mlx-omarchy-ane-worker":
            workers.append(pid)
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, IndexError):
        pass
if members:
    raise SystemExit(f"process_group_residual={members}")
if workers:
    raise SystemExit(f"worker_residual={workers}")
if os.path.exists(lease):
    raise SystemExit(f"lease_residual={lease}")
value = os.stat(quarantine, follow_symlinks=False)
if not stat.S_ISREG(value.st_mode) or value.st_size != 0:
    raise SystemExit(f"quarantine_not_empty mode={oct(value.st_mode)} size={value.st_size}")
fd = os.open("/tmp/m1-gpu.lock", os.O_WRONLY)
try:
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    fcntl.flock(fd, fcntl.LOCK_UN)
finally:
    os.close(fd)
print(f"independent_clearance=pass guardian_absent={guardian} pgid_empty={pgid} workers_absent=true lease_absent=true quarantine_empty=true lock_free=true")
PY

exit "$run_rc"
