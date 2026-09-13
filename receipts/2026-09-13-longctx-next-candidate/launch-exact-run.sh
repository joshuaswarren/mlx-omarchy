#!/usr/bin/env bash
set -euo pipefail

: "${1:?usage: $0 RUN_LABEL ACQUIRE_CUTOFF_EPOCH OUTER_DEADLINE_EPOCH LOG_PATH LOCAL_WORKLOAD}"
: "${2:?usage: $0 RUN_LABEL ACQUIRE_CUTOFF_EPOCH OUTER_DEADLINE_EPOCH LOG_PATH LOCAL_WORKLOAD}"
: "${3:?usage: $0 RUN_LABEL ACQUIRE_CUTOFF_EPOCH OUTER_DEADLINE_EPOCH LOG_PATH LOCAL_WORKLOAD}"
: "${4:?usage: $0 RUN_LABEL ACQUIRE_CUTOFF_EPOCH OUTER_DEADLINE_EPOCH LOG_PATH LOCAL_WORKLOAD}"
: "${5:?usage: $0 RUN_LABEL ACQUIRE_CUTOFF_EPOCH OUTER_DEADLINE_EPOCH LOG_PATH LOCAL_WORKLOAD}"
RUN_LABEL=$1
ACQUIRE_CUTOFF_EPOCH=$2
OUTER_DEADLINE_EPOCH=$3
LOG_PATH=$4
LOCAL_WORKLOAD=$(readlink -f "$5")
HOST=jwm1-linux
HERE=$(cd "$(dirname "$0")" && pwd)
PAYLOAD="$HERE/exact-run-payload.sh"
PAYLOAD_SHA=5f74d4cd7afd81875b54cf33b777ff1028219c6139928dae3293b0fd71bf7726
WORKLOAD_SHA=$(sha256sum "$LOCAL_WORKLOAD" | cut -d' ' -f1)
ROOT="/var/tmp/LongContextCostAttribution-$RUN_LABEL"
REMOTE_PAYLOAD="$ROOT/payload.sh"
REMOTE_WORKLOAD="$ROOT/workload.sh"
LEASE=/tmp/m1-gpu.lease.LongContextCostAttribution
LOCK=/tmp/m1-gpu.lock
SSH=(ssh -o ConnectTimeout=8 -o BatchMode=yes "$HOST")
SCP=(scp -q -o ConnectTimeout=8 -o BatchMode=yes)

[[ "$RUN_LABEL" =~ ^[A-Za-z0-9._-]+$ ]]
[[ "$ACQUIRE_CUTOFF_EPOCH" =~ ^[0-9]+$ ]]
[[ "$OUTER_DEADLINE_EPOCH" =~ ^[0-9]+$ ]]
printf '%s  %s\n' "$PAYLOAD_SHA" "$PAYLOAD" | sha256sum --check --status
[[ -f "$LOCAL_WORKLOAD" ]]
[[ "$WORKLOAD_SHA" =~ ^[0-9a-f]{64}$ ]]
now=$(date +%s)
(( now <= ACQUIRE_CUTOFF_EPOCH ))
(( OUTER_DEADLINE_EPOCH - now >= 820 ))
mkdir -p "$(dirname "$LOG_PATH")"
test ! -e "$LOG_PATH"

REMOTE_STAGE_CHECK="set -e; printf '%s  %s\\n' '$PAYLOAD_SHA' '$REMOTE_PAYLOAD' | sha256sum --check --status; printf '%s  %s\\n' '$WORKLOAD_SHA' '$REMOTE_WORKLOAD' | sha256sum --check --status; test -f '$REMOTE_PAYLOAD'; test -f '$REMOTE_WORKLOAD'"
if "${SSH[@]}" "$REMOTE_STAGE_CHECK"; then
  echo "immutable_staging=reused root=$ROOT"
else
  "${SSH[@]}" "test ! -e '$ROOT' && install -d -m 700 '$ROOT'"
  "${SCP[@]}" "$PAYLOAD" "$HOST:$REMOTE_PAYLOAD.tmp"
  "${SCP[@]}" "$LOCAL_WORKLOAD" "$HOST:$REMOTE_WORKLOAD.tmp"
  "${SSH[@]}" "set -e; printf '%s  %s\\n' '$PAYLOAD_SHA' '$REMOTE_PAYLOAD.tmp' | sha256sum --check --status; printf '%s  %s\\n' '$WORKLOAD_SHA' '$REMOTE_WORKLOAD.tmp' | sha256sum --check --status; mv '$REMOTE_PAYLOAD.tmp' '$REMOTE_PAYLOAD'; mv '$REMOTE_WORKLOAD.tmp' '$REMOTE_WORKLOAD'; chmod 500 '$REMOTE_PAYLOAD' '$REMOTE_WORKLOAD'; $REMOTE_STAGE_CHECK"
  echo "immutable_staging=created root=$ROOT"
fi

remaining=$((OUTER_DEADLINE_EPOCH - $(date +%s)))
(( remaining >= 820 ))
set -o pipefail
set +e
timeout --signal=TERM --kill-after=5s "$((remaining + 5))s" "${SSH[@]}" \
  "set -e; ACQUIRE_CUTOFF_EPOCH=$ACQUIRE_CUTOFF_EPOCH; OUTER_DEADLINE_EPOCH=$OUTER_DEADLINE_EPOCH; $REMOTE_STAGE_CHECK; remote_now=\$(date +%s); remote_session_deadline=\$((remote_now + 900)); (( remote_session_deadline > OUTER_DEADLINE_EPOCH )) && remote_session_deadline=\$OUTER_DEADLINE_EPOCH; remote_limit=\$((remote_session_deadline - remote_now - 15)); (( remote_limit >= 805 )); exec setsid --wait timeout --signal=TERM --kill-after=10s \"\$remote_limit\"s env ACQUIRE_CUTOFF_EPOCH=$ACQUIRE_CUTOFF_EPOCH SESSION_DEADLINE_EPOCH=\$remote_session_deadline EXPECTED_HOST=$HOST LOCK_PATH=$LOCK LEASE_PATH=$LEASE WORKLOAD_PATH=$REMOTE_WORKLOAD WORKLOAD_SHA256=$WORKLOAD_SHA RUN_LABEL=$RUN_LABEL bash '$REMOTE_PAYLOAD'" \
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

"${SSH[@]}" python3 - "$guardian_pid" "$pgid" "$LEASE" "$LOCK" "$ROOT" <<'PY'
import fcntl
import glob
import os
import sys
guardian = int(sys.argv[1])
pgid = int(sys.argv[2])
lease = sys.argv[3]
lock = sys.argv[4]
root = os.fsencode(sys.argv[5])
me = os.getpid()
excluded = set()
pid = me
while pid > 1 and pid not in excluded:
    excluded.add(pid)
    try:
        text = open(f"/proc/{pid}/stat").read()
        close = text.rfind(")")
        pid = int(text[close + 2:].split()[1])
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, IndexError):
        break
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
        cmdline = open(path.rsplit("/", 1)[0] + "/cmdline", "rb").read()
        if pid not in excluded and root in cmdline:
            workers.append(pid)
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, IndexError):
        pass
if members:
    raise SystemExit(f"process_group_residual={members}")
if workers:
    raise SystemExit(f"staged_workload_residual={workers}")
if os.path.exists(lease):
    raise SystemExit(f"lease_residual={lease}")
fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
try:
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
finally:
    os.close(fd)
print(f"independent_clearance=true guardian=absent pgid={pgid} process_group=empty staged_workload=absent lease=absent lock=free")
PY

[[ "$run_rc" == 0 ]]
grep -Fq 'clearance=true outer_lock=free process_group_scan=pass process_group=empty lease=removed' "$LOG_PATH"
echo "launch_complete=true run_label=$RUN_LABEL runner_sha256=$PAYLOAD_SHA workload_sha256=$WORKLOAD_SHA"
