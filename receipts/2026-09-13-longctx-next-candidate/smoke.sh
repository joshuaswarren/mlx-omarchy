#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
OUTPUT=${1:-"$ROOT/smoke-receipt.txt"}
PAYLOAD="$ROOT/exact-run-payload.sh"
WORKLOAD="$ROOT/smoke-workload.sh"
PAYLOAD_SHA=$(sha256sum "$PAYLOAD" | cut -d' ' -f1)
WORKLOAD_SHA=$(sha256sum "$WORKLOAD" | cut -d' ' -f1)
TMP=$(mktemp -d)
cleanup() {
  rm -f "$TMP/lock" "$TMP/lease" "$TMP/run.log"
  rmdir "$TMP"
}
trap cleanup EXIT
now=$(date +%s)
cutoff=$((now + 5))
session_deadline=$((now + 65))

set +e
timeout --signal=TERM --kill-after=2s 15s \
  setsid --wait timeout --signal=TERM --kill-after=2s 10s \
  env ACQUIRE_CUTOFF_EPOCH="$cutoff" SESSION_DEADLINE_EPOCH="$session_deadline" \
    EXPECTED_HOST="$(hostname)" LOCK_PATH="$TMP/lock" LEASE_PATH="$TMP/lease" \
    WORKLOAD_PATH="$WORKLOAD" WORKLOAD_SHA256="$WORKLOAD_SHA" \
    RUN_LABEL=bounded-fake-child-smoke LIFECYCLE_SMOKE_FAKE_CHILD=1 \
    bash "$PAYLOAD" >"$TMP/run.log" 2>&1
run_rc=$?
set -e
cat "$TMP/run.log" >"$OUTPUT"
[[ "$run_rc" == 0 ]]
grep -Fq 'fake_child_detected=true' "$TMP/run.log"
grep -Fq 'fake_child_left_for_cleanup=true' "$TMP/run.log"
grep -Fq 'workload_complete=true run_label=bounded-fake-child-smoke' "$TMP/run.log"
grep -Fq 'clearance=true outer_lock=free process_group_scan=pass process_group=empty descendant_cleanup=terminated_descendants=' "$TMP/run.log"

read -r guardian_pid pgid < <(python3 - "$TMP/run.log" <<'PY'
import re
import sys
text = open(sys.argv[1], encoding="utf-8").read()
matches = re.findall(r"^guardian_identity=true guardian_pid=(\d+) comm=timeout pgid=(\d+)$", text, re.MULTILINE)
if len(matches) != 1:
    raise SystemExit(f"expected one guardian identity, found {matches}")
print(*matches[0])
PY
)

python3 - "$guardian_pid" "$pgid" "$TMP/lease" "$TMP/lock" <<'PY' >>"$OUTPUT"
import fcntl
import glob
import os
import sys
guardian = int(sys.argv[1])
pgid = int(sys.argv[2])
lease = sys.argv[3]
lock = sys.argv[4]
if os.path.exists(f"/proc/{guardian}"):
    raise SystemExit(f"guardian_residual={guardian}")
members = []
for path in glob.glob("/proc/[0-9]*/stat"):
    try:
        text = open(path).read()
        close = text.rfind(")")
        pid = int(text[:text.find(" ")])
        fields = text[close + 2:].split()
        if int(fields[2]) == pgid:
            members.append(pid)
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, IndexError):
        pass
if members:
    raise SystemExit(f"process_group_residual={members}")
if os.path.exists(lease):
    raise SystemExit(f"lease_residual={lease}")
fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
try:
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
finally:
    os.close(fd)
print(f"independent_clearance=true guardian=absent pgid={pgid} process_group=empty lease=absent lock=free")
PY
printf 'payload_sha256=%s workload_sha256=%s run_rc=%s\n' "$PAYLOAD_SHA" "$WORKLOAD_SHA" "$run_rc" >>"$OUTPUT"
cat "$OUTPUT"
