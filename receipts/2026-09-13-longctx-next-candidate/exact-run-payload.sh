#!/usr/bin/env bash
set -euo pipefail

: "${ACQUIRE_CUTOFF_EPOCH:?ACQUIRE_CUTOFF_EPOCH is required}"
: "${SESSION_DEADLINE_EPOCH:?SESSION_DEADLINE_EPOCH is required}"
: "${EXPECTED_HOST:?EXPECTED_HOST is required}"
: "${LOCK_PATH:?LOCK_PATH is required}"
: "${LEASE_PATH:?LEASE_PATH is required}"
: "${WORKLOAD_PATH:?WORKLOAD_PATH is required}"
: "${WORKLOAD_SHA256:?WORKLOAD_SHA256 is required}"
: "${RUN_LABEL:?RUN_LABEL is required}"
if [[ "${LONGCTX_SUBREAPER_ACTIVE:-0}" != 1 ]]; then
  exec python3 - "$0" <<'PY'
import ctypes
import os
import shutil
import sys

libc = ctypes.CDLL(None, use_errno=True)
if libc.prctl(36, 1, 0, 0, 0) != 0:
    raise OSError(ctypes.get_errno(), "PR_SET_CHILD_SUBREAPER")
env = os.environ.copy()
env["LONGCTX_SUBREAPER_ACTIVE"] = "1"
bash = shutil.which("bash")
assert bash is not None
os.execve(bash, [bash, sys.argv[1]], env)
PY
fi

CLEANUP_RESERVE_SECONDS=60
WORK_DEADLINE_EPOCH=0
GUARDIAN_PID=
PGID=

remaining_work_seconds() {
  echo $((WORK_DEADLINE_EPOCH - $(date +%s)))
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
excluded = {script, guardian, me, scanner_parent}
table = {}
for path in glob.glob("/proc/[0-9]*/stat"):
    try:
        text = open(path).read()
        close = text.rfind(")")
        pid = int(text[:text.find(" ")])
        fields = text[close + 2:].split()
        table[pid] = (int(fields[1]), int(fields[2]), fields[0])
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, IndexError):
        pass
owned = {pid for pid, (_, pgid, _) in table.items() if pgid == wanted and pid not in excluded}
changed = True
while changed:
    changed = False
    for pid, (ppid, _, _) in table.items():
        if pid not in excluded and pid not in owned and (ppid == script or ppid in owned):
            owned.add(pid)
            changed = True
for pid in sorted(owned):
    ppid, pgid, state = table[pid]
    print(f"pid={pid} ppid={ppid} pgid={pgid} state={state}")
PY
}

terminate_process_group_members() {
  [[ -n "$PGID" && -n "$GUARDIAN_PID" ]] || return 0
  python3 - "$PGID" "$$" "$GUARDIAN_PID" <<'PY'
import glob
import os
import signal
import sys
import time

wanted = int(sys.argv[1])
script = int(sys.argv[2])
guardian = int(sys.argv[3])
me = os.getpid()
scanner_parent = os.getppid()
excluded = {script, guardian, me, scanner_parent}

def members():
    table = {}
    for path in glob.glob("/proc/[0-9]*/stat"):
        try:
            text = open(path).read()
            close = text.rfind(")")
            pid = int(text[:text.find(" ")])
            fields = text[close + 2:].split()
            table[pid] = (int(fields[1]), int(fields[2]))
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, IndexError):
            pass
    owned = {pid for pid, (_, pgid) in table.items() if pgid == wanted and pid not in excluded}
    changed = True
    while changed:
        changed = False
        for pid, (ppid, _) in table.items():
            if pid not in excluded and pid not in owned and (ppid == script or ppid in owned):
                owned.add(pid)
                changed = True
    return sorted(owned)

targets = members()
for pid in targets:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
deadline = time.monotonic() + 2.0
while time.monotonic() < deadline and members():
    time.sleep(0.05)
survivors = members()
for pid in survivors:
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
deadline = time.monotonic() + 2.0
while time.monotonic() < deadline and members():
    time.sleep(0.05)
survivors = members()
if survivors:
    raise SystemExit(f"descendant_termination_failed={survivors}")
print("terminated_descendants=" + ",".join(map(str, targets)))
PY
}

finish() {
  local rc=$?
  trap - EXIT INT TERM
  set +e
  local initial_group_members terminated_members group_members lock_state group_scan finished_epoch deadline_state
  group_scan=pass
  terminated_members=none
  initial_group_members=$(process_group_members) || group_scan=failed
  group_members=$initial_group_members
  if [[ "$group_scan" == pass && -n "$initial_group_members" ]]; then
    if terminated_members=$(terminate_process_group_members); then
      wait 2>/dev/null || true
      group_members=$(process_group_members) || group_scan=failed
    else
      group_scan=failed
      group_members=$(process_group_members)
    fi
  fi
  flock -u 9
  exec 9>&-
  exec 8>"$LOCK_PATH"
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
  if [[ "$group_scan" == pass && -z "$group_members" && "$lock_state" == free && "$deadline_state" == within ]] && rm -f "$LEASE_PATH" && [[ ! -e "$LEASE_PATH" ]]; then
    echo "clearance=true outer_lock=free process_group_scan=pass process_group=empty descendant_cleanup=$terminated_members lease=removed finished_epoch=$finished_epoch session_deadline_epoch=$SESSION_DEADLINE_EPOCH cleanup_deadline=$deadline_state exit_rc=$rc"
  else
    echo "clearance=false outer_lock=$lock_state process_group_scan=$group_scan process_group=${group_members:-empty} descendant_cleanup=$terminated_members lease_present=$([[ -e "$LEASE_PATH" ]] && echo true || echo false) finished_epoch=$finished_epoch session_deadline_epoch=$SESSION_DEADLINE_EPOCH cleanup_deadline=$deadline_state exit_rc=$rc"
    [[ -n "$group_members" ]] && printf '%s\n' "$group_members"
    exit 125
  fi
  exit "$rc"
}

capture_guardian
[[ "$(hostname)" == "$EXPECTED_HOST" ]]
for command in bash cat cut date flock grep hostname ps python3 rm setsid sha256sum sleep timeout tr; do
  command -v "$command" >/dev/null
done
[[ "$ACQUIRE_CUTOFF_EPOCH" =~ ^[0-9]+$ ]]
[[ "$SESSION_DEADLINE_EPOCH" =~ ^[0-9]+$ ]]
[[ "$WORKLOAD_SHA256" =~ ^[0-9a-f]{64}$ ]]
[[ "$RUN_LABEL" =~ ^[A-Za-z0-9._-]+$ ]]
printf '%s  %s\n' "$WORKLOAD_SHA256" "$WORKLOAD_PATH" | sha256sum --check --status
now=$(date +%s)
(( now <= ACQUIRE_CUTOFF_EPOCH )) || {
  echo "acquire_cutoff_missed=true now=$now cutoff=$ACQUIRE_CUTOFF_EPOCH"
  exit 76
}
(( SESSION_DEADLINE_EPOCH - now > CLEANUP_RESERVE_SECONDS )) || {
  echo "session_deadline_too_short=true remaining=$((SESSION_DEADLINE_EPOCH - now)) cleanup_reserve=$CLEANUP_RESERVE_SECONDS"
  exit 124
}
WORK_DEADLINE_EPOCH=$((SESSION_DEADLINE_EPOCH - CLEANUP_RESERVE_SECONDS))
exec 9>"$LOCK_PATH"
if ! flock -n 9; then
  echo "lock_acquired=false time=$(date -Is)"
  exit 75
fi
if [[ -e "$LEASE_PATH" ]]; then
  echo "lease_acquired=false reason=existing_retained_lease lease_sha256=$(sha256sum "$LEASE_PATH" | cut -d' ' -f1)"
  flock -u 9
  exec 9>&-
  exit 78
fi
[[ "$(ps -o pgid= -p $$ | tr -d ' ')" == "$PGID" ]]
[[ "$PPID" == "$GUARDIAN_PID" ]]
[[ "$(cat "/proc/$GUARDIAN_PID/comm")" == timeout ]]
if ! (set -o noclobber; printf 'agent=LongContextCostAttribution pid=%s pgid=%s guardian_pid=%s started=%s work_deadline_epoch=%s session_deadline_epoch=%s cleanup_reserve_seconds=%s workload_sha256=%s run_label=%s\n' \
  "$$" "$PGID" "$GUARDIAN_PID" "$(date -Is)" "$WORK_DEADLINE_EPOCH" "$SESSION_DEADLINE_EPOCH" "$CLEANUP_RESERVE_SECONDS" "$WORKLOAD_SHA256" "$RUN_LABEL" >"$LEASE_PATH") 2>/dev/null; then
  echo "lease_acquired=false reason=atomic_create_refused"
  flock -u 9
  exec 9>&-
  exit 78
fi
echo "lock_acquired=true pid=$$ pgid=$PGID guardian_pid=$GUARDIAN_PID work_deadline_epoch=$WORK_DEADLINE_EPOCH session_deadline_epoch=$SESSION_DEADLINE_EPOCH"
trap finish EXIT INT TERM

if [[ "${LIFECYCLE_SMOKE_FAKE_CHILD:-0}" == 1 ]]; then
  sleep 30 &
  fake_child=$!
  fake_members=$(process_group_members)
  grep -Fq "pid=$fake_child " <<<"$fake_members"
  echo "fake_child_detected=true pid=$fake_child"
  echo "fake_child_left_for_cleanup=true pid=$fake_child"
fi
if [[ "${LIFECYCLE_SMOKE_DETACHED_CHILD:-0}" == 1 ]]; then
  setsid sleep 30 </dev/null >/dev/null 2>&1 &
  fake_detached_child=$!
  sleep 0.1
  fake_detached_pgid=$(ps -o pgid= -p "$fake_detached_child" | tr -d ' ')
  [[ "$fake_detached_pgid" == "$fake_detached_child" && "$fake_detached_pgid" != "$PGID" ]]
  fake_members=$(process_group_members)
  grep -Fq "pid=$fake_detached_child " <<<"$fake_members"
  echo "fake_detached_child_detected=true pid=$fake_detached_child pgid=$fake_detached_pgid"
  echo "fake_detached_child_left_for_cleanup=true pid=$fake_detached_child"
fi

work_seconds=$(remaining_work_seconds)
(( work_seconds > 0 )) || {
  echo "work_deadline_expired=true"
  exit 124
}
timeout --foreground --signal=TERM --kill-after=5s "${work_seconds}s" bash "$WORKLOAD_PATH"
echo "workload_complete=true run_label=$RUN_LABEL"
