#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
OUTPUT=${1:-"$ROOT/failure-smoke-receipt.txt"}
PAYLOAD="$ROOT/exact-run-payload.sh"
TMP=$(mktemp -d)
cleanup() {
  rm -f "$TMP/lock" "$TMP/lease" "$TMP/payload-scan-fail.sh" \
    "$TMP/workload-ok.sh" "$TMP/workload-124.sh" "$TMP/scan-fail.log" \
    "$TMP/retained-lease.log" "$TMP/exit-124.log"
  rmdir "$TMP"
}
trap cleanup EXIT

python3 - "$PAYLOAD" "$TMP/payload-scan-fail.sh" <<'PY'
from pathlib import Path
import sys
lines = Path(sys.argv[1]).read_text().splitlines()
index = lines.index("process_group_members() {")
assert lines[index + 1] == '  [[ -n "$PGID" && -n "$GUARDIAN_PID" ]] || return 0'
lines.insert(index + 1, "  return 97")
Path(sys.argv[2]).write_text("\n".join(lines) + "\n")
PY
printf '#!/usr/bin/env bash\nset -euo pipefail\necho workload_ok=true\n' >"$TMP/workload-ok.sh"
printf '#!/usr/bin/env bash\nset -euo pipefail\necho workload_exit_124=true\nexit 124\n' >"$TMP/workload-124.sh"
chmod 500 "$TMP/payload-scan-fail.sh" "$TMP/workload-ok.sh" "$TMP/workload-124.sh"
: >"$OUTPUT"

run_payload() {
  local label=$1 payload=$2 workload=$3 log=$4
  local now cutoff session_deadline workload_sha
  now=$(date +%s)
  cutoff=$((now + 5))
  session_deadline=$((now + 65))
  workload_sha=$(sha256sum "$workload" | cut -d' ' -f1)
  set +e
  timeout --signal=TERM --kill-after=2s 15s \
    setsid --wait timeout --signal=TERM --kill-after=2s 10s \
    env ACQUIRE_CUTOFF_EPOCH="$cutoff" SESSION_DEADLINE_EPOCH="$session_deadline" \
      EXPECTED_HOST="$(hostname)" LOCK_PATH="$TMP/lock" LEASE_PATH="$TMP/lease" \
      WORKLOAD_PATH="$workload" WORKLOAD_SHA256="$workload_sha" \
      RUN_LABEL="$label" bash "$payload" >"$log" 2>&1
  case_rc=$?
  set -e
}

run_payload scanner-failure "$TMP/payload-scan-fail.sh" "$TMP/workload-ok.sh" "$TMP/scan-fail.log"
[[ "$case_rc" == 125 ]]
grep -Fq 'clearance=false outer_lock=free process_group_scan=failed' "$TMP/scan-fail.log"
grep -Fq 'lease_present=true' "$TMP/scan-fail.log"
[[ -s "$TMP/lease" ]]
lease_sha_before=$(sha256sum "$TMP/lease" | cut -d' ' -f1)
printf 'scanner_failure_case_rc=%s lease_retained=true lock=' "$case_rc" >>"$OUTPUT"
if flock -n "$TMP/lock" true; then
  echo free >>"$OUTPUT"
else
  echo busy >>"$OUTPUT"
  exit 1
fi
cat "$TMP/scan-fail.log" >>"$OUTPUT"
run_payload retained-lease-refusal "$PAYLOAD" "$TMP/workload-ok.sh" "$TMP/retained-lease.log"
[[ "$case_rc" == 78 ]]
grep -Fq 'lease_acquired=false reason=existing_retained_lease' "$TMP/retained-lease.log"
lease_sha_after=$(sha256sum "$TMP/lease" | cut -d' ' -f1)
[[ "$lease_sha_after" == "$lease_sha_before" ]]
printf 'retained_lease_refusal_rc=%s lease_unchanged=true lease_sha256=%s\n' \
  "$case_rc" "$lease_sha_after" >>"$OUTPUT"
cat "$TMP/retained-lease.log" >>"$OUTPUT"
rm -f "$TMP/lease"

run_payload exit-124 "$PAYLOAD" "$TMP/workload-124.sh" "$TMP/exit-124.log"
[[ "$case_rc" == 124 ]]
grep -Fq 'clearance=true outer_lock=free process_group_scan=pass process_group=empty' "$TMP/exit-124.log"
grep -Fq 'cleanup_deadline=within exit_rc=124' "$TMP/exit-124.log"
[[ ! -e "$TMP/lease" ]]
printf 'exit_124_case_rc=%s lease_removed=true lock=' "$case_rc" >>"$OUTPUT"
if flock -n "$TMP/lock" true; then
  echo free >>"$OUTPUT"
else
  echo busy >>"$OUTPUT"
  exit 1
fi
cat "$TMP/exit-124.log" >>"$OUTPUT"
cat "$OUTPUT"
