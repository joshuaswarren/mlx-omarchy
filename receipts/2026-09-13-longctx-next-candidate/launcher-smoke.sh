#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
TMP=$(mktemp -d /var/tmp/longctx-launcher-smoke.XXXXXX)
RUN_LABEL="launcher-local-$$"
REMOTE_ROOT="/var/tmp/LongContextCostAttribution-$RUN_LABEL"
DEADLINE_RUN_LABEL="launcher-deadline-local-$$"
DEADLINE_REMOTE_ROOT="/var/tmp/LongContextCostAttribution-$DEADLINE_RUN_LABEL"
RECEIPT="$HERE/launcher-smoke-receipt.txt"
cleanup() {
  local rc=$?
  trap - EXIT
  [[ "$TMP" == /var/tmp/longctx-launcher-smoke.* ]]
  [[ "$REMOTE_ROOT" == /var/tmp/LongContextCostAttribution-launcher-local-* ]]
  [[ "$DEADLINE_REMOTE_ROOT" == /var/tmp/LongContextCostAttribution-launcher-deadline-local-* ]]
  rm -r -- "$TMP"
  if [ -e "$REMOTE_ROOT" ]; then
    rm -r -- "$REMOTE_ROOT"
  fi
  if [ -e "$DEADLINE_REMOTE_ROOT" ]; then
    rm -r -- "$DEADLINE_REMOTE_ROOT"
  fi
  exit "$rc"
}
trap cleanup EXIT

mkdir "$TMP/bin"
cat > "$TMP/bin/ssh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
while (($#)); do
  case "$1" in
    -o)
      shift 2
      ;;
    *)
      shift
      break
      ;;
  esac
done
if (($# == 1)); then
  exec bash -c "$1"
fi
exec "$@"
SH
cat > "$TMP/bin/scp" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
while (($#)); do
  case "$1" in
    -q)
      shift
      ;;
    -o)
      shift 2
      ;;
    *)
      break
      ;;
  esac
done
src=$1
dest=${2#*:}
cp -- "$src" "$dest"
SH
cat > "$TMP/bin/hostname" <<'SH'
#!/usr/bin/env bash
printf 'jwm1-linux\n'
SH
chmod 700 "$TMP/bin/ssh" "$TMP/bin/scp" "$TMP/bin/hostname"

now=$(date +%s)
PATH="$TMP/bin:$PATH" "$HERE/launch-exact-run.sh" \
  "$RUN_LABEL" "$((now + 30))" "$((now + 900))" "$TMP/remote.log" \
  "$HERE/smoke-workload.sh" | tee "$RECEIPT"

grep -Fq 'launch_complete=true' "$RECEIPT"
python3 - "$TMP/remote.log" <<'PY'
import sys

records = [
    line for line in open(sys.argv[1], encoding="utf-8")
    if line.startswith("clearance=")
]
assert len(records) == 1, records
fields = dict(token.split("=", 1) for token in records[0].split())
assert fields["clearance"] == "true"
assert fields["process_group_scan"] == "pass"
assert fields["process_group"] == "empty"
assert fields["lease"] == "removed"
assert fields["exit_rc"] == "0"
PY
printf 'launcher_contract_smoke=pass\n' | tee -a "$RECEIPT"
cat > "$TMP/bin/timeout" <<'SH'
#!/usr/bin/env bash
for arg in "$@"; do
  if [ "$arg" = python3 ]; then
    echo independent_clearance_timeout_injected=true >&2
    exit 124
  fi
done
exec /usr/bin/timeout "$@"
SH
chmod 700 "$TMP/bin/timeout"
now=$(date +%s)
set +e
PATH="$TMP/bin:$PATH" "$HERE/launch-exact-run.sh" \
  "$DEADLINE_RUN_LABEL" "$((now + 30))" "$((now + 900))" \
  "$TMP/deadline-remote.log" "$HERE/smoke-workload.sh" \
  >"$TMP/deadline-launch.log" 2>&1
deadline_rc=$?
set -e
[[ "$deadline_rc" == 124 ]]
grep -Fq 'independent_clearance_timeout_injected=true' "$TMP/deadline-launch.log"
if grep -Fq 'launch_complete=true' "$TMP/deadline-launch.log"; then
  exit 1
fi
cat "$TMP/deadline-launch.log" >>"$RECEIPT"
printf 'independent_clearance_deadline_case_rc=%s launch_complete=false\n' \
  "$deadline_rc" | tee -a "$RECEIPT"
