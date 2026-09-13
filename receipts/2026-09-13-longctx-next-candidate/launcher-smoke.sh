#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
TMP=$(mktemp -d /var/tmp/longctx-launcher-smoke.XXXXXX)
RUN_LABEL="launcher-local-$$"
REMOTE_ROOT="/var/tmp/LongContextCostAttribution-$RUN_LABEL"
RECEIPT="$HERE/launcher-smoke-receipt.txt"
cleanup() {
  local rc=$?
  trap - EXIT
  [[ "$TMP" == /var/tmp/longctx-launcher-smoke.* ]]
  [[ "$REMOTE_ROOT" == /var/tmp/LongContextCostAttribution-launcher-local-* ]]
  rm -r -- "$TMP"
  if [ -e "$REMOTE_ROOT" ]; then
    rm -r -- "$REMOTE_ROOT"
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
