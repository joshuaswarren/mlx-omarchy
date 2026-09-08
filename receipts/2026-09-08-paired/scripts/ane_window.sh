#!/usr/bin/env bash
# ANE-only window: one flock on /tmp/m1-gpu.lock, never unlink.
set -uo pipefail
echo "hostname: $(hostname)"
date -u +"window-start %Y-%m-%dT%H:%M:%S.%6NZ"
exec 9>/tmp/m1-gpu.lock
if ! flock -x -w 60 9; then
  echo "ABORT: could not acquire /tmp/m1-gpu.lock within 60s"
  exit 2
fi
echo "lock acquired (fd 9, tree $$)"
bash /tmp/ane_timed.sh
RC=$?
echo "ane rc=$RC"
date -u +"window-end %Y-%m-%dT%H:%M:%S.%6NZ"
flock -u 9
echo "lock released"
exit $RC
