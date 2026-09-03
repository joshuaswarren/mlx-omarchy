#!/bin/sh
# Extract the tape-tests failure and run without override.
set -eu
grep -n -B8 "ERROR" "$HOME/benchq/logs/failclosed-tapetests.log" | head -30 || true
echo "---NO-OVERRIDE RUN---"
cd "$HOME/src/mlx-omarchy-bqm1-build"
if .work/mlx/build/tests/omarchy/omarchy_compiled_tape_tests \
    > "$HOME/benchq/logs/tapetests-nooverride.log" 2>&1; then
  echo "NOOVERRIDE_RC=0"
else
  RC=$?
  echo "NOOVERRIDE_RC=$RC"
fi
tail -6 "$HOME/benchq/logs/tapetests-nooverride.log"
