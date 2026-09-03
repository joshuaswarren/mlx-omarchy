#!/bin/sh
# Kick the analyzer detached; returns immediately.
rm_marker() { :; }
if [ -f "$HOME/benchq/analyzer-DONE" ]; then
  rm -f "$HOME/benchq/analyzer-DONE"
fi
nohup sh "$HOME/benchq/analyzer_run.sh" >/dev/null 2>&1 &
echo "KICKED pid=$!"
