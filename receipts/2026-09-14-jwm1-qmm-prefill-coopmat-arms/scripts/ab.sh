#!/usr/bin/env bash
# ab.sh <rounds> <name=python> [<name=python> ...]
# Interleaved A/B of the kernel-isolated Q4 prefill probe: every round
# runs each arm once, in the order given, inside one GPU lock hold.
set -euo pipefail
ROUNDS=${1:?rounds}
shift
PROBE=/home/joshuawarren/benchq/qmmpad/probe.py
for r in $(seq 1 "$ROUNDS"); do
  for spec in "$@"; do
    name=${spec%%=*}
    py=${spec#*=}
    echo "== round $r arm $name"
    "$py" "$PROBE" | sed "s/^{/{\"round\": $r, \"name\": \"$name\", /"
  done
done
