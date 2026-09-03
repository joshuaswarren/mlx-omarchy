#!/bin/sh
# Show only non-pycache differences between the three installed mlx trees.
set -eu
for PAIR in "0606 0928" "0606 0512" "0928 0512"; do
  set -- $PAIR
  A=$1; B=$2
  echo "=== $A vs $B: total changed lines ==="
  diff ~/benchq/pkg-$A.sha ~/benchq/pkg-$B.sha | grep -c "^[<>]" || true
  echo "=== $A vs $B: changed lines NOT in __pycache__ ==="
  diff ~/benchq/pkg-$A.sha ~/benchq/pkg-$B.sha | grep "^[<>]" | grep -v "__pycache__" || true
done
