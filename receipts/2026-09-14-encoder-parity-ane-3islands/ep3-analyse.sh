#!/bin/bash
# Off-device analysis: contract comparison for all three arms, section-42/43
# invariants for the two ANE arms, then the environment and golden anchors.
# No GPU, no ANE, no lock -- numpy and hashing only.
set -uo pipefail

ROOT=/var/tmp/EncoderParity3Islands
CAPTURE=/var/tmp/EncoderParityAne/capture
LOCK=$ROOT/package/coreml/parakeet-reference.lock
PY=$HOME/venv-agxgen/bin/python
rc=0

compare() {
  local label="$1" out="$2" title="$3"
  "$PY" "$ROOT/derivation/compare_golden.py" \
    --run "$ROOT/$out" --capture "$CAPTURE" --lock "$LOCK" \
    --label "$title" --out "$ROOT/compare-$label.json" || rc=$?
}

compare ane3   out-ane3   "ANE islands A+B+C on all 24 layers, Vulkan rest"
compare ane2   out-ane2   "ANE islands A+C on all 24 layers, Vulkan rest"
compare vulkan out-vulkan "vulkan-only control (no ANE)"

for label in ane3 ane2; do
  echo "--- invariants $label"
  "$PY" "$ROOT/derivation/check_invariants.py" \
    --run-report "$ROOT/run-report-$label.json" \
    --package "$ROOT/package" \
    --liveness "$ROOT/tools/ane_worker_liveness.py" \
    --out "$ROOT/invariants-$label.json" || rc=$?
done

echo "--- conv semantics"
"$PY" "$ROOT/derivation/check_conv_semantics.py" \
  --source /var/tmp/EncoderParityAne/encoder-source 2>&1 | tail -5 || true

echo "--- environment and anchors"
"$PY" "$ROOT/ep3-collect.py" > "$ROOT/collect.log" 2>&1 || rc=$?
"$PY" -c "
import json
a=json.load(open('$ROOT/golden-anchors.json'))
print('golden anchors all_match', a['all_match'])
"
exit $rc
