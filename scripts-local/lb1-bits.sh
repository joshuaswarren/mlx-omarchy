#!/bin/bash
# Jw16LevelBatch W2 gate C: dg bitcheck rows identical, control vs candidate
# (flag on). Runs INSIDE a gpuwin window. Reuses the DecodeGap5 scripts.
set -u
OUT=/var/tmp/lb1/bits; mkdir -p "$OUT"
SERVE=/var/tmp/v072-venv-fused/bin/python3
CAND=/var/tmp/lb1-venv/bin/python3
env X=1 "$SERVE" /var/tmp/dg5/dg_bitcheck_conv.py "$OUT/bitconv-serve.json" > "$OUT/bitconv-serve.log" 2>&1
echo "conv-serve rc=$?"
env X=1 MLX_OMARCHY_LEVEL_BATCH=1 "$CAND" /var/tmp/dg5/dg_bitcheck_conv.py "$OUT/bitconv-cand.json" > "$OUT/bitconv-cand.log" 2>&1
echo "conv-cand rc=$?"
env X=1 "$SERVE" /var/tmp/dg/dg_bitcheck.py "$OUT/bit-serve.json" > "$OUT/bit-serve.log" 2>&1
echo "full-serve rc=$?"
env X=1 MLX_OMARCHY_LEVEL_BATCH=1 "$CAND" /var/tmp/dg/dg_bitcheck.py "$OUT/bit-cand.json" > "$OUT/bit-cand.log" 2>&1
echo "full-cand rc=$?"
python3 - "$OUT" <<'PYEOF'
import json, sys
out = sys.argv[1]
def rows(name):
    try:
        return json.load(open(f"{out}/{name}"))
    except Exception as e:
        print(f"LOAD-FAIL {name}: {e}")
        return {}
sv, cd = rows("bitconv-serve.json"), rows("bitconv-cand.json")
fs, fc = rows("bit-serve.json"), rows("bit-cand.json")
ref_keys = sorted(k for k in sv if "ref" in k)
bad = [k for k in ref_keys if sv.get(k) != cd.get(k)]
print("conv ref/refnn rows:", len(ref_keys), "mismatched:", bad)
full_keys = sorted(set(fs) | set(fc))
badf = [k for k in full_keys if fs.get(k) != fc.get(k)]
print("full bitcheck rows:", len(full_keys), "mismatched:", badf)
ok = bool(ref_keys) and bool(full_keys) and not bad and not badf
print("BITCHECK-GATE", "PASS" if ok else "FAIL")
PYEOF
echo BITCHECK-DONE
