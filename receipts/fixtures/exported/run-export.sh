#!/bin/sh
set -u
cd "$HOME/src/ane-export-20260901" || exit 1
TOOLS="$HOME/src/mil-oneop-proof-20260831"
COMMIT=c17bb1b82490b661ef6a3acd1fa008265383802b
rm -f export.log
for d in desc-add-1x512 desc-add-1x896 desc-mul-1x512 desc-matmul-1x16-x-16x32; do
  "$TOOLS/.venv7/bin/python" ane_export.py "$d.json" --out-dir "out-$d" \
    --tools-dir "$TOOLS" --target h13 --source-commit "$COMMIT" || echo "FAILED $d"
done
echo "=== anec inventory ==="
ls -l out-*/bundle/ 2>/dev/null
