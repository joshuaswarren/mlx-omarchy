#!/usr/bin/env bash
# Stage the ICD-arm wrappers for the agx_insert_waits A/B on jwm1.
# Nothing here installs a driver or edits a shared ICD: each wrapper sets
# VK_DRIVER_FILES for its own python process only.
set -euo pipefail

D=/home/joshuawarren/benchq/waitspatch
BENCH=/home/joshuawarren/benchq/qmm-coop-bench
VENV=/var/tmp/mlx-omarchy-profile-enabled-b41e2b74/venv/bin/python

mkdir -p "$D"

test -x "$VENV" || { echo "missing venv python: $VENV"; exit 1; }
test -f "$BENCH/icd-waitbatch-base.json" || { echo "missing base ICD"; exit 1; }
test -f "$BENCH/icd-waitbatch-patched.json" || { echo "missing patched ICD"; exit 1; }

# Shared env for both arms. MESA_SHADER_CACHE_DISABLE is set on timing runs
# too, not just dumps: across two driver builds a stale cached binary is a
# live risk (QmmPrefillOpt, 2026-09-14).
make_arm() {
  local name=$1 icd=$2
  cat > "$D/py-$name" <<EOF
#!/usr/bin/env bash
exec env \\
  VK_DRIVER_FILES=$BENCH/icd-waitbatch-$icd.json \\
  VK_ICD_FILENAMES=$BENCH/icd-waitbatch-$icd.json \\
  MESA_SHADER_CACHE_DISABLE=true \\
  AGX_SIMDMAT=1 \\
  $VENV "\$@"
EOF
  chmod +x "$D/py-$name"
}

make_arm base base
make_arm patched patched

echo "== wrappers"
ls -la "$D"

echo "== driver identity per arm (proves the override binds)"
for a in base patched; do
  echo "-- $a"
  "$D/py-$a" -c '
import ctypes, os
print("VK_DRIVER_FILES =", os.environ.get("VK_DRIVER_FILES"))
' || true
done
