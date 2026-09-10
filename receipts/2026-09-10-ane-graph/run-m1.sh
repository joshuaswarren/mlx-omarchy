#!/usr/bin/env bash
set -euo pipefail

root="$HOME/src/mlx-AneGraphIntegration"
receipt="$root/receipts/2026-09-10-ane-graph"
wheel="$root/dist/mlx_omarchy-0.32.2.dev202609101742+786553-cp314-cp314-linux_aarch64.whl"
venv="$root/.work/venv-ane"
compiler="$root/.work/mil-hwx-compiler"
driver="$HOME/src/ane-eightcore-20260906/runtime-lifecycle5/ane/ane.ko"
libane="$HOME/src/ane-eightcore-20260906/runtime-abi1/bindings/python/dylib/libane_python.so"
source_commit="078655346699154df3948d97d4624113f8fac78a"
wheel_sha256="9d620b2ed61a51d396fda5d94dcaca8fa08f950df0be56a66ca1c2aa72475e1f"
compiler_sha256="ceaf24c3bf38f7375b6c2e4caa7e978b0eb5c6d5459aaafbbb758eefdeb0b1ac"
loaded_here=0
started=$(date +%s)

mkdir -p "$receipt"
exec > >(tee "$receipt/m1-run.log") 2>&1

state() {
  local phase=$1
  {
    printf 'phase=%s\n' "$phase"
    printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
    printf 'hostname=%s\n' "$(hostname)"
    printf 'kernel=%s\n' "$(uname -r)"
    printf 'machine=%s\n' "$(uname -m)"
    printf 'boot_id=%s\n' "$(cat /proc/sys/kernel/random/boot_id)"
    printf 'module_present=%s\n' "$([[ -e /sys/module/ane ]] && echo true || echo false)"
    printf 'device_present=%s\n' "$([[ -e /dev/accel/accel0 ]] && echo true || echo false)"
    if [[ -e /sys/module/ane/version ]]; then
      printf 'module_version=%s\n' "$(cat /sys/module/ane/version)"
    fi
    printf 'driver_sha256=%s\n' "$(sha256sum "$driver" | cut -d' ' -f1)"
    printf 'libane_sha256=%s\n' "$(sha256sum "$libane" | cut -d' ' -f1)"
    printf 'compiler_commit=%s\n' "$(git -C "$compiler" rev-parse HEAD)"
    printf 'compiler_binary_sha256=%s\n' "$compiler_sha256"
    printf 'wheel_sha256=%s\n' "$wheel_sha256"
    while IFS= read -r -d '' compatible; do
      printf 'firmware_compatible[%s]=%s\n' "$compatible" "$(tr '\0' ',' < "$compatible")"
    done < <(find /sys/firmware/devicetree/base -type f -name compatible -path '*ane*' -print0 2>/dev/null)
  } | tee "$receipt/m1-state-$phase.txt"
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM HUP
  if (( loaded_here )); then
    if ! sudo rmmod ane; then
      status=70
      echo 'ERROR: failed to unload ane module' >&2
    fi
  fi
  state post || status=71
  if [[ -e /sys/module/ane || -e /dev/accel/accel0 ]]; then
    status=72
    echo 'ERROR: ANE state was not restored' >&2
  fi
  printf 'lock_hold_seconds=%s\n' "$(( $(date +%s) - started ))"
  exit "$status"
}
trap cleanup EXIT
trap 'exit 143' INT TERM HUP

cd "$root"
exec 9>/tmp/m1-gpu.lock
flock -x -w 2400 9
started=$(date +%s)
state pre
if [[ -e /sys/module/ane || -e /dev/accel/accel0 ]]; then
  echo 'ERROR: ANE was already active; refusing to alter another worker state' >&2
  exit 73
fi

sudo insmod "$driver"
loaded_here=1
[[ -e /sys/module/ane && -e /dev/accel/accel0 ]]
state loaded

printf 'mlx_provenance_command=%q %q %q %q\n' "$venv/bin/python" scripts/mlx_provenance.py --expect-wheel "$wheel"
timeout 60 "$venv/bin/python" scripts/mlx_provenance.py --expect-wheel "$wheel" | tee "$receipt/mlx-provenance.json"

if [[ ${1:-} != large-only ]]; then
  printf 'qualifier_command=MLX_OMARCHY_ANE_REGION=1 %q scripts/qualify_ane_graph.py [arguments recorded in verdict.json]\n' "$venv/bin/python"
  MLX_OMARCHY_ANE_REGION=1 timeout 300 "$venv/bin/python" scripts/qualify_ane_graph.py \
    --package "$receipt/package" \
    --compiler-root "$compiler" \
    --libane-library "$libane" \
    --driver-module "$driver" \
    --compiler-binary-sha256 "$compiler_sha256" \
    --wheel-sha256 "$wheel_sha256" \
    --source-commit "$source_commit" \
    --output "$receipt/verdict.json" \
    --warmup 5 \
    --iterations 50 \
    --timeout-seconds 120
fi

printf 'large_qualifier_command=MLX_OMARCHY_ANE_REGION=1 %q %q [arguments recorded in large-region.json]\n' "$venv/bin/python" "$receipt/qualify-large.py"
MLX_OMARCHY_ANE_REGION=1 timeout 900 "$venv/bin/python" "$receipt/qualify-large.py" \
  --compiler-root "$compiler" \
  --package "$receipt/package-b1-current" \
  --libane "$libane" \
  --input "$receipt/b1-x.fp16" \
  --weights "$receipt/weights.bin" \
  --expected "$receipt/b1-y.fp16" \
  --output "$receipt/large-region.json" \
  --warmup 3 \
  --iterations 20 \
  --timeout 120
