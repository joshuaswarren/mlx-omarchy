#!/bin/bash
# Parakeet wheel packaging gate (jwm1, aarch64). Proves the installed
# product runs the pinned public reference end to end from a CLEAN
# install: fresh prefix, clean venv, clean HOME, wheel-installed files
# only. Lock /tmp/m1-gpu.lock: flock -w 900, never steal, never unlink.
#
# Usage (on jwm1): gate-jwm1.sh WHEEL_PATH [OUT_DIR]
set -euo pipefail

WHEEL="${1:?usage: gate-jwm1.sh WHEEL_PATH [OUT_DIR]}"
OUT="${2:-$(cd "$(dirname "$0")" && pwd)/gate-run}"
WORK="$(mktemp -d /tmp/parakeet-gate.XXXXXX)"
LOCK=/tmp/m1-gpu.lock

# CLEAN_HOME may be pre-seeded (model cache) outside the lock; the gate
# still verifies everything inside.
CLEAN_HOME="${CLEAN_HOME:-$WORK/home}"
VENV="$WORK/venv"
mkdir -p "$CLEAN_HOME"
FAILURES=()

log() { printf '%s\n' "$*" | tee -a "$OUT/gate.log"; }
fail() { FAILURES+=("$1"); log "FAIL: $1"; }

cd "$(dirname "$0")/../.."  # repo root
mkdir -p "$OUT"
: > "$OUT/gate.log"
WHEEL="$(readlink -f "$WHEEL")"
log "gate start $(date -Iseconds) wheel=$WHEEL"
log "wheel sha256: $(sha256sum "$WHEEL" | cut -d' ' -f1)"

# --- clean install: fresh venv, clean HOME -----------------------------
env HOME="$CLEAN_HOME" python3 -m venv "$VENV"
# Host-provided transcribe dependencies (the wheel deliberately declares
# none; the CLI refuses naming them when absent).
env HOME="$CLEAN_HOME" "$VENV/bin/pip" install --quiet numpy protobuf soundfile
env HOME="$CLEAN_HOME" "$VENV/bin/pip" install --quiet "$WHEEL"

SITE="$("$VENV/bin/python" -c 'import mlx, pathlib; print(pathlib.Path(mlx.__path__[0]).resolve())')"
log "installed package dir: $SITE"
CLI="$SITE/bin/mlx-omarchy-parakeet"
WORKER="$SITE/bin/mlx-omarchy-ane-worker"
[[ -x "$CLI" ]] || { fail "mlx-omarchy-parakeet launcher not installed at $CLI"; }
[[ -x "$WORKER" ]] || { fail "mlx-omarchy-ane-worker not installed at $WORKER"; }

# --- installed surface -------------------------------------------------
for rel in bin/mlx-omarchy-parakeet bin/mlx-omarchy-ane-worker \
           coreml/parakeet-reference.lock \
           coreml/vulkan_encoder.py coreml/vulkan_mel.py \
           share/mlx-omarchy/parakeet-1/parakeet-runtime-pin.json \
           share/mlx-omarchy/parakeet-1/libane/libane-strict.so; do
  [[ -e "$SITE/$rel" ]] || fail "installed surface missing: $rel"
done
for bundle in island-attn-a-kt island-pv island-select-8head; do
  [[ -d "$SITE/share/mlx-omarchy/parakeet-1/bundles/$bundle" ]] \
    || fail "installed bundle missing: $bundle"
done

# --- download (clean HOME; the gate verifies, pre-seeded or not) --------
# The installed launcher runs under `env python3`; the venv owns the
# dependencies, so invoke it with the venv interpreter explicitly.
if ! flock -w 900 "$LOCK" env HOME="$CLEAN_HOME" \
    "$VENV/bin/python" "$CLI" download >> "$OUT/gate.log" 2>&1; then
  fail "download failed (see gate.log)"
fi

# --- transcribe 3x warm under the lock ---------------------------------
for run in 1 2 3; do
  if ! flock -w 900 "$LOCK" env HOME="$CLEAN_HOME" \
      "$VENV/bin/python" "$CLI" transcribe -o "$OUT/run-$run" \
      >> "$OUT/gate.log" 2>&1; then
    fail "transcribe run-$run failed (see gate.log)"
  fi
done

python3 - "$OUT" <<'PY' || fail "runs diverged or pins failed"
import hashlib, json, sys
from pathlib import Path

out = Path(sys.argv[1])
def text_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

summaries = []
for run in (1, 2, 3):
    report = json.loads(
        (out / f"run-{run}" / "transcribe-report.json").read_text())
    assert report["status"] == "match", (run, report["status"])
    assert not [c for c in report["verification"]["checks"] if not c["pass"]], run
    identity = (
        text_sha(out / f"run-{run}" / "transcript.txt"),
        text_sha(out / f"run-{run}" / "encoder_hidden.npy"),
        text_sha(out / f"run-{run}" / "token_ids.json"),
    )
    summaries.append(identity)
assert summaries[0] == summaries[1] == summaries[2], f"runs differ: {summaries}"
print("3x warm identical, status=match on every run")
PY

# --- prefix provenance: every path the runtime touched -----------------
python3 - "$OUT" "$SITE" "$CLEAN_HOME" <<'PY' || fail "runtime touched paths outside the install and cache"
import json, sys
from pathlib import Path
out, site, home = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
report = json.loads((out / "run-1" / "transcribe-report.json").read_text())
paths = [
    report["mlx"]["libmlx_path"], report["mlx"]["core_path"],
    report["inputs"]["audio"]["path"],
    report["inputs"]["encoder_source"]["root"],
    report["ane"]["worker"], report["ane"]["libane"],
]
bad = [p for p in paths if not (p.startswith(site) or p.startswith(home))]
sys.exit(f"paths outside install/cache: {bad}" if bad else 0)
PY

# --- no-ANE refusal (explicit, nonzero exit) ---------------------------
if env HOME="$CLEAN_HOME" MLX_OMARCHY_ANE_DEVICE=off \
    "$VENV/bin/python" "$CLI" transcribe -o "$OUT/refusal" > "$OUT/refusal.stdout" 2> "$OUT/refusal.stderr"; then
  fail "kill switch did not refuse (exit 0)"
fi
grep -q "MLX_OMARCHY_ANE_DEVICE=off" "$OUT/refusal.stderr" \
  || fail "kill-switch refusal did not name the switch"

# --- clean up the scratch prefix ---------------------------------------
rm -rf "$WORK"

log "gate end $(date -Iseconds)"
if [[ ${#FAILURES[@]} -eq 0 ]]; then
  log "GATE PASS: clean install runs the pinned reference end to end"
else
  log "GATE FAIL: ${FAILURES[*]}"
  exit 1
fi
