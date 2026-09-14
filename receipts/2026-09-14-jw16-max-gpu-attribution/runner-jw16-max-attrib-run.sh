#!/usr/bin/env bash
# jw16 M1 Max GPU attribution: phase-isolated 1053-token Q4 profile.
# Holds /tmp/m1-gpu.lock (flock -w 60, never steals, never unlinks).
# GPU only: no ANE, no reboot, no 1x896, no SET write, no driver change.
set -euo pipefail

OUT=/var/tmp/jw16-max-attrib-20260914
ROOT=/var/tmp/mlx-omarchy-prof-b41e2b74
PY="$ROOT/venv-diag/bin/python"
MODEL="$1"   # model path
mkdir -p "$OUT"

log() { printf '%s %s\n' "$(date -Is)" "$*"; }

log "== identity =="
vulkaninfo --summary 2>/dev/null | grep -E 'driverName|driverInfo|deviceName|apiVersion' | head -8 \
  | tee "$OUT/driver.txt"
"$PY" - <<'PY' | tee "$OUT/device.txt"
import json
import mlx.core as mx
print("mx", mx.__version__)
print("device", mx.default_device())
try:
    info = mx.device_info()
    print(json.dumps({k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                      for k, v in info.items()}, indent=1, sort_keys=True))
except Exception as exc:  # pragma: no cover
    print("device_info failed:", exc)
PY
"$PY" -c "import mlx_lm, mlx.core as mx, hashlib, pathlib
so = pathlib.Path(mx.__file__).parent
print('mlx_lm', mlx_lm.__version__)
print('mlx pkg', so)
for p in sorted(so.rglob('lib*.so*')) + sorted(so.glob('core*.so')):
    print(p.relative_to(so), hashlib.sha256(p.read_bytes()).hexdigest())
" | tee "$OUT/provenance.txt"
"$PY" -c "
import mlx.core as mx, pathlib, subprocess
p = pathlib.Path(mx.__file__).parent
for f in sorted(p.rglob('lib*.so*')) + sorted(p.glob('core*.so')):
    out = subprocess.run(['grep','-c','MLX_OMARCHY_GPU_PROFILE',str(f)],
                         capture_output=True, text=True)
    print(f.relative_to(p), 'MLX_OMARCHY_GPU_PROFILE literal count',
          out.stdout.strip() or out.returncode)
" | tee "$OUT/profiler-gate.txt"

PROMPT_FILE="$OUT/prompt.txt"
"$PY" - "$ROOT" "$PROMPT_FILE" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(root / "scripts"))
from bench_matrix import prompt_text
manifest = json.load(open(root / "scripts" / "bench_matrix.json"))
text = prompt_text(manifest, "ctx1024")
open(sys.argv[2], "w").write(text)
print("prompt utf8 bytes", len(text.encode()))
PY

log "== take lock =="
exec 9>/tmp/m1-gpu.lock
flock -w 60 9 || { log "lock busy, aborting"; exit 3; }
stat -c 'lock inode=%i' /tmp/m1-gpu.lock | tee "$OUT/lock.txt"
flock -n /tmp/m1-gpu.lock -c true && echo "NESTED-FLOCK-UNEXPECTEDLY-FREE" >>"$OUT/lock.txt" || echo "nested flock -n rc=1 (held)" >>"$OUT/lock.txt"

run_profile() {
  local tag="$1" maxtok="$2"
  log "== profile $tag (max_tokens=$maxtok) =="
  env -i \
    HOME="$HOME" PATH=/usr/bin:/bin \
    MLX_DISABLE_COMPILE=1 \
    HF_HUB_OFFLINE=1 \
    MESA_SHADER_CACHE_DISABLE=true \
    MLX_OMARCHY_GPU_PROFILE="$OUT/profile-$tag.jsonl" \
    MLX_OMARCHY_GPU_PROFILE_LABEL="jw16-max-attrib-$tag" \
    "$PY" "$ROOT/scripts/profile_generate.py" \
      --model "$MODEL" \
      --prompt "$(cat "$PROMPT_FILE")" \
      --max-tokens "$maxtok" --temp 0 --seed 0 \
      --markers "$OUT/markers-$tag.jsonl" 2>&1 | tee "$OUT/generate-$tag.log"
  log "rc=${PIPESTATUS[0]} profile bytes: $(stat -c %s "$OUT/profile-$tag.jsonl")"
}

run_profile m1053-t2 2
run_profile m1053-t32 32

log "== release =="
exec 9>&-
flock -n /tmp/m1-gpu.lock -c true && echo "lock free after run" | tee -a "$OUT/lock.txt"
ls -la "$OUT"
log "done"
