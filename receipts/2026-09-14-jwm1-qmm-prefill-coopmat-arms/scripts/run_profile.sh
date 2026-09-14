#!/usr/bin/env bash
# run_profile.sh <venv-python> <outdir> <label>
# One locked 1053-token Q4 prefill profile run: host phase markers plus
# the GPU dispatch stream, then the kernel table. Same protocol and
# prompt as receipts/2026-09-13-jwm1-phase-profile-enabled.
set -euo pipefail
PY=${1:?python}
OUT=${2:?outdir}
LABEL=${3:?label}
SRC=/home/joshuawarren/src/mlx-bf16-grouped-base-b41e2b74
PROMPT=/tmp/jwm1-phase-profile-20260913/prompt.txt
MODEL=/home/joshuawarren/models/Qwen2.5-0.5B-Instruct-4bit-mlx
mkdir -p "$OUT"
rm -f "$OUT/profile.jsonl" "$OUT/markers.jsonl"
LOCK=/tmp/m1-gpu.lock
touch "$LOCK"
exec 9>"$LOCK"
if ! flock -w 60 9; then
  echo "GPU lock busy" >&2
  exit 1
fi
python3 - "$OUT" <<'PY'
import json, os, pathlib, subprocess, sys
st = os.stat("/tmp/m1-gpu.lock")
nested = subprocess.run(["flock", "-n", "/tmp/m1-gpu.lock", "-c", "true"]).returncode
pathlib.Path(sys.argv[1], "lock.json").write_text(json.dumps(
    {"inode": st.st_ino, "nested_flock_n_returncode": nested,
     "path": "/tmp/m1-gpu.lock"}, indent=2) + "\n")
print("lock_inode", st.st_ino, "nested", nested)
if nested == 0:
    raise SystemExit("nested flock unexpectedly succeeded")
PY
export MLX_DISABLE_COMPILE=1
export MLX_OMARCHY_GPU_PROFILE=$OUT/profile.jsonl
export MLX_OMARCHY_GPU_PROFILE_LABEL=$LABEL
export HF_HUB_OFFLINE=1
set +e
timeout -k 10s 600s "$PY" - "$SRC/scripts/profile_generate.py" "$MODEL" \
  "$PROMPT" "$OUT/markers.jsonl" "$OUT/run.log" <<'PY'
import pathlib, subprocess, sys
script, model, prompt_path, markers, log_path = sys.argv[1:]
prompt = pathlib.Path(prompt_path).read_text()
cmd = [sys.executable, script, "--model", model, "--prompt", prompt,
       "--max-tokens", "2", "--temp", "0", "--seed", "0", "--markers", markers]
with open(log_path, "w") as log:
    rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
raise SystemExit(rc)
PY
rc=$?
set -e
exec 9>&-
echo "generate rc=$rc"
fuser "$LOCK" >/dev/null 2>&1 && echo "lock after: held" || echo "lock after: free"
if [[ $rc -ne 0 ]]; then
  cat "$OUT/run.log" >&2
  exit $rc
fi
"$PY" "$SRC/scripts/profile_analyze.py" "$OUT/profile.jsonl" \
  --compute-h "$SRC/overlay/mlx/backend/omarchy/compute.h" \
  --markers "$OUT/markers.jsonl" > "$OUT/analysis.txt" 2>"$OUT/analysis.err"
python3 - "$OUT" <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
marks = [json.loads(l) for l in (out / "markers.jsonl").open()]
by = {}
for m in marks:
    by.setdefault(m.get("phase") or m.get("name"), []).append(m)
print("markers:", json.dumps(marks))
PY
grep -E "GPU busy fraction|QmmPrefillCoopmatF16|MatmulRbF16|prefill: dispatches" \
  "$OUT/analysis.txt"
