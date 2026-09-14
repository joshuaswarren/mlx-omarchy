#!/usr/bin/env bash
# Kernel-isolated probe + end-to-end digest legs for the fat-shape A/B.
# Usage: measure-fat.sh <root> <model-path> <tree> <venv-src-or-empty>
# base arm  = MLX_OMARCHY_QMM_COOPMAT_N16_WG_PER_CORE=0 (shipped pick)
# n16 arm   = compiled-in default threshold
set -euo pipefail
ROOT=${1:?root}; MODEL=${2:?model}; TREE=${3:?tree}; VENV_SRC=${4:-}
WHEEL=$(echo "$ROOT"/dist/mlx_omarchy-*.whl)
VENV="$ROOT/venv-run"

if [[ ! -x "$VENV/bin/python" ]]; then
  if [[ -n "$VENV_SRC" ]]; then cp -a "$VENV_SRC" "$VENV"; fi
fi
if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv --system-site-packages "$VENV"
fi
py="$VENV/bin/python"
"$py" -m pip install -q --no-deps --no-index --force-reinstall "$WHEEL"
"$py" - <<'EOF'
import mlx.core as mx
print("venv ok", mx.__version__, mx.device_info())
EOF

# pinned prompts, expanded from the repo manifest
"$py" - "$TREE" <<'EOF'
import json, sys
sys.path.insert(0, sys.argv[1] + "/scripts")
from bench_matrix import prompt_text
m = json.load(open(sys.argv[1] + "/scripts/bench_matrix.json"))
for pid in ("short", "ctx1024"):
    text = prompt_text(m, pid)
    open(f"/var/tmp/prompts-{pid}.txt", "w").write(text)
    print(pid, "bytes", len(text))
# mid-length leg: the ctx1024 numbered template cut to 5 entries,
# ~262 chat-template tokens - the m where the 16-column twin measured
# its wins
e = dict(m["prompts"]["ctx1024"])
e["items"] = 5
text = prompt_text({"prompts": {"ctx262": e}}, "ctx262")
open("/var/tmp/prompts-ctx262.txt", "w").write(text)
print("ctx262 bytes", len(text))
EOF
run_leg() { # mode leg promptfile outfile extra-env...
  local mode=$1 leg=$2 pf=$3 out=$4; shift 4
  local rc=0
  env MLX_DISABLE_COMPILE=1 HF_HUB_OFFLINE=1 "$@" \
    flock -w 60 /tmp/m1-gpu.lock "$py" "$TREE/scripts/bench_decode.py" \
      --model "$MODEL" --prompt "$(cat "$pf")" --tokens 32 \
      --temp 0 --seed 0 --warmup-tokens 4 --wheel "$WHEEL" \
      > "$out" 2>&1 || rc=$?
  echo "[$mode/$leg] rc=$rc -> $out"
  tail -2 "$out"
}


run_probe() { # mode outfile extra-env...
  local mode=$1 out=$2; shift 2
  env MLX_DISABLE_COMPILE=1 HF_HUB_OFFLINE=1 "$@" \
    flock -w 60 /tmp/m1-gpu.lock "$py" "$ROOT/probe.py" > "$out" 2>&1
  echo "[$mode/probe] -> $out"
}

echo "== probe base =="
run_probe base "$ROOT/probe-base.jsonl" MLX_OMARCHY_QMM_COOPMAT_N16_WG_PER_CORE=0
echo "== probe n16 =="
run_probe n16 "$ROOT/probe-n16.jsonl"

for leg in short ctx262 ctx1024; do
  pf=/var/tmp/prompts-$leg.txt
  echo "== e2e base $leg =="
  run_leg base "$leg" "$pf" "$ROOT/e2e-base-$leg.log" \
    MLX_OMARCHY_QMM_COOPMAT_N16_WG_PER_CORE=0
  echo "== e2e n16 $leg =="
  run_leg n16 "$leg" "$pf" "$ROOT/e2e-n16-$leg.log"
done
echo "== done =="
