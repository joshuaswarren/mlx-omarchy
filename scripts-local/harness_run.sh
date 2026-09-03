#!/bin/sh
# Differential harness on jwm1 hardware (TapeCorruptionBisect protocol).
set -eu
cd ~/src/mlx-omarchy-bqm1-build
VENV="$HOME/src/mlx-omarchy-bqm1-build/.work/venv-run"
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv --system-site-packages "$VENV"
  "$VENV/bin/pip" install -q mlx-lm==0.31.3
  "$VENV/bin/pip" install -q --no-deps --force-reinstall dist/mlx_omarchy-*.whl
fi
echo "== provenance gate =="
WHEELVER=$("$VENV/bin/pip" show mlx-omarchy | grep ^Version | cut -d" " -f2)
echo "installed mlx-omarchy: $WHEELVER"
echo "upstream mlx present: $("$VENV/bin/pip" show mlx >/dev/null 2>&1 && echo YES || echo no)"
LIB="$VENV/lib/python3.14/site-packages/mlx/lib/libmlx.so"
INSTALLED=$(sha256sum "$LIB" | cut -d" " -f1)
MEMBER=$(unzip -p dist/mlx_omarchy-*.whl mlx/lib/libmlx.so | sha256sum | cut -d" " -f1)
echo "installed libmlx.so: $INSTALLED"
echo "wheel member libmlx.so: $MEMBER"
[ "$INSTALLED" = "$MEMBER" ] || { echo "FATAL: installed payload is not the wheel under test" >&2; exit 8; }
"$VENV/bin/python" -c "import mlx.core as mx; print('mx.__version__', mx.__version__)"
unset MLX_DISABLE_COMPILE
unset MLX_OMARCHY_ALLOW_NON_APPLE
MODEL="$HOME/models/Qwen2.5-0.5B-Instruct-4bit-mlx"
echo "== step 1: mechanism probe =="
RC=0
"$VENV/bin/python" scripts/probe_tape_eager.py > "$HOME/benchq/logs/harness-probe.log" 2>&1 || RC=$?
echo "probe rc=$RC"
tail -8 "$HOME/benchq/logs/harness-probe.log"
if [ "$RC" -ne 0 ] && [ "$RC" -ne 3 ]; then exit "$RC"; fi
echo "== step 2: realpath reproduction =="
RC2=0
"$VENV/bin/python" scripts/differential_compile.py --mode realpath \
  --model "$MODEL" --prompt "What is the capital of France?" --steps 32 \
  --out-dir "$HOME/benchq/diff-m1" > "$HOME/benchq/logs/harness-realpath.log" 2>&1 || RC2=$?
echo "realpath rc=$RC2"
tail -25 "$HOME/benchq/logs/harness-realpath.log"
if [ "$RC2" -eq 3 ]; then
  echo "== step 3: localization + shrink =="
  "$VENV/bin/python" scripts/differential_compile.py --mode model \
    --model "$MODEL" --prompt "What is the capital of France?" --steps 4 \
    --shrink --shrink-dir "$HOME/benchq/repro_m1" \
    > "$HOME/benchq/logs/harness-model.log" 2>&1 || RC3=$?
  echo "model rc=${RC3:-0}"
  tail -30 "$HOME/benchq/logs/harness-model.log"
fi
