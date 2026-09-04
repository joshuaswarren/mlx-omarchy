#!/usr/bin/env bash
# SDPA f16-scores A/B runner. Usage: sdpa-ab.sh <before-sha> <after-sha>
#
# Two modes:
#   distinct-wheels   BEFORE != AFTER: builds both SHAs from detached
#                     checkouts, asserts the two wheels DIFFER, runs the
#                     SDPA equivalence suite on the after side, then five
#                     pinned decode runs per dtype per side in one session.
#   same-wheel-gates  BEFORE == AFTER: builds ONE wheel, installs it in ONE
#                     venv, and A/Bs ONE named optimization gate on that
#                     identical binary. GATE_ENV must name the gate
#                     (MLX_OMARCHY_QMM_TILE, MLX_OMARCHY_ROPE_BF16_DIRECT,
#                     MLX_OMARCHY_SDPA_BF16_FAST, or a future fusion gate):
#                     side 'off' runs with GATE_ENV=0, side 'on' with
#                     GATE_ENV=1. No default gate is guessed; same-SHA
#                     without GATE_ENV is refused. MLX_DISABLE_COMPILE
#                     NEVER varies between sides in this mode (bf16 stays
#                     =1, q4 stays unset - the existing baseline), so the
#                     measured delta is the gate, not compile behavior.
#                     Both sides print the same wheel hash by construction;
#                     differences are GATE effects and must never be
#                     reported as binary differences. There is no
#                     equivalence suite in this mode (the binary is
#                     identical by definition).
#
# Prints its PID first. Hardened 2026-09-04 (BenchmarkIntegrity):
# every path this script writes is unique per run (date+PID) - no
# rm -rf of shared trees, no venv collisions between reruns. Every
# command's exit status is checked EXPLICITLY; errexit is NOT relied
# on, because functions called in a conditional context
# (`build ... || exit`) run with it disabled. No pipeline can mask a
# failure: the wheel glob uses find -print -quit (no head SIGPIPE) and
# its emptiness is a loud error, and each bench run's python status is
# captured and fatal. One residual shared path: $SRC/dist is emptied by
# build-wheel.sh, so two concurrent A/Bs on one host still race there -
# the protocol is one session per host; if that changes, lock
# $SRC/dist.
set -uo pipefail
echo "PID $$ host $(hostname) start $(date +%T)"
[ $# -eq 2 ] || { echo "usage: $0 <before-sha> <after-sha>" >&2; exit 2; }
BEFORE=$1; AFTER=$2
if [ "$BEFORE" = "$AFTER" ]; then
  MODE=same-wheel-gates
else
  MODE=distinct-wheels
fi
echo "mode: $MODE"
if [ "$MODE" = same-wheel-gates ]; then
  # A same-wheel A/B measures ONE named optimization gate. No guessing:
  # without an explicit GATE_ENV there is nothing to toggle and the run
  # would silently measure noise, so it is refused.
  if [ -z "${GATE_ENV:-}" ]; then
    echo "same-wheel mode requires GATE_ENV=<gate name>, e.g." >&2
    echo "  GATE_ENV=MLX_OMARCHY_QMM_TILE (or _ROPE_BF16_DIRECT," >&2
    echo "  _SDPA_BF16_FAST, or the future fusion gate)." >&2
    echo "Refusing to run without a named gate." >&2
    exit 2
  fi
  case "$GATE_ENV" in
    MLX_OMARCHY_[A-Z0-9_]*) ;;
    *) echo "GATE_ENV must be an MLX_OMARCHY_* gate name, got '$GATE_ENV'" >&2
       exit 2;;
  esac
fi
RUN_ID=$(date +%Y%m%d-%H%M%S)-$$
LOG=~/benchq/logs/${AB_NAME:-sdpa-ab}-$RUN_ID
mkdir -p "$LOG" || { echo "cannot create $LOG" >&2; exit 1; }
SRC=${SRC:-$HOME/src/mlx-omarchy-bqm1-build}
TEMPLATE=${TEMPLATE:-$HOME/venv-bqm1-arm-v033}
PROMPT="What is the capital of France? Answer in one word."
Q4=~/models/Qwen2.5-0.5B-Instruct-4bit-mlx
BF=~/models/Qwen2.5-0.5B-Instruct-bf16-mlx
[ -d "$TEMPLATE" ] || { echo "template venv $TEMPLATE missing" >&2; exit 2; }
ulimit -c 0

build() { # sha -> preserved wheel filename on stdout (run in a subshell
          # so the caller's cwd never moves)
  local sha=$1
  (
    cd "$SRC" || exit 1
    git fetch -q origin || { echo "git fetch failed" >&2; exit 1; }
    git checkout -q --detach "$sha" || { echo "checkout $sha failed" >&2; exit 1; }
    # per-run build workspace: build-wheel.sh honors MLX_OMARCHY_WORK_DIR
    local work="$SRC/.work-$RUN_ID/$sha"
    mkdir -p "$work" || { echo "mkdir $work failed" >&2; exit 1; }
    if ! MLX_OMARCHY_WORK_DIR="$work" DEV_RELEASE=1 \
        ./scripts/build-wheel.sh > "$LOG/build-$sha.log" 2>&1; then
      echo "build $sha failed (log: $LOG/build-$sha.log)" >&2
      tail -5 "$LOG/build-$sha.log" >&2
      exit 1
    fi
    local w
    w=$(find dist -name "*+${sha:0:7}*aarch64.whl" -print -quit)
    if [ -z "$w" ]; then
      echo "no wheel matching +${sha:0:7} in $SRC/dist after build" >&2
      ls dist >&2
      exit 1
    fi
    cp -- "$w" "$LOG/" || { echo "cp $w failed" >&2; exit 1; }
    basename "$w"
  )
}

install_wheel() { # wheel-path side -> venv path on stdout
  local w=$1 side=$2
  local v=~/venv-${AB_NAME:-sdpa-ab}-$RUN_ID-$side
  rm -rf "$v" || { echo "rm -rf $v failed" >&2; exit 1; }
  cp -a "$TEMPLATE" "$v" || { echo "venv copy for $side failed" >&2; exit 1; }
  if ! "$v/bin/python" -m pip install -q --force-reinstall --no-deps "$w"; then
    echo "install $side failed: $w" >&2
    exit 1
  fi
  echo "$v"
}

if [ "$MODE" = distinct-wheels ]; then
  WB="$LOG/$(build "$BEFORE")" || { echo "BEFORE build failed" >&2; exit 1; }
  WA="$LOG/$(build "$AFTER")" || { echo "AFTER build failed" >&2; exit 1; }
  echo "before: $WB $(sha256sum "$WB" | cut -c1-16)"
  echo "after:  $WA $(sha256sum "$WA" | cut -c1-16)"
  [ "$(sha256sum "$WB" | cut -d' ' -f1)" != "$(sha256sum "$WA" | cut -d' ' -f1)" ] \
    || { echo "SIDES IDENTICAL - refusing to measure a distinct-wheel A/B"; exit 1; }
  VB=$(install_wheel "$WB" before) || exit 1
  VA=$(install_wheel "$WA" after) || exit 1
  SIDES="before after"
else
  W1="$LOG/$(build "$BEFORE")" || { echo "build failed" >&2; exit 1; }
  echo "wheel: $W1 $(sha256sum "$W1" | cut -c1-16)"
  echo "same-wheel mode: both sides run THIS IDENTICAL WHEEL in ONE venv."
  echo "  gate under test: $GATE_ENV   side 'off' = $GATE_ENV=0, side 'on' = $GATE_ENV=1"
  echo "  MLX_DISABLE_COMPILE does NOT vary between sides: =1 for bf16"
  echo "  (compiled bf16 tape is gated by design), unset for q4 - on"
  echo "  BOTH sides, so no unrelated compile behavior is measured."
  echo "  Any off/on delta below is a $GATE_ENV effect, not a binary"
  echo "  difference, and must be reported as such."
  VG=$(install_wheel "$W1" gate) || exit 1
  echo "[$(date +%T)] installed: $("$VG/bin/python" -c 'import mlx.core as mx; print(mx.__version__)')"
  SIDES="off on"
fi

# Run manifest: the persisted identity of this session, so a log dir
# can never be misread later.
{
  echo "mode=$MODE"
  echo "before_sha=$BEFORE"
  echo "after_sha=$AFTER"
  if [ "$MODE" = same-wheel-gates ]; then
    echo "gate_env=$GATE_ENV"
    echo "gate_values=off=0 on=1"
  fi
  echo "compile_policy=q4: MLX_DISABLE_COMPILE unset on BOTH sides; bf16: MLX_DISABLE_COMPILE=1 on BOTH sides"
  echo "prompt=$PROMPT"
  echo "tokens=64 runs=5"
} > "$LOG/config.txt"

if [ "$MODE" = distinct-wheels ]; then
  # EQ_CMD runs at the after checkout with the after venv first on PATH; non-zero exit = no numbers.
  EQ_CMD=${EQ_CMD:-"python scripts/sdpa_equivalence.py"}
  echo "[$(date +%T)] equivalence on after side (checkout is at $AFTER): $EQ_CMD"
  ( cd "$SRC" && PATH="$VA/bin:$PATH" bash -c "$EQ_CMD" ) \
    > "$LOG/equivalence.log" 2>&1
  EQ=$?
  tail -3 "$LOG/equivalence.log"
  echo "equivalence exit=$EQ"
  [ $EQ -eq 0 ] || { echo "EQUIVALENCE FAILED - no numbers"; exit 1; }
fi

for side in $SIDES; do
  if [ "$MODE" = distinct-wheels ]; then
    [ $side = before ] && V=$VB || V=$VA
    [ $side = before ] && W=$WB || W=$WA
  else
    V=$VG
    W=$W1
  fi
  for dtype in q4 bf16; do
    [ $dtype = q4 ] && M=$Q4 || M=$BF
    # The compile setting NEVER varies between sides. Distinct mode:
    # the protocol fixes bf16 eager (compiled bf16 tape is gated by
    # design) and q4 default. Same-wheel mode: the same per-dtype
    # setting holds on both sides; only $GATE_ENV toggles (off=0, on=1).
    if [ "$MODE" = distinct-wheels ]; then
      [ $dtype = bf16 ] && export MLX_DISABLE_COMPILE=1 || unset MLX_DISABLE_COMPILE
      gatelabel=$([ $dtype = bf16 ] && echo "compile=off" || echo "compile=on")
    else
      [ $dtype = bf16 ] && export MLX_DISABLE_COMPILE=1 || unset MLX_DISABLE_COMPILE
      if [ $side = on ]; then
        export "$GATE_ENV=1"
        gatelabel="$GATE_ENV=1"
      else
        export "$GATE_ENV=0"
        gatelabel="$GATE_ENV=0"
      fi
    fi
    for run in 1 2 3 4 5; do
      OUT="$LOG/$side-$dtype-run$run.log"
      "$V/bin/python" "$SRC/scripts/bench_decode.py" --model "$M" --prompt "$PROMPT" \
        --tokens 64 --wheel "$W" > "$OUT" 2>&1
      rc=$?
      if [ $rc -ne 0 ]; then
        echo "BENCH FAILED [$side $dtype $gatelabel run$run] exit=$rc (log: $OUT)" >&2
        tail -5 "$OUT" >&2
        exit 1
      fi
      lines=$(grep -hE "^decode [0-9]|^provenance|^generated_ids" "$OUT" \
        | sed "s/^/[$side $dtype $gatelabel r$run] /")
      if [ -z "$lines" ]; then
        echo "NO RATE LINE [$side $dtype $gatelabel run$run]: python exited 0 but printed no decode/provenance line (log: $OUT)" >&2
        exit 1
      fi
      printf '%s\n' "$lines"
    done
  done
done
echo "[$(date +%T)] DONE mode=$MODE"
