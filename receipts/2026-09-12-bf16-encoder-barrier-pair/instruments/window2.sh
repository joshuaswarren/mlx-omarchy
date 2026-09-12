#!/usr/bin/env bash
# Deferred-post-barrier A/B window 2 (wave/Bf16EncoderBoundary, cand >=
# ABI rework commit). Replaces the invalid window 1 (fork matrix runs
# died on an empty env-array expansion; probes crashed on mlx namespace
# package; checker accepted a vacuous leg set). Window 1 artifacts stay
# in ~/enc-ab-window untouched as build evidence only.
# ONE top-level flock on /tmp/m1-gpu.lock covering the candidate source
# build AND every GPU run. The base wheel (pristine a12eafd) is reused
# from window 1; the candidate wheel is rebuilt from the re-staged tree.
# Protocol: identity, cand build + venv reinstall, quiet gate (three
# checks >= 20 s apart at load < 1.0), engagement probes, discarded
# warmup per wheel, A/A baseline pair, interleaved base/candidate 3x3
# (counterbalanced), one stock-driver run per wheel, digest assertions
# against committed pins on every measured leg of every run, paired
# medians. The checker requires the full six-leg measured set per run;
# a missing leg is a failure, never a vacuous pass.
set -uo pipefail
BASE_COMMIT=a12eafd994af881dbc2b39c0b9807523f27b171a
CAND_COMMIT=$(cat "$HOME/src/mlx-enc-cand/.cand_commit" 2>/dev/null || echo unknown)
B="$HOME/src/mlx-enc-base"
C="$HOME/src/mlx-enc-cand"
R="$HOME/enc-ab-window2"
STOCK_ICD="$HOME/stock-mesa/stock-icd.json"
PARITY_WORK="$HOME/src/mlx-omarchy-parity/.work"

exec 9>/tmp/m1-gpu.lock
echo "$(date -u +%FT%TZ) waiting for m1-gpu.lock pid $$"
flock -w 3600 9 || { echo "FATAL: lock wait exceeded"; exit 4; }
echo "$(date -u +%FT%TZ) LOCK ACQUIRED"
T0=$(date +%s)
mkdir -p "$R"

{
  echo "host=$(hostname) kernel=$(uname -r) nproc=$(nproc)"
  pacman -Q mesa-honeykrisp-omarchy 2>/dev/null
  echo "base_commit=$BASE_COMMIT (wheel reused from window 1)"
  echo "cand_commit=$CAND_COMMIT"
  echo "base_wheel_sha256=$(sha256sum "$B"/dist/mlx_omarchy-*.whl | cut -d' ' -f1)"
} > "$R/drivers.txt"
cat "$R/drivers.txt"

echo "=== build cand $(date -u +%H:%M:%S) ==="
(cd "$C" \
  && MLX_OMARCHY_SOURCE_COMMIT="$CAND_COMMIT" DEV_RELEASE=1 \
     CMAKE_BUILD_PARALLEL_LEVEL="$(nproc)" \
     ./scripts/build-wheel.sh) > "$R/build-cand.log" 2>&1
rc=$?
echo "build cand rc=$rc $(date -u +%H:%M:%S)"
[ $rc -ne 0 ] && { echo "FATAL: build cand failed"; exit 5; }
sha256sum "$C"/dist/mlx_omarchy-*.whl >> "$R/drivers.txt"

rm -rf "$C/.work/venv-cand"
cp -a "$PARITY_WORK/venv-run" "$C/.work/venv-cand"
"$C/.work/venv-cand/bin/python" -m pip install --quiet \
  --force-reinstall --no-deps --no-index \
  "$C"/dist/mlx_omarchy-*.whl > "$R/venv-cand.log" 2>&1
rc=$?
echo "venv cand rc=$rc"
[ $rc -ne 0 ] && { echo "FATAL: venv cand install failed"; exit 6; }

BASE_WHEEL=$(ls "$B"/dist/mlx_omarchy-*.whl)
CAND_WHEEL=$(ls "$C"/dist/mlx_omarchy-*.whl)
BASE_PY="$B/.work/venv-base/bin/python"
CAND_PY="$C/.work/venv-cand/bin/python"

# quiet gate before any measurement
ok=0
for i in $(seq 1 270); do
  load=$(cut -d" " -f1 /proc/loadavg)
  pass=$(awk -v l="$load" 'BEGIN{print (l<1.0)?1:0}')
  if [ "$pass" = 1 ]; then
    ok=$((ok+1))
    [ $ok -ge 3 ] && break
    sleep 20
  else
    ok=0
    sleep 20
  fi
done
echo "quiet gate done ok=$ok load=$(cut -d' ' -f1 /proc/loadavg)"
[ $ok -ge 3 ] || { echo "FATAL: quiet gate never passed"; exit 7; }

( while :; do echo "$(date +%s) $(cut -d' ' -f1-3 /proc/loadavg)"; sleep 10; done ) > "$R/loadavg.txt" 2>&1 &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

matrix_run() {  # tree py wheel label out icd
  local tree="$1" py="$2" wheel="$3" label="$4" out="$5" icd="$6"
  local -a ICD=(env)
  [ "$icd" = stock ] && ICD+=("VK_DRIVER_FILES=$STOCK_ICD")
  echo "=== matrix $out $(date -u +%H:%M:%S) ==="
  (
    cd "$tree" || exit 8
    timeout 1500 "${ICD[@]}" HF_HUB_OFFLINE=1 MLX_DISABLE_COMPILE=1 \
      "$py" scripts/bench_matrix.py --mode run \
      --python "$py" \
      --wheel "$wheel" \
      --expect-pins qwen25-0.5b-4bit=a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3 \
      --expect-pins qwen25-0.5b-bf16=56d07e766edd7159fbe12ed12d9cf114bf38bf1e \
      --host-label "jwm1 $label" \
      --timeout 900 \
      --out "$R/$out.json" > "$R/$out.log" 2>&1
    echo "$out exit=$? $(date -u +%H:%M:%S)"
  )
}

probe_run() {  # py label
  local py="$1" label="$2"
  timeout 300 "$py" "$R/probe.py" > "$R/probe-$label.txt" 2>&1
  echo "probe $label exit=$?"
  grep '^PROBE' "$R/probe-$label.txt" || tail -3 "$R/probe-$label.txt"
}

probe_run "$BASE_PY" base
probe_run "$CAND_PY" cand

# discarded warmups, one full matrix per wheel
matrix_run "$B" "$BASE_PY" "$BASE_WHEEL" "base a12eafd warmup-discarded" base-warmup fork
matrix_run "$C" "$CAND_PY" "$CAND_WHEEL" "cand warmup-discarded" cand-warmup fork
# A/A noise pair on the baseline
matrix_run "$B" "$BASE_PY" "$BASE_WHEEL" "base a12eafd aa1" base-aa1 fork
matrix_run "$B" "$BASE_PY" "$BASE_WHEEL" "base a12eafd aa2" base-aa2 fork
# interleaved, counterbalanced: b c c b b c
matrix_run "$B" "$BASE_PY" "$BASE_WHEEL" "base a12eafd r1" base-r1 fork
matrix_run "$C" "$CAND_PY" "$CAND_WHEEL" "cand r1" cand-r1 fork
matrix_run "$C" "$CAND_PY" "$CAND_WHEEL" "cand r2" cand-r2 fork
matrix_run "$B" "$BASE_PY" "$BASE_WHEEL" "base a12eafd r2" base-r2 fork
matrix_run "$B" "$BASE_PY" "$BASE_WHEEL" "base a12eafd r3" base-r3 fork
matrix_run "$C" "$CAND_PY" "$CAND_WHEEL" "cand r3" cand-r3 fork
# stock-driver smoke, one run per wheel
matrix_run "$B" "$BASE_PY" "$BASE_WHEEL" "base a12eafd stock" base-stock stock
matrix_run "$C" "$CAND_PY" "$CAND_WHEEL" "cand stock" cand-stock stock

kill $SAMPLER 2>/dev/null

echo "=== digest + summary $(date -u +%FT%TZ)"
python3 - "$R" <<'PYCHECK'
import glob
import json
import statistics
import sys

r = sys.argv[1]
LEGIDS = [
    "qwen25-0.5b-4bit:short-decode-32",
    "qwen25-0.5b-4bit:long-decode-128",
    "qwen25-0.5b-4bit:longctx-1024-decode-32",
    "qwen25-0.5b-bf16:short-decode-32",
    "qwen25-0.5b-bf16:long-decode-128",
    "qwen25-0.5b-bf16:longctx-1024-decode-32",
]
FORK = {
    LEGIDS[0]: "7fd25a869ff21678",
    LEGIDS[1]: "4cc08910089477fd",
    LEGIDS[2]: "7da83f06ec9f001d",
    LEGIDS[3]: "f26175202f3dabe9",
    LEGIDS[4]: "8690dc83246b39f8",
    LEGIDS[5]: "ff502900d2a179a5",
}
STOCK = dict(FORK)
STOCK.update({
    LEGIDS[3]: "7fc0f968789b1882",
    LEGIDS[4]: "46108ad71157cb4d",
})
RUNS = ["base-warmup", "cand-warmup", "base-aa1", "base-aa2",
        "base-r1", "cand-r1", "cand-r2", "base-r2", "base-r3",
        "cand-r3", "base-stock", "cand-stock"]
failures = []
summary = {}
for run in RUNS:
    path = f"{r}/{run}.json"
    try:
        d = json.load(open(path))
    except Exception as e:  # noqa: BLE001
        failures.append(f"{run}: missing or unparsable ({e})")
        continue
    measured = {leg["leg_id"] for leg in d["legs"] if leg.get("measured")}
    for want_leg in LEGIDS:
        if want_leg not in measured:
            failures.append(f"{run}: required leg not measured: {want_leg}")
    pins = STOCK if run.endswith("-stock") else FORK
    for leg in d["legs"]:
        if not leg.get("measured"):
            continue
        lid = leg["leg_id"]
        got = leg["metrics"].get("generated_ids_sha256_16")
        want = pins.get(lid)
        if want is None:
            failures.append(f"{run}:{lid}: no pin for measured leg")
        elif got != want:
            failures.append(f"{run}:{lid}: digest {got} != pin {want}")
        if "verified=match" not in leg["metrics"].get("provenance_line", ""):
            failures.append(f"{run}:{lid}: provenance not verified=match")
        summary.setdefault(run, {})[lid] = {
            "decode_tok_s": leg["metrics"]["decode_tok_s"],
            "prefill_tok_s": leg["metrics"]["prefill_tok_s"],
        }

paired = {}
for arm in ("base", "cand"):
    reps = [summary.get(f"{arm}-{rep}") for rep in ("r1", "r2", "r3")]
    if any(rep is None for rep in reps):
        failures.append(f"{arm}: missing interleaved repetitions")
        continue
    for lid in LEGIDS:
        vals = [rep[lid]["decode_tok_s"] for rep in reps]
        paired.setdefault(lid, {})[arm] = vals
print("== paired decode tok/s (medians of 3; higher is better)")
for lid in LEGIDS:
    arms = paired.get(lid, {})
    b = statistics.median(arms.get("base", [])) if arms.get("base") else float("nan")
    c = statistics.median(arms.get("cand", [])) if arms.get("cand") else float("nan")
    print(f"{lid}: base={b:.2f} cand={c:.2f} cand/base={c / b:.4f}")
print("== A/A baseline pair (in-window noise)")
for lid in LEGIDS:
    try:
        v1 = summary["base-aa1"][lid]["decode_tok_s"]
        v2 = summary["base-aa2"][lid]["decode_tok_s"]
        print(f"{lid}: {v1:.2f} vs {v2:.2f} ({abs(v1 - v2) / min(v1, v2) * 100:.2f}%)")
    except KeyError:
        failures.append(f"aa: missing leg {lid}")
json.dump({"summary": summary, "paired": paired},
          open(r + "/summary.json", "w"), indent=1)
if failures:
    print("FAILURES:")
    for f in failures:
        print(" " + f)
    sys.exit(1)
print("ALL_DIGEST_ASSERTIONS_PASSED (12 runs x 6 legs, nonempty)")
PYCHECK
rc=$?
echo "checker rc=$rc"
echo "$(date -u +%FT%TZ) window complete, elapsed $(( $(date +%s) - T0 ))s -- LOCK RELEASED"
exit $rc
