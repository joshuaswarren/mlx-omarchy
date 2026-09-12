#!/usr/bin/env bash
# Deferred-post-barrier A/B window 3 (wave/Bf16EncoderBoundary f5649936).
# Window 2 was a valid BUILD/digest run but measured gate-off vs gate-off:
# the candidate arm never set MLX_OMARCHY_DEFERRED_POST_BARRIERS (probe:
# post_barriers_deferred=0, barriers_emitted=400 for 200 dispatches).
# Window 3 re-measures with the gate ON for every candidate run:
#   - probes: candidate wheel gate-off (expect base pattern), candidate
#     wheel gate-on (expect deferred ~= 1/dispatch, emitted ~= 1/dispatch);
#     the base wheel has no mlx_omarchy_trace_barriers symbol (expected
#     AttributeError, quoted as baseline-composition evidence).
#   - fork matrix: base r1-r3 (gate irrelevant), cand r1-r3 with the gate
#     env set; discarded warmups both arms; A/A pair on base.
#   - stock smoke: one base run, one cand run with the gate env set.
# One top-level flock /tmp/m1-gpu.lock; no builds (wheels reused).
set -uo pipefail
BASE_COMMIT=a12eafd994af881dbc2b39c0b9807523f27b171a
CAND_COMMIT=f5649936f68a2262e6f1e51ad9b146ec035808cb
B="$HOME/src/mlx-enc-base"
C="$HOME/src/mlx-enc-cand"
R="$HOME/enc-ab-window3"
STOCK_ICD="$HOME/stock-mesa/stock-icd.json"

exec 9>/tmp/m1-gpu.lock
echo "$(date -u +%FT%TZ) waiting for m1-gpu.lock pid $$"
flock -w 3600 9 || { echo "FATAL: lock wait exceeded"; exit 4; }
echo "$(date -u +%FT%TZ) LOCK ACQUIRED"
T0=$(date +%s)
mkdir -p "$R"

{
  echo "host=$(hostname) kernel=$(uname -r) nproc=$(nproc)"
  pacman -Q mesa-honeykrisp-omarchy 2>/dev/null
  echo "base_commit=$BASE_COMMIT"
  echo "cand_commit=$CAND_COMMIT"
  echo "base_wheel_sha256=$(sha256sum "$B"/dist/mlx_omarchy-*.whl | cut -d' ' -f1)"
  echo "cand_wheel_sha256=$(sha256sum "$C"/dist/mlx_omarchy-*.whl | cut -d' ' -f1)"
} > "$R/drivers.txt"
cat "$R/drivers.txt"

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

matrix_run() {  # tree py wheel label out icd deferred
  local tree="$1" py="$2" wheel="$3" label="$4" out="$5" icd="$6" deferred="$7"
  local -a ICD=(env)
  [ "$icd" = stock ] && ICD+=("VK_DRIVER_FILES=$STOCK_ICD")
  [ "$deferred" = 1 ] && ICD+=("MLX_OMARCHY_DEFERRED_POST_BARRIERS=1")
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

probe_run() {  # py label deferred
  local py="$1" label="$2" deferred="$3"
  local -a ENVV=(env)
  [ "$deferred" = 1 ] && ENVV+=("MLX_OMARCHY_DEFERRED_POST_BARRIERS=1")
  timeout 300 "${ENVV[@]}" "$py" "$R/probe.py" \
    > "$R/probe-$label.txt" 2>&1
  echo "probe $label exit=$?"
  grep '^PROBE' "$R/probe-$label.txt" || tail -2 "$R/probe-$label.txt"
}

probe_run "$CAND_PY" cand-off 0
probe_run "$CAND_PY" cand-on 1
probe_run "$BASE_PY" base 0

# discarded warmups, one full matrix per arm
matrix_run "$B" "$BASE_PY" "$BASE_WHEEL" "base a12eafd warmup-discarded" base-warmup fork 0
matrix_run "$C" "$CAND_PY" "$CAND_WHEEL" "cand deferred warmup-discarded" cand-warmup fork 1
# A/A noise pair on the baseline
matrix_run "$B" "$BASE_PY" "$BASE_WHEEL" "base a12eafd aa1" base-aa1 fork 0
matrix_run "$B" "$BASE_PY" "$BASE_WHEEL" "base a12eafd aa2" base-aa2 fork 0
# interleaved, counterbalanced: b c c b b c; candidate runs gate ON
matrix_run "$B" "$BASE_PY" "$BASE_WHEEL" "base a12eafd r1" base-r1 fork 0
matrix_run "$C" "$CAND_PY" "$CAND_WHEEL" "cand deferred r1" cand-r1 fork 1
matrix_run "$C" "$CAND_PY" "$CAND_WHEEL" "cand deferred r2" cand-r2 fork 1
matrix_run "$B" "$BASE_PY" "$BASE_WHEEL" "base a12eafd r2" base-r2 fork 0
matrix_run "$B" "$BASE_PY" "$BASE_WHEEL" "base a12eafd r3" base-r3 fork 0
matrix_run "$C" "$CAND_PY" "$CAND_WHEEL" "cand deferred r3" cand-r3 fork 1
# stock-driver smoke, one run per arm, candidate with the gate env set
matrix_run "$B" "$BASE_PY" "$BASE_WHEEL" "base a12eafd stock" base-stock stock 0
matrix_run "$C" "$CAND_PY" "$CAND_WHEEL" "cand deferred stock" cand-stock stock 1

kill $SAMPLER 2>/dev/null

echo "=== digest + summary $(date -u +%FT%TZ)"
python3 - "$R" <<'PYCHECK'
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
    b = statistics.median(arms["base"]) if arms.get("base") else float("nan")
    c = statistics.median(arms["cand"]) if arms.get("cand") else float("nan")
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
