#!/bin/bash
# TDT joint ILP2 COUNTERBALANCED confirmation window (jw16).
# Pre-registered per Main: the A/B's 6/6 A-then-B paired wins may carry
# order/warmup bias, so measured rounds ALTERNATE arm order (AB, BA, AB...),
# schedule fixed by this commit before execution, no tuning interim.
# ONE wheel (corrected libmlx bytes), two pkg overlays (origin/main vs
# agent/tdt-joint-ilp2); full fatal pins per run; whole-path stage timing and
# per-round paired deltas recorded. If the repeat fails, NO-LAND.
#
# Usage: run on jw16 as
#   TDT_AB_LIBMLX_SHA=<libmlx16 sha of the built wheel> ./tdt-pairload-ab-window.sh
set -uo pipefail
W=${TDT_AB_OUT:-/var/tmp/tdt-pairload-ab}
# PINS (full SHAs, resolved at pre-registration - not mutable refs):
#   BASE fabe6697052d2afdebb60a752c7ab84175dca9a8 (post-#12 main)
#   CAND 040f456eb867595f984656fa2580a7a6a24584aa (ILP2 + orchestration)
# Exercised file (vulkan_tdt_loop.py) at pins:
#   BASE blob 9047bbb7e91ed184c97dabaf49e40683787e8308 sha256 76aef6670417b46d1ef8a57f1a2fe9c5db0e13a8ac42c57a810765c89b903a0a
#   CAND blob d53cd1d886a560f4c7f0a2a66255b42f083c761a sha256 fa9e23f4e37afa0329a59afb3127d1d43df3cabf6c206fda1fdfc5dba6ca1eac
# Orchestration-only proof: git diff f7fb7aec..040f456e -- overlay/tools/coreml/ is EMPTY.
BASE_COMMIT=${TDT_AB_BASE:-fabe6697052d2afdebb60a752c7ab84175dca9a8}
CAND_COMMIT=${TDT_AB_CAND:-040f456eb867595f984656fa2580a7a6a24584aa}
FORK=${TDT_AB_FORK:-$HOME/src/mlx-omarchy}
RUN=/var/tmp/ParakeetE2EJw16
RUNNER=/var/tmp/encwall-v071/base/vulkan_encoder.py
MODEL=$HOME/.cache/mlx-omarchy/parakeet-reference/mweinbach1/parakeet-tdt-0.6b-v3-coreml/b650695c2322ee5281dff48d7345b2f3a58ff018
MEASURED=${TDT_AB_MEASURED:-6}
mkdir -p "$W"
unset PYTHONPATH
export MLX_OMARCHY_SPIRV_CACHE=/var/tmp/MelFrontendPerf/spirv-ab.3KDGHZ
unset HK_PERF HK_PERFTEST MLX_OMARCHY_GATED_BARRIERS MLX_OMARCHY_GPU_PROFILE ANE_OP_WALL || true

exec 9>/tmp/m1-gpu.lock
flock -x -w 3600 9 || { echo "LOCK-FAIL"; exit 4; }
echo "lock held $(date -Iseconds) inode $(stat -c %i /tmp/m1-gpu.lock)"

# ---- prep: venv from the corrected-libmlx wheel + two pkg trees
cd "$FORK"
git fetch origin --prune || exit 4
git rev-parse --verify "$BASE_COMMIT^{commit}" >/dev/null || exit 4
git rev-parse --verify "$CAND_COMMIT^{commit}" >/dev/null || exit 4
[ "$(git merge-base "$BASE_COMMIT" "$CAND_COMMIT")" = "$(git rev-parse "$BASE_COMMIT^{commit}")" ] \
  || { echo "CANDIDATE-NOT-BASED-ON-BASE"; exit 4; }

if [ -n "${TDT_AB_WHEEL:-}" ]; then
  WHL=$TDT_AB_WHEEL
else
  (cd "$FORK" && scripts/build-wheel.sh) || exit 4
  WHL=$(ls -t "$FORK"/dist/mlx_omarchy-*aarch64.whl | head -1)
fi
if [ ! -x "$W/venv/bin/python" ]; then
  "${TDT_AB_PYTHON:-python3.14}" -m venv "$W/venv"
  "$W/venv/bin/pip" install --quiet "$WHL" || exit 4
fi
PY=$W/venv/bin/python
[ -n "${TDT_AB_LIBMLX_SHA:-}" ] && {
  "$PY" /var/tmp/v063-jw16/scripts/venv-identity-guard.py --expect "$TDT_AB_LIBMLX_SHA" "$W/venv" || exit 4
}
"$PY" -c 'import importlib.metadata as M; print("venv dist:", M.version("mlx-omarchy"))'

for arm in base cand; do
  ref=$([ "$arm" = base ] && echo "$BASE_COMMIT" || echo "$CAND_COMMIT")
  rm -rf "$W/pkg-$arm"; mkdir -p "$W/pkg-$arm"
  git archive "$ref" overlay/tools/coreml | tar -x -C "$W/pkg-$arm"
  mv "$W/pkg-$arm/overlay/tools/coreml" "$W/pkg-$arm"
done
diff -rq "$W/pkg-base/coreml" "$W/pkg-cand/coreml" | sed 's/^/pkg-delta: /'

run_one () { # arm idx pkg
  local name=$1 pkg=$2
  local out=$W/out-$name scratch=$W/scratch-$name
  rm -rf "$out" "$scratch"; mkdir -p "$out" "$scratch"
  MLX_OMARCHY_PLACED=AC ANE_ISLAND_MODE=resident-batch \
  "$PY" "$RUN/fused_e2e.py" \
    --audio /var/tmp/ParakeetE2E/audio/fixture.flac \
    --golden /var/tmp/EncoderParityAne/capture \
    --model "$MODEL" \
    --pkg "$pkg" \
    --encoder-runner "$RUNNER" \
    --source /var/tmp/EncoderParityAne/encoder-source \
    --ane-reference /var/tmp/EncoderParityAne/capture/encoder_hidden.npy \
    --bundles /var/tmp/jw16-conv-place/bundles-conv \
    --worker /var/tmp/jw16-oproj-place/mlx-omarchy-ane-worker \
    --libane /var/tmp/jw16-oproj-place/libane-strict-fill.so \
    --scratch "$scratch" --out "$out" \
    --deadline-ms 20000 > "$W/log-$name.txt" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then echo "RUN-FAILED $name rc=$rc"; tail -8 "$W/log-$name.txt"; exit 3; fi
  cp "$out/token_ids.json" "$W/decoder-trace-$name.json" 2>/dev/null
  echo "run $name done $(date -Iseconds)"
}

# PRE-REGISTERED counterbalanced schedule (fixed before the run):
# warm-up each arm (excluded), then MEASURED rounds alternate arm ORDER to
# cancel order/warmup bias — odd rounds A(base)-then-B(cand), even rounds
# B(cand)-then-A(base). The schedule is committed before execution and is
# not tuned between runs. Full paired deltas + whole-path stage timing are
# recorded per round.
run_one base-warm "$W/pkg-base"
run_one cand-warm "$W/pkg-cand"
: > "$W/schedule.txt"
for i in $(seq 1 "$MEASURED"); do
  if [ $((i % 2)) -eq 1 ]; then
    order="AB"
    run_one "base-$i" "$W/pkg-base"
    run_one "cand-$i" "$W/pkg-cand"
  else
    order="BA"
    run_one "cand-$i" "$W/pkg-cand"
    run_one "base-$i" "$W/pkg-base"
  fi
  echo "round $i: $order" >> "$W/schedule.txt"
done

python3 - "$W" "$MEASURED" <<'PYEOF'
import hashlib, json, statistics, sys
W, N = sys.argv[1], int(sys.argv[2])
TRANSCRIPT = "db501a8c080380ea027ffa50a4b4956c39df77cb692c4fb78e556311a11a0790"
HIDDEN = "38c73261f29230276ed76f1fc017b76b024156d79218bd5f1347fdc7e7d43ec7"
MEL = "5b54f4a9a2ba3434cd69b6e48e6780d3bcb6c635d9ce85cda3d85c60f2455bde"
STAGES = ["audio_load", "mel_frontend", "encoder_ane", "decoder_load", "tdt_decode", "detokenize"]

def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()

def collect(arm):
    rows = []
    for i in range(1, N + 1):
        r = json.load(open(f"{W}/out-{arm}-{i}/e2e-report.json"))
        st = {s["stage"]: s.get("wall_ms") for s in r["stages"] if isinstance(s, dict) and "stage" in s}
        ane = r.get("ane") or {}
        ex = r.get("execution") or {}
        row = {
            "run": i, "status": r["status"],
            "control": ex.get("control"), "fallback": ex.get("tdt_fallback_reason"),
            "prefix": r["layers"]["layer_6_decoder_sequence"].get("matching_prefix_length"),
            "bounds": r["layers"]["layer_5_encoder"]["all_bounds_pass"],
            "cpu_ev": r["execution"]["cpu_tensor_events"], "timeouts": ane.get("timeouts"),
            "transcript_sha": sha(f"{W}/out-{arm}-{i}/transcript.txt"),
            "hidden_sha": sha(f"{W}/out-{arm}-{i}/encoder_hidden.npy"),
            "mel_sha": sha(f"{W}/out-{arm}-{i}/mel.npy"),
            "stages": {k: st.get(k) for k in STAGES},
        }
        rows.append(row)
    return rows

def gate(rows, arm):
    for row in rows:
        bad = []
        if row["status"] != "match": bad.append(f"status={row['status']}")
        if row["prefix"] != 104: bad.append(f"prefix={row['prefix']}")
        if row["control"] != "gpu-loop": bad.append(f"control={row['control']}")
        if row["fallback"] is not None: bad.append(f"fallback={row['fallback']}")
        if row["transcript_sha"] != TRANSCRIPT: bad.append("transcript")
        if row["hidden_sha"] != HIDDEN: bad.append("hidden")
        if row["mel_sha"] != MEL: bad.append("mel")
        if not row["bounds"]: bad.append("bounds")
        if row["cpu_ev"] != 0: bad.append(f"cpu_ev={row['cpu_ev']}")
        if row["timeouts"] != 0: bad.append(f"timeouts={row['timeouts']}")
        if bad:
            print(f"GATE-FAIL {arm}-{row['run']}: {', '.join(bad)}")
            sys.exit(2)
    print(f"{arm} gate: {N}/{N} pins-EXACT")

summary = {}
for arm in ("base", "cand"):
    rows = collect(arm)
    gate(rows, arm)
    tdt = sorted(r["stages"]["tdt_decode"] for r in rows)
    summary[arm] = {
        "rows": rows,
        "tdt_decode_median_ms": round(statistics.median(tdt), 1),
        "tdt_decode_per_run_ms": [round(r["stages"]["tdt_decode"], 1) for r in rows],
        "stage_medians_ms": {k: round(statistics.median([r["stages"][k] for r in rows]), 1) for k in STAGES},
        "stage_per_run_ms": {k: [r["stages"][k] for r in rows] for k in STAGES},
    }
delta = summary["cand"]["tdt_decode_median_ms"] - summary["base"]["tdt_decode_median_ms"]
summary["tdt_decode_delta_ms"] = round(delta, 1)
# pre-registered counterbalanced schedule + per-round paired deltas
summary["schedule"] = [ln.strip() for ln in open(f"{W}/schedule.txt").read().splitlines() if ln.strip()]
summary["paired_deltas_by_round_ms"] = [
    round(summary["cand"]["tdt_decode_per_run_ms"][i]
          - summary["base"]["tdt_decode_per_run_ms"][i], 1)
    for i in range(N)
]
summary["paired_wins_cand"] = sum(
    1 for d in summary["paired_deltas_by_round_ms"] if d < 0)
json.dump(summary, open(f"{W}/summary.json", "w"), indent=2)
print(json.dumps({a: {"tdt_decode_median_ms": summary[a]["tdt_decode_median_ms"],
                      "per_run": summary[a]["tdt_decode_per_run_ms"]} for a in ("base", "cand")}, indent=2))
print("paired deltas by round (cand-base):", summary["paired_deltas_by_round_ms"])
print("schedule:", summary["schedule"])
print(f"TDT-AB-DONE delta={summary['tdt_decode_delta_ms']}ms paired_wins={summary['paired_wins_cand']}/{N}")
PYEOF
echo "lock released $(date -Iseconds)"
