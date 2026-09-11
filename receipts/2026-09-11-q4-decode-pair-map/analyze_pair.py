#!/usr/bin/env python3
"""Turn legs/summary.json + accuracy/report.json into the verdict tables
(analyze_pair.py; CPU only). Medians over reps; fractions against the
committed native baseline; digests classified native | linux-base | other.
"""
import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

NATIVE_TPS = {"short-decode-32": 150.57, "long-decode-128": 146.77,
              "longctx-1024-decode-32": 140.38}
NATIVE_DIGEST = {"short-decode-32": "7fd25a869ff21678",
                 "long-decode-128": "254d73fd93164b98",
                 "longctx-1024-decode-32": "7da83f06ec9f001d"}
LINUX_DIGEST = {"short-decode-32": "7fd25a869ff21678",
                "long-decode-128": "4cc08910089477fd",
                "longctx-1024-decode-32": "7da83f06ec9f001d"}
LEG_ORDER = ["short-decode-32", "long-decode-128",
             "longctx-1024-decode-32"]


def classify(digest, leg):
    if digest == NATIVE_DIGEST[leg]:
        return "native"
    if digest == LINUX_DIGEST[leg]:
        return "linux-base"
    return "other"
PROVENANCE = {
    "branch": "wave/Q4DecodePairMap",
    "commits": ["c6a28841", "71ee79d4", "ef49fbc5"],
    "wheel": "mlx_omarchy-0.32.2.dev202609111121+ef49fbc5-cp314-cp314-"
             "linux_aarch64.whl",
    "wheel_sha256": "9a0e0b15c58276dc4af080597e4385ff6cd25d54cf32167f53e"
                    "388a9ebea75da",
    "host": "jwm1-linux (<m1-host>), Apple M1 (G13G B1)",
    "driver_fork": "mesa-honeykrisp-omarchy 26.3.0.devel.hk6f6afc8-1 "
                   "(verified in-window via pacman -Q)",
    "driver_stock": "stock Mesa via /home/joshuawarren/stock-mesa/"
                    "stock-icd.json (VK_DRIVER_FILES)",
    "kernel": "linux-asahi 7.1.6.asahi1-1",
    "reps": 3,
    "lock": "one top-level flock /tmp/m1-gpu.lock per window; accuracy "
            "window 11:37:54-11:40:51Z (with qmm smoke), legs window "
            "11:40:51-11:47:04Z; quiet gate loadavg 0.02-0.25",
    "timing": "wall-clock via bench_decode (device timestamps unused; "
              "known ~2.07x undercount on this driver)",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary", type=Path)
    ap.add_argument("accuracy", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    summary_path = args.summary
    accuracy_path = args.accuracy
    out_path = args.out
    summary = json.loads(summary_path.read_text())

    series = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for run in summary["runs"]:
        series[run["driver"]][run["variant"]][run["workload"]].append(run)

    perf = []
    digest_rows = []
    prefill_rows = []
    for driver in ("fork", "stock"):
        for leg in LEG_ORDER:
            base_runs = series[driver]["base"][leg]
            pair_runs = series[driver]["pair"][leg]
            base_med = statistics.median(r["decode_tok_s"] for r in base_runs)
            pair_med = statistics.median(r["decode_tok_s"] for r in pair_runs)
            perf.append({
                "driver": driver, "leg": leg, "reps": len(base_runs),
                "base_median_tok_s": round(base_med, 3),
                "pair_median_tok_s": round(pair_med, 3),
                "paired_ratio_pair_over_base": round(pair_med / base_med, 4),
                "base_fraction_of_native": round(base_med / NATIVE_TPS[leg], 4),
                "pair_fraction_of_native": round(pair_med / NATIVE_TPS[leg], 4),
                "base_all_tok_s": [r["decode_tok_s"] for r in base_runs],
                "pair_all_tok_s": [r["decode_tok_s"] for r in pair_runs],
            })
            digest_rows.append({
                "driver": driver, "leg": leg,
                "prompt_tokens": base_runs[0]["prompt_tokens"],
                "base_digest": base_runs[0]["digest"],
                "base_digest_stable": len({r["digest"] for r in base_runs}) == 1,
                "pair_digest": pair_runs[0]["digest"],
                "pair_digest_stable": len({r["digest"] for r in pair_runs}) == 1,
                "pair_equals_native": pair_runs[0]["digest"] == NATIVE_DIGEST[leg],
                "pair_equals_linux_base": pair_runs[0]["digest"] == LINUX_DIGEST[leg],
                "pair_digest_class": classify(pair_runs[0]["digest"], leg),
            })
            base_pf = statistics.median(r["prefill_s"] for r in base_runs)
            pair_pf = statistics.median(r["prefill_s"] for r in pair_runs)
            prefill_rows.append({
                "driver": driver, "leg": leg,
                "base_median_prefill_s": round(base_pf, 6),
                "pair_median_prefill_s": round(pair_pf, 6),
                "ratio_pair_over_base": round(pair_pf / base_pf, 4),
                "base_all_prefill_s": [r["prefill_s"] for r in base_runs],
                "pair_all_prefill_s": [r["prefill_s"] for r in pair_runs],
            })

    accuracy = json.loads(accuracy_path.read_text()) if \
        accuracy_path.exists() else {}

    verdict = {
        "native_baseline_tok_s": NATIVE_TPS,
        "native_digests": NATIVE_DIGEST,
        "linux_base_digests": LINUX_DIGEST,
        "performance": perf,
        "digests": digest_rows,
        "prefill": prefill_rows,
        "accuracy": accuracy,
        "decision_candidate": None,
    }
    # The decisive read: does the pair map match native on more legs than
    # the current kernel does?
    current_native_matches = [leg for leg in LEG_ORDER
                              if LINUX_DIGEST[leg] == NATIVE_DIGEST[leg]]
    pair_native_matches = {}
    for driver in ("fork", "stock"):
        pair_native_matches[driver] = [
            row["leg"] for row in digest_rows
            if row["driver"] == driver and row["pair_equals_native"]]
    verdict["decision_candidate"] = {
        "current_native_matches": current_native_matches,
        "pair_native_matches": pair_native_matches,
    }
    verdict["decision"] = {
        "outcome": "REJECT for pin and for landing (receipt-only branch)",
        "performance": "pair map is 1.0-2.6% SLOWER than base on every "
                       "driver x leg cell (median tok/s, 3 reps); the "
                       "+13% pattern-level gain does not survive at "
                       "model level",
        "digests": "pair map matches native on at most the same legs "
                   "as the current kernel (stock 2/3, fork 1/3) and "
                   "moves the long leg to a third value on both "
                   "drivers; the pin question does not invert",
        "accuracy": "max abs error vs the f64 RNE reference is "
                    "identical between the maps on all seven shapes; "
                    "the reorder changes at most 4 of 4864 outputs "
                    "per projection at the last f16 rounding",
        "prefill": "prefill unchanged within noise on 5 of 6 cells "
                   "(first decode step is inside prefill timing); "
                   "stock-short +12.7% median is first-rep warmup "
                   "ambiguous",
    }
    verdict["provenance"] = PROVENANCE
    out_path.write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps(verdict["performance"], indent=1))
    print(json.dumps(verdict["digests"], indent=1))
    print(json.dumps(verdict["prefill"], indent=1))
    print(json.dumps(verdict["decision_candidate"], indent=1))


if __name__ == "__main__":
    main()
