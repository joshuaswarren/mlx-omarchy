#!/usr/bin/env python3
"""Cross-session digest-oracle and denominator delta for 16m1mbp.

Compares the 2026-09-11 accepted native matrix (this receipt) against the
committed 2026-09-10 matrix (receipts/2026-09-10-native-macos-metal-baseline/
native-baseline-16m1mbp/) and the committed base-M1 native denominator
(receipts/native-baseline-2026-09-06/), and emits reproducibility-<date>.json:

- per leg: digest set across ALL reps of BOTH sessions, reference-digest
  match, digest stability verdict, decode/prefill median drift (%),
  clean/contended rep counts per session;
- same-die die-difference sanity row vs base M1 (direction/magnitude only;
  cross-chip, never a parity divisor).

Pure post-processing over committed JSON; no measurement.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PREV = (REPO / "receipts/2026-09-10-native-macos-metal-baseline"
        / "native-baseline-16m1mbp")
BASE_M1 = REPO / "receipts/native-baseline-2026-09-06"
LEGS = ["q4_short", "q4_long", "q4_longctx",
        "bf16_short", "bf16_long", "bf16_longctx"]
# Committed base-M1 receipt leg names (2026-09-06 protocol naming).
BASE_M1_KEYS = {
    "q4_short": "qwen25-0.5b-4bit:short-decode-32",
    "q4_long": "qwen25-0.5b-4bit:long-decode-128",
    "q4_longctx": "qwen25-0.5b-4bit:longctx-1024-decode-32",
    "bf16_short": "qwen25-0.5b-bf16:short-decode-32",
    "bf16_long": "qwen25-0.5b-bf16:long-decode-128",
    "bf16_longctx": "qwen25-0.5b-bf16:longctx-1024-decode-32",
}


def load(path):
    return json.loads(Path(path).read_text())


def drift(new, old):
    return round(100.0 * (new - old) / old, 2)


def main(cur_path, out_path):
    cur = load(cur_path)
    prev = load(PREV / "native-baseline-16M1MBP.json")
    base = load(BASE_M1 / "native-2026-09-06-summary.json")

    legs, all_digest_sets = {}, {}
    for leg in LEGS:
        c, p = cur["legs"][leg], prev["legs"][leg]
        cur_digests = sorted({r["ids_sha256_16"] for r in c["all_rows"]})
        prev_digests = sorted({r["ids_sha256_16"] for r in p["all_rows"]})
        ref = cur["reference_native_digests"][leg]
        bkey = BASE_M1_KEYS[leg]
        bleg = base[bkey]
        legs[leg] = {
            "reference_native_digest": ref,
            "digests_2026_09_10": prev_digests,
            "digests_2026_09_11": cur_digests,
            "digest_stable_within_2026_09_11": len(cur_digests) == 1,
            "digest_stable_within_2026_09_10": len(prev_digests) == 1,
            "digest_stable_across_sessions":
                cur_digests == prev_digests == [ref],
            "reproducible_digest_oracle":
                len(cur_digests) == 1 and cur_digests[0] == ref,
            "decode_median_2026_09_10": p["decode_tok_s"]["median"],
            "decode_median_2026_09_11": c["decode_tok_s"]["median"],
            "decode_median_drift_pct": drift(
                c["decode_tok_s"]["median"], p["decode_tok_s"]["median"]),
            "prefill_median_2026_09_10": p["prefill_tok_s"]["median"],
            "prefill_median_2026_09_11": c["prefill_tok_s"]["median"],
            "prefill_median_drift_pct": drift(
                c["prefill_tok_s"]["median"], p["prefill_tok_s"]["median"]),
            "clean_reps_2026_09_10": p["clean_reps"],
            "clean_reps_2026_09_11": c["clean_reps"],
            "contended_reps_2026_09_10": p["contended_reps_excluded"],
            "contended_reps_2026_09_11": c["contended_reps_excluded"],
            "base_m1_native_decode_median": bleg["decode_tok_s_median"],
            "base_m1_native_prefill_median": bleg["prefill_tok_s_median"],
            "die_decode_ratio_max_vs_base_m1": round(
                c["decode_tok_s"]["median"] / bleg["decode_tok_s_median"], 3),
            "die_prefill_ratio_vs_base_m1": round(
                c["prefill_tok_s"]["median"] / bleg["prefill_tok_s_median"],
                3),
        }
        all_digest_sets[leg] = cur_digests + prev_digests

    out = {
        "schema": "native-macos-cross-session-delta/1",
        "date": "2026-09-11",
        "compares": {
            "this_session": str(Path(cur_path).name),
            "previous_session": "receipts/2026-09-10-native-macos-metal-baseline"
                                "/native-baseline-16m1mbp/native-baseline-16M1MBP.json",
            "base_m1_denominator": "receipts/native-baseline-2026-09-06/"
                                   "native-2026-09-06-summary.json",
        },
        "digest_oracle_verdict": {
            "all_legs_reproducible": all(
                legs[l]["reproducible_digest_oracle"] for l in LEGS),
            "all_legs_stable_across_sessions": all(
                legs[l]["digest_stable_across_sessions"] for l in LEGS),
        },
        "versions_2026_09_11": {
            "mlx": cur["versions"]["mlx"],
            "mlx_lm": cur["versions"]["mlx_lm"],
            "macos": cur["host"]["macos"],
            "macos_build": cur["host"]["macos_build"],
        },
        "legs": legs,
    }
    Path(out_path).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps(out["digest_oracle_verdict"]))
    for leg in LEGS:
        l = legs[leg]
        print(f"{leg}: digests {l['digests_2026_09_11']} "
              f"cross-session={l['digest_stable_across_sessions']} "
              f"decode drift {l['decode_median_drift_pct']:+.2f}% "
              f"prefill drift {l['prefill_median_drift_pct']:+.2f}% "
              f"vs baseM1 {l['die_decode_ratio_max_vs_base_m1']}x")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
