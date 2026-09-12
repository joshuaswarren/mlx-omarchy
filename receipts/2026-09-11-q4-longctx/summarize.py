#!/usr/bin/env python3
"""Summarize the Q4LongCtxScaling window into verdict.json.

Reads equiv.json, sdpa_chain.json, leg-*.json, qmm_m262.json from this
receipt directory and writes the verdict: bit-identity result, paired
timings with native fractions, digest gate status, and the prefill
decomposition across the three context rungs.
"""
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

NATIVE = {  # receipts/native-baseline-2026-09-06, qwen25-0.5b-4bit
    "short-decode-32": {"decode_tok_s": 150.57, "prefill_tok_s": 294.1},
    "long-decode-128": {"decode_tok_s": 146.77, "prefill_tok_s": 1213.0},
    "longctx-1024-decode-32": {"decode_tok_s": 140.38,
                               "prefill_tok_s": 1840.9},
}
PINS = {
    "short-decode-32": "7fd25a869ff21678",
    "long-decode-128": "4cc08910089477fd",
    "longctx-1024-decode-32": "7da83f06ec9f001d",
}


def legs_for(arm, rep):
    path = HERE / f"leg-{arm}-{rep}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    out = {}
    for leg in data["legs"]:
        m = leg["metrics"]
        out[leg["workload_id"]] = {
            "decode_tok_s": m.get("decode_tok_s"),
            "prefill_tok_s": m.get("prefill_tok_s"),
            "digest": m.get("generated_ids_sha256_16"),
            "prompt_tokens": m.get("prompt_tokens"),
            "provenance": (data.get("binary_provenance") or {}).get(
                "omarchy", {}).get("verified"),
        }
    return out


def med(vals):
    return round(statistics.median(vals), 2) if vals else None


def main():
    verdict = {"schema": "mlx-omarchy/q4-longctx/verdict/1",
               "base": "origin/main 63c9a8d8 + d89c4e9f (sdpa decode "
                       "batched packed loads)"}

    equiv = json.loads((HERE / "equiv.json").read_text())
    verdict["bit_identity_gate"] = {
        "pass": equiv["bit_exact_all"],
        "cases": equiv["cases"],
        "failures": equiv["failures"],
    }

    chain = json.loads((HERE / "sdpa_chain.json").read_text())
    verdict["sdpa_chain_paired"] = chain["rows"]

    packed_runs, scalar_runs = {}, {}
    digest_failures = []
    for rep in ("r1", "r2"):
        for arm, sink in (("packed", packed_runs), ("scalar", scalar_runs)):
            legs = legs_for(arm, rep)
            if not legs:
                continue
            for wl, leg in legs.items():
                sink.setdefault(wl, []).append(leg)
    matrix = {}
    for wl, native in NATIVE.items():
        entry = {"native": native}
        for arm, sink in (("packed", packed_runs), ("scalar", scalar_runs)):
            runs = sink.get(wl, [])
            digests = {r["digest"] for r in runs}
            provs = {r["provenance"] for r in runs}
            dec = med([r["decode_tok_s"] for r in runs])
            pre = med([r["prefill_tok_s"] for r in runs])
            entry[arm] = {
                "decode_tok_s_median": dec,
                "prefill_tok_s_median": pre,
                "reps": len(runs),
                "digests": sorted(digests),
                "digest_matches_pin": digests == {PINS[wl]},
                "provenance": sorted(x for x in provs if x),
            }
            if digests and digests != {PINS[wl]}:
                digest_failures.append({"workload": wl, "arm": arm,
                                        "digests": sorted(digests)})
            if dec:
                entry[arm]["decode_fraction_of_native"] = round(
                    dec / native["decode_tok_s"], 3)
            if pre:
                entry[arm]["prefill_fraction_of_native"] = round(
                    pre / native["prefill_tok_s"], 3)
        if (entry.get("packed", {}).get("decode_tok_s_median")
                and entry.get("scalar", {}).get("decode_tok_s_median")):
            p, s = (entry["packed"]["decode_tok_s_median"],
                    entry["scalar"]["decode_tok_s_median"])
            entry["packed_vs_scalar_decode_pct"] = round(
                100.0 * (p - s) / s, 2)
            pw = 1000.0 / p
            sw = 1000.0 / s
            entry["per_token_ms"] = {"packed": round(pw, 3),
                                     "scalar": round(sw, 3),
                                     "native": round(
                                         1000.0 / native["decode_tok_s"], 3)}
        matrix[wl] = entry
    verdict["leg_matrix"] = matrix
    verdict["digest_gate"] = {
        "all_pins_hold_both_arms": not digest_failures,
        "failures": digest_failures,
    }

    qmm262 = json.loads((HERE / "qmm_m262.json").read_text())
    verdict["qmm_m262"] = qmm262

    ok = (verdict["bit_identity_gate"]["pass"]
          and verdict["digest_gate"]["all_pins_hold_both_arms"]
          and all(len(entry.get("packed", {}).get("digests", [])) > 0
                  for entry in matrix.values()))
    verdict["gate"] = "PASS" if ok else "FAIL"
    print(json.dumps(verdict, indent=2))
    (HERE / "verdict.json").write_text(json.dumps(verdict, indent=2))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
