#!/usr/bin/env python3
"""Generate verdict.json for the BF16 prefill qualification from stage-2
artifacts. Inputs (all inside this receipt directory):
  matrix/           - pulled from jwm1 (warmup + r{1,2,3}-{fork,stock}-{base,cand})
  summary.json      - output of ../2026-09-10-bf16-prefill-close/summarize_prefill.py
  suite-status.json - family/runtime suite rc + row-mismatch scan per cell
  provenance.json   - commits, wheels, host, lock windows
Decision: LAND requires summary failures empty (six canonical Q4 digests
unchanged on both drivers, BF16 per-driver pins unchanged, 1K native
digest unchanged, clean/AC/prompt/gen checks) AND the paired BF16 prefill
medians repeating the gain (cand > base) on both drivers at all three
legs. Anything else is a negative naming the binding constraint.
"""
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BF16_LEGS = [
    "qwen25-0.5b-bf16:short-decode-32",
    "qwen25-0.5b-bf16:long-decode-128",
    "qwen25-0.5b-bf16:longctx-1024-decode-32",
]
LEG_LABEL = {
    "qwen25-0.5b-bf16:short-decode-32": "BF16 short (30/32)",
    "qwen25-0.5b-bf16:long-decode-128": "BF16 long (262/128)",
    "qwen25-0.5b-bf16:longctx-1024-decode-32": "BF16 1K ctx (1053/32)",
}
NATIVE_PREFILL = {
    "qwen25-0.5b-bf16:short-decode-32": 232.6,
    "qwen25-0.5b-bf16:long-decode-128": 1007.7,
    "qwen25-0.5b-bf16:longctx-1024-decode-32": 1655.7,
}
Q4_CANON = {
    "qwen25-0.5b-4bit:short-decode-32": "7fd25a869ff21678",
    "qwen25-0.5b-4bit:long-decode-128": "4cc08910089477fd",
    "qwen25-0.5b-4bit:longctx-1024-decode-32": "7da83f06ec9f001d",
}
BF16_PINS = {
    "qwen25-0.5b-bf16:short-decode-32": {"fork": "f26175202f3dabe9",
                                          "stock": "7fc0f968789b1882"},
    "qwen25-0.5b-bf16:long-decode-128": {"fork": "8690dc83246b39f8",
                                          "stock": "46108ad71157cb4d"},
    "qwen25-0.5b-bf16:longctx-1024-decode-32": {"fork": "ff502900d2a179a5",
                                                 "stock": "ff502900d2a179a5"},
}


def cand_digests(summary, leg, drv):
    """Digests of the cand cells for one (leg, driver) across reps.

    summary.json keys digests as "label|leg" with label = r{rep}-{drv}-cell;
    the summarizer already fails a cell whose digest varies across reps."""
    digs = set()
    for rep in (1, 2, 3):
        entry = summary["digests"].get(f"r{rep}-{drv}-cand|{leg}")
        if entry:
            digs.add(entry[0])
    return sorted(digs)


def main():
    summary = json.loads((HERE / "summary.json").read_text())
    suite = json.loads((HERE / "suite-status.json").read_text())
    failures = list(summary.get("failures", []))

    digest_gates = {}
    for leg, want in Q4_CANON.items():
        for drv in ("fork", "stock"):
            got_list = cand_digests(summary, leg, drv)
            got = got_list[0] if len(got_list) == 1 else None
            key = f"{leg}|{drv}"
            digest_gates[key] = {"expected": want, "got": got,
                                 "unchanged": got == want}
            if got != want:
                failures.append(f"q4 digest moved: {key} {got_list} != {want}")
    for leg, per in BF16_PINS.items():
        for drv, want in per.items():
            got_list = cand_digests(summary, leg, drv)
            got = got_list[0] if len(got_list) == 1 else None
            key = f"{leg}|{drv}"
            digest_gates[key] = {"expected": want, "got": got,
                                 "unchanged": got == want}
            if got != want:
                failures.append(f"bf16 pin moved: {key} {got_list} != {want}")

    if not suite.get("all_pass"):
        failures.append(f"suites: {suite.get('detail', 'not all pass')}")

    rates = {}
    for label, legs in summary["rates"].items():
        for lid in BF16_LEGS:
            if lid in legs:
                rates.setdefault(label, {})[lid] = legs[lid]["prefill_median"]

    missing = [f"r{r}-{d}-{c}" for r in (1, 2, 3) for d in ("fork", "stock")
               for c in ("base", "cand") if f"r{r}-{d}-{c}" not in rates]
    for lbl in missing:
        failures.append(f"missing matrix cell {lbl}")

    perf = {}
    if not missing:
        for drv in ("fork", "stock"):
            perf[drv] = {}
            for lid in BF16_LEGS:
                b = statistics.median(rates[f"r{r}-{drv}-base"][lid]
                                      for r in (1, 2, 3))
                c = statistics.median(rates[f"r{r}-{drv}-cand"][lid]
                                      for r in (1, 2, 3))
                gain = c > b
                perf[drv][LEG_LABEL[lid]] = {
                    "base_median_tok_s": round(b, 1),
                    "cand_median_tok_s": round(c, 1),
                    "speedup": round(c / b, 3),
                    "gain_repeats": gain,
                    "fraction_of_native_base": round(b / NATIVE_PREFILL[lid], 4),
                    "fraction_of_native_cand": round(c / NATIVE_PREFILL[lid], 4),
                }
                if not gain:
                    failures.append(
                        f"gain does not repeat: {drv} {LEG_LABEL[lid]}")

    decision = "LAND" if not failures else "NEGATIVE"
    verdict = {
        "schema": "bf16-prefill-close/1",
        "date": "2026-09-11",
        "agent": "Bf16PrefillGate",
        "decision": decision,
        "provenance": json.loads((HERE / "provenance.json").read_text()),
        "digest_gates": digest_gates,
        "suites": suite,
        "paired_prefill": perf,
        "native_baseline_tok_s": {LEG_LABEL[k]: v
                                  for k, v in NATIVE_PREFILL.items()},
        "failures": failures,
    }
    (HERE / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    print(f"decision={decision} failures={len(failures)}")
    for f in failures:
        print("FAIL:", f)
    return 0 if decision == "LAND" else 1


if __name__ == "__main__":
    sys.exit(main())
