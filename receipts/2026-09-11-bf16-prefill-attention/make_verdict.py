#!/usr/bin/env python3
"""Assemble the 2026-09-11-bf16-prefill-attention verdict from the collected
runs: paired matrix medians and fractions, digest gates per cell, the f64
oracle comparison, and the component attribution table.

Usage: make_verdict.py RECEIPT_DIR
"""

import json
import statistics
import sys
from pathlib import Path

NATIVE_RATE = {
    "qwen25-0.5b-bf16:short-decode-32": 232.60,
    "qwen25-0.5b-bf16:long-decode-128": 1007.70,
    "qwen25-0.5b-bf16:longctx-1024-decode-32": 1655.70,
    "qwen25-0.5b-4bit:short-decode-32": 294.10,
    "qwen25-0.5b-4bit:long-decode-128": 1213.00,
    "qwen25-0.5b-4bit:longctx-1024-decode-32": 1840.90,
}
LINUX_CANONICAL = {
    "qwen25-0.5b-4bit:short-decode-32": "7fd25a869ff21678",
    "qwen25-0.5b-4bit:long-decode-128": "4cc08910089477fd",
    "qwen25-0.5b-4bit:longctx-1024-decode-32": "7da83f06ec9f001d",
    "qwen25-0.5b-bf16:short-decode-32": {"fork": "f26175202f3dabe9",
                                          "stock": "7fc0f968789b1882"},
    "qwen25-0.5b-bf16:long-decode-128": {"fork": "8690dc83246b39f8",
                                          "stock": "46108ad71157cb4d"},
    "qwen25-0.5b-bf16:longctx-1024-decode-32": "ff502900d2a179a5",
}
BF16_LEGS = [
    "qwen25-0.5b-bf16:short-decode-32",
    "qwen25-0.5b-bf16:long-decode-128",
    "qwen25-0.5b-bf16:longctx-1024-decode-32",
]


def leg_view(matrix_json):
    run = json.loads(Path(matrix_json).read_text())
    out = {}
    for leg in run["legs"]:
        if leg["status"] != "measured":
            continue
        m = leg["metrics"]
        out[leg["leg_id"]] = {
            "digest": m["generated_ids_sha256_16"],
            "prefill_tok_s": m.get("prefill_tok_s"),
            "decode_tok_s": m.get("decode_tok_s"),
            "clean": run.get("clean_check", {}).get("status"),
        }
    return out


def main(d):
    d = Path(d)
    verdict = {"schema": "bf16-prefill-attention/1", "date": "2026-09-11",
               "agent": "Bf16PrefillAttention"}

    # ---- gate A/B (single runs, fork driver, base wheel) ----
    ab = {}
    for label in ("attn-gate-off-fork", "attn-gate-on-fork"):
        p = d / "matrix" / label / "matrix.json"
        if p.exists():
            ab[label] = leg_view(p)
    if ab:
        verdict["gate_ab"] = {
            label: {lid: {"digest": v["digest"],
                          "prefill_tok_s": v["prefill_tok_s"]}
                    for lid, v in view.items()}
            for label, view in ab.items()}

    # ---- paired matrix ----
    paired = {}
    matrix_dir = d / "matrix"
    cells = {}
    for run_dir in sorted(matrix_dir.glob("r*-*")):
        mj = run_dir / "matrix.json"
        if not mj.exists():
            continue
        label = run_dir.name
        driver = label.rsplit("-", 1)[-1]
        kind = label.rsplit("-", 2)[-2]
        rep = label.split("-")[0]
        cells.setdefault((driver, kind), {}).setdefault(rep, {})
        cells[(driver, kind)][rep] = leg_view(mj)
    for (driver, kind), reps in sorted(cells.items()):
        for lid in BF16_LEGS:
            pre = [v[lid]["prefill_tok_s"]
                   for v in reps.values() if lid in v]
            dig = [v[lid]["digest"] for v in reps.values() if lid in v]
            key = f"{lid}|{driver}|{kind}"
            paired[key] = {
                "reps": len(pre),
                "median_prefill_tok_s": statistics.median(pre) if pre else None,
                "digest_consistent": len(set(dig)) == 1,
                "digest": dig[0] if dig else None,
                "fraction_of_native": (statistics.median(pre)
                                       / NATIVE_RATE[lid]) if pre else None,
            }
    verdict["paired_prefill"] = paired

    # speedups
    for lid in BF16_LEGS:
        for driver in ("fork", "stock"):
            base = paired.get(f"{lid}|{driver}|base", {})
            cand = paired.get(f"{lid}|{driver}|cand", {})
            if base.get("median_prefill_tok_s") and cand.get(
                    "median_prefill_tok_s"):
                paired.setdefault(f"{lid}|{driver}|speedup", {}) = {
                    "base": base["median_prefill_tok_s"],
                    "cand": cand["median_prefill_tok_s"],
                    "speedup": round(cand["median_prefill_tok_s"]
                                     / base["median_prefill_tok_s"], 3),
                    "gain_repeats": (cand["median_prefill_tok_s"]
                                     > base["median_prefill_tok_s"] * 1.02),
                }

    # ---- oracle ----
    oracle_p = d / "m1-logs" / "oracle.ndjson"
    if oracle_p.exists():
        orows = [json.loads(l) for l in oracle_p.read_text().splitlines()]
        verdict["oracle"] = [r for r in orows if r.get("part") in (
            "stream_match", "stream_cross", "attn_ulp", "attn_cross")]

    # ---- attribution ----
    for cell in ("base", "cand"):
        ap = d / "m1-logs" / f"attribution-{cell}.ndjson"
        if ap.exists():
            rows = [json.loads(l) for l in ap.read_text().splitlines()]
            verdict[f"attribution_{cell}_rows"] = len(rows)

    out = d / "verdict.json"
    out.write_text(json.dumps(verdict, indent=1))
    print(f"wrote {out}")
    print(json.dumps(verdict.get("gate_ab", {}), indent=1)[:2000])


if __name__ == "__main__":
    main(sys.argv[1])
