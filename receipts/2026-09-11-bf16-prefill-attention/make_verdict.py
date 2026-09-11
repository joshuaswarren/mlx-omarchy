#!/usr/bin/env python3
"""Assemble the 2026-09-11-bf16-prefill-attention verdict from the collected
runs: paired matrix medians and fractions, digest gates per cell, the f64
oracle comparison on both wheels (base and candidate), the component
attribution tables including lm_head, and the branch decision.

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
Q4_LEGS = [
    "qwen25-0.5b-4bit:short-decode-32",
    "qwen25-0.5b-4bit:long-decode-128",
    "qwen25-0.5b-4bit:longctx-1024-decode-32",
]
ORACLE_PARTS = ("stream_match", "stream_cross", "attn_ulp", "attn_cross")


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


def expected_digest(lid, driver):
    want = LINUX_CANONICAL[lid]
    return want[driver] if isinstance(want, dict) else want


def ndjson(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main(d):
    d = Path(d)
    verdict = {"schema": "bf16-prefill-attention/2", "date": "2026-09-11",
               "agent": "Bf16AlphaFixAndVerdict"}

    # ---- gate A/B (window-1 single runs, fork driver, base wheel) ----
    ab = {}
    for label in ("attn-gate-off-fork", "attn-gate-on-fork"):
        p = d / "matrix" / label / "matrix.json"
        if p.exists():
            ab[label] = leg_view(p)
    if ab:
        verdict["gate_ab_window1"] = {
            label: {lid: {"digest": v["digest"],
                          "prefill_tok_s": v["prefill_tok_s"]}
                    for lid, v in view.items()}
            for label, view in ab.items()}

    # ---- paired matrix ----
    paired = {}
    cells = {}
    for run_dir in sorted((d / "matrix").glob("r*-*")):
        mj = run_dir / "matrix.json"
        if not mj.exists():
            continue
        label = run_dir.name
        rep, driver, kind = label.split("-", 2)
        cells.setdefault((driver, kind), {}).setdefault(rep, {})
        cells[(driver, kind)][rep] = leg_view(mj)
    for (driver, kind), reps in sorted(cells.items()):
        for lid in BF16_LEGS + Q4_LEGS:
            pre = [v[lid]["prefill_tok_s"]
                   for v in reps.values() if lid in v]
            dig = [v[lid]["digest"] for v in reps.values() if lid in v]
            key = f"{lid}|{driver}|{kind}"
            want = expected_digest(lid, driver)
            paired[key] = {
                "reps": len(pre),
                "median_prefill_tok_s": statistics.median(pre) if pre else None,
                "digest_consistent": len(set(dig)) == 1,
                "digest": dig[0] if dig else None,
                "matches_canonical": bool(dig) and all(x == want for x in dig),
                "expected_canonical": want,
                "fraction_of_native": (statistics.median(pre)
                                       / NATIVE_RATE[lid]) if pre else None,
            }
    verdict["paired_prefill"] = paired

    for lid in BF16_LEGS:
        for driver in ("fork", "stock"):
            base = paired.get(f"{lid}|{driver}|base", {})
            cand = paired.get(f"{lid}|{driver}|cand", {})
            if base.get("median_prefill_tok_s") and cand.get(
                    "median_prefill_tok_s"):
                paired[f"{lid}|{driver}|speedup"] = {
                    "base": base["median_prefill_tok_s"],
                    "cand": cand["median_prefill_tok_s"],
                    "speedup": round(cand["median_prefill_tok_s"]
                                     / base["median_prefill_tok_s"], 3),
                    "gain_repeats": (cand["median_prefill_tok_s"]
                                     > base["median_prefill_tok_s"] * 1.02),
                }
    # digest gate summary
    gates = {}
    for lid in Q4_LEGS:
        for driver in ("fork", "stock"):
            for kind in ("base", "cand"):
                row = paired.get(f"{lid}|{driver}|{kind}", {})
                if row:
                    gates[f"{lid}|{driver}|{kind}"] = row["matches_canonical"]
    for lid in BF16_LEGS:
        for driver in ("fork", "stock"):
            row = paired.get(f"{lid}|{driver}|base", {})
            if row:
                gates[f"{lid}|{driver}|base"] = row["matches_canonical"]
            cand = paired.get(f"{lid}|{driver}|cand", {})
            if cand:
                gates[f"{lid}|{driver}|cand"] = {
                    "digest": cand["digest"],
                    "consistent": cand["digest_consistent"],
                    "held_canonical": cand["matches_canonical"],
                }
    verdict["digest_gates"] = gates
    verdict["q4_canonical_held_base"] = all(
        gates.get(f"{lid}|{drv}|base", False)
        for lid in Q4_LEGS for drv in ("fork", "stock"))
    verdict["bf16_pins_held_base"] = all(
        gates.get(f"{lid}|{drv}|base", False)
        for lid in BF16_LEGS for drv in ("fork", "stock"))

    # ---- oracle on both wheels ----
    for tag in ("base", "cand"):
        rows = ndjson(d / "m1-logs" / f"oracle-{tag}.ndjson")
        rows = [r for r in rows if r.get("part") in ORACLE_PARTS]
        if rows:
            verdict[f"oracle_{tag}"] = rows
    cand_rows = verdict.get("oracle_cand", [])
    ulp = {r["m"]: r for r in cand_rows if r.get("part") == "attn_ulp"}
    decision = {"rule": "flip only if candidate bits are equal or closer "
                        "to f64 truth than the f32 composition on the "
                        "affected operations"}
    favours = None
    detail = {}
    for m in sorted({r["m"] for r in cand_rows
                     if r.get("part") == "attn_ulp"}):
        gates_m = {r["gate"]: r for r in cand_rows
                   if r.get("part") == "attn_ulp" and r["m"] == m}
        f32 = gates_m.get("f32comp")
        bf16 = gates_m.get("bf16fast")
        if not f32 or not bf16:
            continue
        closer = (bf16["mean_ulp"] <= f32["mean_ulp"]
                  and bf16["exact_frac"] >= f32["exact_frac"])
        detail[f"m{m}"] = {
            "f32comp": {k: f32[k] for k in
                        ("exact_frac", "mean_ulp", "max_ulp", "kept")},
            "bf16fast": {k: bf16[k] for k in
                         ("exact_frac", "mean_ulp", "max_ulp", "kept")},
            "candidate_closer_or_equal": closer,
        }
        favours = closer if favours is None else (favours and closer)
    for m in sorted({r["m"] for r in verdict.get("oracle_base", [])
                     if r.get("part") == "attn_ulp"}):
        gates_m = {r["gate"]: r for r in verdict["oracle_base"]
                   if r.get("part") == "attn_ulp" and r["m"] == m}
        f32 = gates_m.get("f32comp")
        bf16 = gates_m.get("bf16fast")
        if f32 and bf16:
            detail[f"m{m}_base_wheel_fallback_route"] = {
                "f32comp_mean_ulp": f32["mean_ulp"],
                "bf16fast_mean_ulp": bf16["mean_ulp"],
            }
    streams = [r for r in cand_rows if r.get("part") == "stream_cross"]
    if streams:
        detail["stream_cross_cand_wheel"] = {
            r["leg"]: {k: r[k] for k in
                       ("both_right", "both_wrong",
                        "f32_only_right", "bf16_only_right")}
            for r in streams}
        token_favours = all(
            r["bf16_only_right"] >= r["f32_only_right"] for r in streams)
        if favours is None:
            favours = token_favours
        else:
            favours = favours and token_favours
        detail["stream_tokens_candidate_closer_or_equal"] = token_favours
    decision["oracle_detail"] = detail
    decision["oracle_favours_candidate"] = favours
    decision["branch_action"] = ("owner-evidence (do not flip; write up a "
                                 "rule-2 amendment claim)" if favours
                                 else "revert-default-flip (receipt-only "
                                      "negative)")
    verdict["decision"] = decision

    # ---- attribution ----
    for cell in ("base", "cand"):
        rows = ndjson(d / "m1-logs" / f"attribution-{cell}.ndjson")
        if not rows:
            continue
        pick = {}
        for r in rows:
            k = r.get("k")
            if k in ("whole_prefill", "sdpa_whole"):
                pick.setdefault(k, []).append(
                    {kk: r[kk] for kk in
                     ("leg", "gate", "us_median", "tok_s") if kk in r})
            elif k == "lm_head_bf16":
                pick.setdefault(k, []).append(
                    {"m": r["m"], "us_median": r["us_median"],
                     "tflops": r.get("tflops")})
        verdict[f"attribution_{cell}"] = pick
        verdict[f"attribution_{cell}_rows"] = len(rows)

    out = d / "verdict.json"
    out.write_text(json.dumps(verdict, indent=1))
    print(f"wrote {out}")
    print(json.dumps({"oracle_favours_candidate": favours,
                      "branch_action": decision["branch_action"],
                      "q4_base_held": verdict["q4_canonical_held_base"],
                      "bf16_base_held": verdict["bf16_pins_held_base"]},
                     indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
