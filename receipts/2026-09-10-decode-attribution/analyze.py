#!/usr/bin/env python3
"""Assemble the decode token attribution table from arms + microbench.

Reads receipts/2026-09-10-decode-attribution/arms/legs.ndjson and
microbench.json and writes table.json + a rendered markdown table:
per-workload token-time attribution by ablation class, the skeleton
(host pacing + launch cadence) row, the unattributed residual, and the
microbench cross-check with disagreements called out.
"""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

CLASSES = ["gemv", "rope", "rms", "kvwrite", "attn", "swiglu", "sampler"]
# dispatches per token at short context (decode-bound-3 profile)
DISPATCH = {"gemv": 97, "rope": 48, "rms": 49, "kvwrite": 24,
            "attn": 24, "swiglu": 24, "sampler": 2}
FLOOR_US = 4.5


def med(vals):
    return statistics.median(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    root = args.root
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    rows = defaultdict(lambda: defaultdict(list))
    for line in (root / "arms/legs.ndjson").read_text().splitlines():
        r = json.loads(line)
        rows[r["workload"]][r["arm"]].append(r["decode_tok_s"])

    table = {}
    for workload, byarm in rows.items():
        t = {}
        for arm, tps in byarm.items():
            tok_ms = med([1000.0 / v for v in tps])
            t[arm] = {
                "n": len(tps),
                "tok_s_median": round(med(tps), 3),
                "tok_s_min": round(min(tps), 3),
                "tok_s_max": round(max(tps), 3),
                "token_ms_median": round(tok_ms, 4),
            }
        if "baseline" in t and "all" in t:
            t0 = t["baseline"]["token_ms_median"]
            s = t["all"]["token_ms_median"]
            marg = {}
            for c in CLASSES:
                if c in t:
                    marg[c] = round(t0 - t[c]["token_ms_median"], 4)
            total_marg = round(sum(marg.values()), 4)
            t["attribution"] = {
                "token_ms": round(t0, 4),
                "marginals_ms": marg,
                "skeleton_ms": round(s, 4),
                "sum_check_ms": round(total_marg + s, 4),
                "residual_ms": round(t0 - total_marg - s, 4),
                "skeleton_cadence_ms": round(273 * FLOOR_US / 1e3, 4),
                "skeleton_minus_cadence_ms": round(
                    s - 273 * FLOOR_US / 1e3, 4),
            }
        table[workload] = t

    micro = {}
    mp = root / "microbench.json"
    if mp.exists():
        m = json.loads(mp.read_text())
        for name, d in m["chains"].items():
            micro[name] = d

    result = {"schema": "decode-attribution-table/1", "table": table,
              "microbench": micro}
    (out / "table.json").write_text(json.dumps(result, indent=2,
                                               sort_keys=True) + "\n")

    lines = []
    for workload in ("short-decode-32", "long-decode-128",
                     "longctx-1024-decode-32"):
        if workload not in table:
            continue
        lines.append(f"## {workload}")
        t = table[workload]
        lines.append("| arm | n | tok/s median | token ms |")
        lines.append("|---|---|---|---|")
        for arm in ["baseline"] + CLASSES + ["all"]:
            if arm in t:
                d = t[arm]
                lines.append(f"| {arm} | {d['n']} | {d['tok_s_median']} |"
                             f" {d['token_ms_median']} |")
        if "attribution" in t:
            a = t["attribution"]
            lines.append("")
            lines.append("marginals (baseline - ablated), ms/token:")
            for c, v in a["marginals_ms"].items():
                lines.append(f"- {c}: {v}")
            lines.append(f"- skeleton (all ablated): {a['skeleton_ms']}")
            lines.append(f"- sum(marginals)+skeleton: "
                         f"{a['sum_check_ms']} vs token {a['token_ms']} "
                         f"-> residual {a['residual_ms']} ms")
            lines.append(f"- skeleton minus 273x{FLOOR_US}us cadence: "
                         f"{a['skeleton_minus_cadence_ms']} ms "
                         "(host pacing + tail kernels + bodies)")
        lines.append("")
    if micro:
        lines.append("## microbench cross-check (per token, ms)")
        lines.append("| chain | dispatches | per-dispatch us (median) |"
                     " minus floor | gpu chain ms est |")
        lines.append("|---|---|---|---|---|")
        for name, d in micro.items():
            lines.append(
                f"| {name} | {d['dispatches']} |"
                f" {d['per_dispatch_us_median']} |"
                f" {d['per_dispatch_us_minus_receipt_floor']} |"
                f" {d['gpu_chain_ms_est_minus_receipt_floor']} |")
    (out / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
