#!/usr/bin/env python3
"""Gap table: native macOS Metal baselines vs the committed Linux canonical
12-matrix verdict (receipts/2026-09-10-main-parity-12-matrix/verdict.json).

The Linux fork/stock legs ran on jwm1-linux, a base Apple M1 (T8103,
8-core GPU). The native legs run on M1 Max / M1 Ultra / M2 Max hosts, so
every native:linux ratio is CROSS-CHIP and never a same-chip denominator.
"""
import json
import sys
from pathlib import Path

VERDICT = Path(sys.argv[1])
NATIVE = [Path(p) for p in sys.argv[2:]]
LEG_ORDER = ["q4_short", "q4_long", "q4_longctx",
             "bf16_short", "bf16_long", "bf16_longctx"]

v = json.loads(VERDICT.read_text())
natives = [json.loads(p.read_text()) for p in NATIVE]

rows = []
for leg in LEG_ORDER:
    fk = v["legs"]["fork:" + leg]
    st = v["legs"]["stock:" + leg]
    row = {
        "leg": leg,
        "prompt_tokens": fk["prompt_tokens"],
        "generated_tokens": fk["generated_tokens"],
        "linux_fork": {"chip": v["host"]["device"],
                       "decode_med": fk["decode_tok_s"]["median"],
                       "prefill_med": fk["prefill_tok_s"]["median"]},
        "linux_stock": {"chip": v["host"]["device"],
                        "decode_med": st["decode_tok_s"]["median"],
                        "prefill_med": st["prefill_tok_s"]["median"]},
        "native": [],
    }
    for n in natives:
        nl = n["legs"][leg]
        chip = n["host"]["chip"]
        row["native"].append({
            "host": n["host"]["hostname"].split(".")[0],
            "chip": chip,
            "decode_med": nl["decode_tok_s"]["median"],
            "decode_range": [nl["decode_tok_s"]["min"],
                             nl["decode_tok_s"]["max"]],
            "prefill_med": nl["prefill_tok_s"]["median"],
            "prefill_range": [nl["prefill_tok_s"]["min"],
                              nl["prefill_tok_s"]["max"]],
            "digest": nl["digest"],
            "digest_matches_reference_native":
                nl["digest_matches_reference_native"],
            "ratio_decode_vs_fork":
                round(nl["decode_tok_s"]["median"]
                      / row["linux_fork"]["decode_med"], 3),
            "ratio_decode_vs_stock":
                round(nl["decode_tok_s"]["median"]
                      / row["linux_stock"]["decode_med"], 3),
            "ratio_prefill_vs_fork":
                round(nl["prefill_tok_s"]["median"]
                      / row["linux_fork"]["prefill_med"], 3),
            "ratio_prefill_vs_stock":
                round(nl["prefill_tok_s"]["median"]
                      / row["linux_stock"]["prefill_med"], 3),
        })
    rows.append(row)

out = {
    "schema": "native-metal-gap-table/1",
    "linux_verdict": {
        "path": str(VERDICT),
        "commit": v["source"]["commit"],
        "host": v["host"],
        "note": "fork/stock are Linux Vulkan legs ON the base-M1 host "
                "itself; they are the same-chip Linux reference, not the "
                "macOS denominator",
    },
    "chip_equivalence": {
        "linux_host_chip": v["host"]["device"]
        + " (base M1, T8103, 8-core GPU)",
        "statement": "None of the measured macOS hosts is a base M1. "
                     "M1 Max / M1 Ultra / M2 Max have different GPU core "
                     "counts and memory bandwidth (roughly 2x-4x). A "
                     "non-base-M1 chip is NOT a valid same-chip "
                     "denominator for jwm1-linux; native ratios below are "
                     "cross-chip context, never a parity divisor. Per-chip "
                     "numbers are reported unnormalized.",
    },
    "legs": rows,
}
dest = VERDICT.parent / "gap-table.json"
dest.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
print("wrote", dest)

# Markdown rendering
md = ["# Native macOS Metal baselines vs Linux canonical 12-matrix", ""]
md += ["Linux legs: fork/stock on jwm1-linux (base M1, T8103, 8-core GPU),"
       " commit " + v["source"]["commit"] + ".", "",
       "**No macOS host measured is a base M1.** All native rows are"
       " cross-chip and are not a same-chip denominator for the Linux"
       " host. Per-chip numbers, unnormalized.", "",
       "| leg (prompt/gen) | host (chip) | decode med tok/s | prefill med"
       " tok/s | d vs fork | d vs stock | p vs fork | p vs stock | digest"
       " =ref |",
       "|---|---|---|---|---|---|---|---|---|"]
for row in rows:
    lg = (f"{row['leg'].replace('longctx', '1K ctx')} "
          f"({row['prompt_tokens']}/{row['generated_tokens']})")
    md.append(f"| {lg} | jwm1-linux fork (base M1) | "
              f"{row['linux_fork']['decode_med']} | "
              f"{row['linux_fork']['prefill_med']} | 1.00 | — | 1.00 | — "
              f"| — |")
    md.append(f"| {lg} | jwm1-linux stock (base M1) | "
              f"{row['linux_stock']['decode_med']} | "
              f"{row['linux_stock']['prefill_med']} | — | 1.00 | — | 1.00"
              f" | — |")
    for nat in row["native"]:
        ref = "yes" if nat["digest_matches_reference_native"] else "NO"
        md.append(
            f"| {lg} | {nat['host']} ({nat['chip']}) | "
            f"{nat['decode_med']} ({nat['decode_range'][0]}–"
            f"{nat['decode_range'][1]}) | "
            f"{nat['prefill_med']} ({nat['prefill_range'][0]}–"
            f"{nat['prefill_range'][1]}) | "
            f"{nat['ratio_decode_vs_fork']}x | "
            f"{nat['ratio_decode_vs_stock']}x | "
            f"{nat['ratio_prefill_vs_fork']}x | "
            f"{nat['ratio_prefill_vs_stock']}x | {ref} |")
    md.append("")
(VERDICT.parent / "gap-table.md").write_text("\n".join(md) + "\n")
print("wrote", VERDICT.parent / "gap-table.md")
