#!/usr/bin/env python3
"""Summarize the qmm schedule screen + attribution probes into a verdict table.

Usage: summarize_screen.py OUTDIR  (the benchq/q4prefill-window1 dir)
Baseline arm digests are checked against the committed probe values in
receipts/2026-09-10-qmm-splitk-parity/probe-f16-scales.json (f16 scales,
same shapes, same seeds).
"""
import json
import re
import sys
from pathlib import Path

COMMITTED = {  # shape -> f16 digest, wheel 1323dc80 default arm
    "1053x896x128": "bffbe02190af",
    "1053x896x896": "34d26ab6ae96",
    "1053x896x9728": "432aac265a26",
    "1053x4864x896": "f4e16a2ffc45",
}


def parse_arm(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.startswith("SUMMARY "):
            return json.loads(line[len("SUMMARY "):])["qmm"]
        if line.startswith("{"):
            rows.append(json.loads(line))
    return rows


def main():
    outdir = Path(sys.argv[1])
    arms = {}
    for arm in range(4):
        rows = parse_arm(outdir / f"probe-qmm-arm{arm}.log")
        arms[arm] = {(r["shape"].split("x")[0], r["shape"]): r for r in rows
                     if r["shape"].count("x") == 2}
    names = {0: "base32x16", 1: "m64", 2: "k32", 3: "m64+k32"}
    digest_ok = {}
    for arm, table in arms.items():
        ok = True
        for _, row in table.items():
            shape = row["shape"]
            ref = COMMITTED.get(shape)
            if arm == 0 and ref is not None:
                if row["f16_digest"] != ref:
                    ok = False
                    print(f"[DIGEST] arm0 {shape} {row['f16_digest']} != "
                          f"committed {ref}")
            elif arm != 0:
                base = arms[0].get((shape.split("x")[0], shape))
                if base and row["f16_digest"] != base["f16_digest"]:
                    ok = False
                    print(f"[DIGEST] arm{arm} {shape} differs from arm0")
        digest_ok[arm] = ok
        print(f"[DIGEST] arm{arm} ({names[arm]}) digest-clean: {ok}")

    print("\n[TIME] medians (ms) and speed vs base arm:")
    shapes = [k for k, v in sorted(arms[0].items())]
    print(f"{'shape':>16} " + " ".join(f"{names[a]:>14}" for a in arms))
    for key in shapes:
        line = f"{key[1]:>16} "
        for arm in arms:
            row = arms[arm].get(key)
            line += f"{row['median_ms']:>14.4f} " if row else f"{'--':>14} "
        print(line)
        print(f"{'  gflops':>16} " + " ".join(
            f"{arms[a][key]['gflops']:>14.1f}" for a in arms))
    print("\n[DECISION INPUTS] speed vs arm0 on the two big shapes:")
    for key in shapes:
        if key not in arms[0]:
            continue
        base = arms[0][key]["median_ms"]
        cells = []
        for arm in (1, 2, 3):
            row = arms[arm].get(key)
            if row:
                cells.append(f"{names[arm]}:{base / row['median_ms']:.3f}")
        print(f"  {key[1]:>16} " + "  ".join(cells))

    # attribution
    attrib_path = outdir / "probe-attrib.log"
    if attrib_path.exists():
        rows = [json.loads(l[len("SUMMARY "):]) if l.startswith("SUMMARY ")
                else json.loads(l)
                for l in attrib_path.read_text().splitlines()
                if l.startswith("{") or l.startswith("SUMMARY ")]
        merged = {}
        for r in rows:
            merged.update({k: v for k, v in r.items()
                           if k in ("attn", "norms", "lmhead")})
        print("\n[ATTRIBUTION] per-op isolated medians at m=1053:")
        for section in ("attn", "norms", "lmhead"):
            for r in merged.get(section, []):
                print(f"  {section:>7} {r['shape']:>16} "
                      f"{r['median_ms']:>10.4f} ms")


if __name__ == "__main__":
    main()
