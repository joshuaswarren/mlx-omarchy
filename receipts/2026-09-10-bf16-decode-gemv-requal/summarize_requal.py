#!/usr/bin/env python3
"""Summarize the requal matrix: {r{rep}-{fork,stock}-{base,cand}}/matrix.json.

Validates per-leg agreement (status, prompt tokens, generated count, decode
span, clean gate, AC), asserts digest uniformity within each cell, classifies
every digest (canonical-Linux / native / other), computes paired decode and
prefill medians per cell and their fraction of the committed base-M1 native
baseline (ea09eefa: native-baseline-2026-09-06).
"""
import json
import statistics
import sys
from pathlib import Path

LEG_IDS = [
    "qwen25-0.5b-4bit:short-decode-32",
    "qwen25-0.5b-4bit:long-decode-128",
    "qwen25-0.5b-4bit:longctx-1024-decode-32",
    "qwen25-0.5b-bf16:short-decode-32",
    "qwen25-0.5b-bf16:long-decode-128",
    "qwen25-0.5b-bf16:longctx-1024-decode-32",
]
SHORT = {l.split(":")[1].split("-decode")[0]: l for l in LEG_IDS}
EXPECT_PROMPT = {"short-decode-32": 30, "long-decode-128": 262,
                 "longctx-1024-decode-32": 1053}
EXPECT_GEN = {"short-decode-32": 32, "long-decode-128": 128,
              "longctx-1024-decode-32": 32}
LINUX_CANONICAL = {
    "qwen25-0.5b-4bit:short-decode-32": "7fd25a869ff21678",
    "qwen25-0.5b-4bit:long-decode-128": "4cc08910089477fd",
    "qwen25-0.5b-4bit:longctx-1024-decode-32": "7da83f06ec9f001d",
    "qwen25-0.5b-bf16:short-decode-32": "f26175202f3dabe9",
    "qwen25-0.5b-bf16:long-fork-canonical": "ad964232ee67fecd",
    "qwen25-0.5b-bf16:long-stock-canonical": "c5be9207833d2a26",
    "qwen25-0.5b-bf16:longctx-1024-decode-32": "ff502900d2a179a5",
}
NATIVE = {
    "qwen25-0.5b-4bit:short-decode-32": "7fd25a869ff21678",
    "qwen25-0.5b-4bit:long-decode-128": "254d73fd93164b98",
    "qwen25-0.5b-4bit:longctx-1024-decode-32": "7da83f06ec9f001d",
    "qwen25-0.5b-bf16:short-decode-32": "7fc0f968789b1882",
    "qwen25-0.5b-bf16:long-decode-128": "407b7624ed1b3b29",
    "qwen25-0.5b-bf16:longctx-1024-decode-32": "ff502900d2a179a5",
}
# ea09eefa committed base-M1 native denominator (decode tok/s, prefill tok/s)
NATIVE_RATE = {
    "qwen25-0.5b-4bit:short-decode-32": (150.57, 294.10),
    "qwen25-0.5b-4bit:long-decode-128": (146.77, 1213.00),
    "qwen25-0.5b-4bit:longctx-1024-decode-32": (140.38, 1840.90),
    "qwen25-0.5b-bf16:short-decode-32": (56.43, 232.60),
    "qwen25-0.5b-bf16:long-decode-128": (55.72, 1007.70),
    "qwen25-0.5b-bf16:longctx-1024-decode-32": (54.55, 1655.70),
}


def classify(lid, dig):
    if dig == NATIVE[lid]:
        return "NATIVE"
    if lid == "qwen25-0.5b-bf16:long-decode-128":
        if dig == "ad964232ee67fecd":
            return "linux-canonical-fork"
        if dig == "c5be9207833d2a26":
            return "linux-canonical-stock"
        if dig == "8690dc83246b39f8":
            return "FORBIDDEN-8690dc(prior-candidate)"
        return "OTHER"
    if dig == LINUX_CANONICAL.get(lid):
        return "linux-canonical"
    return "OTHER"


def main(d):
    d = Path(d)
    failures = []
    cells = {}
    for run_dir in sorted(d.iterdir()):
        mj = run_dir / "matrix.json"
        if not mj.exists():
            continue
        run = json.loads(mj.read_text())
        label = run_dir.name
        if label == "warmup":
            continue
        if run.get("clean_check", {}).get("status") != "clean":
            failures.append(f"{label}: contended")
        if run.get("power", {}).get("source") != "AC":
            failures.append(f"{label}: not on AC")
        cell = cells.setdefault(label, {})
        for leg in run["legs"]:
            lid = leg["leg_id"]
            if leg["status"] != "measured":
                failures.append(f"{label} {lid}: status {leg['status']}")
                continue
            wl = leg["workload_id"]
            m = leg["metrics"]
            if leg["prompt_tokens"] != EXPECT_PROMPT[wl]:
                failures.append(f"{label} {lid}: prompt_tokens {leg['prompt_tokens']}")
            if m["generated_ids_n"] != EXPECT_GEN[wl]:
                failures.append(f"{label} {lid}: gen n {m['generated_ids_n']}")
            if m["decode_tokens"] != EXPECT_GEN[wl] - 1:
                failures.append(f"{label} {lid}: decode span {m['decode_tokens']}")
            cell.setdefault(lid, []).append(
                (m["decode_tok_s"], m["prefill_tok_s"], m["generated_ids_sha256_16"]))

    print("== digest classification per cell ==")
    digest_table = {}
    for label in sorted(cells):
        for lid, vals in sorted(cells[label].items()):
            digs = {v[2] for v in vals}
            if len(digs) > 1:
                failures.append(f"{label} {lid}: digest varied across reps {digs}")
            dig = sorted(digs)[0]
            cls = classify(lid, dig)
            digest_table[(label, lid)] = (dig, cls)
            print(f"  {label:18s} {lid:42s} {dig} {cls}")

    print("\n== paired medians and fraction of committed base-M1 native ==")
    print("  cell             leg                                  dec_med  x-native   pre_med  x-native")
    summary = {}
    for label in sorted(cells):
        for lid in LEG_IDS:
            vals = cells[label].get(lid)
            if not vals:
                failures.append(f"{label}: missing leg {lid}")
                continue
            dec = statistics.median(v[0] for v in vals)
            pre = statistics.median(v[1] for v in vals)
            nd, np_ = NATIVE_RATE[lid]
            summary.setdefault(label, {})[lid] = {
                "decode_median": round(dec, 3), "prefill_median": round(pre, 3),
                "fraction_of_native_decode": round(dec / nd, 4),
                "fraction_of_native_prefill": round(pre / np_, 4),
                "reps": len(vals),
            }
            print(f"  {label:18s} {lid:42s} {dec:8.2f} {dec/nd:9.3f} {pre:8.1f} {pre/np_:9.3f}")

    out = d / "summary.json"
    out.write_text(json.dumps(
        {"digests": {f"{k[0]}|{k[1]}": v for k, v in digest_table.items()},
         "rates": summary, "failures": failures}, indent=2) + "\n")
    print(f"\n== validation: {len(failures)} problem(s) -> {out}")
    for f in failures:
        print("FAIL:", f)


if __name__ == "__main__":
    main(sys.argv[1])
