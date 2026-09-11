#!/usr/bin/env python3
"""Summarize the closing canonical-protocol 12-rep paired parity session.

Reads {fork,stock}-rep{01..12}.json (scripts/bench_matrix.py run outputs),
validates every measured leg (status, prompt tokens, generated token count,
decode span, digest) against the current per-driver pins, prints per-leg
min/median/max per driver, fractions of the committed base-M1 native macOS
baseline, and the delta versus the previous canonical matrix (b6d662a8,
receipts/2026-09-10-main-parity-12-matrix). Writes a machine-readable
verdict JSON when --verdict is given.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

# Current pins (assignment 2026-09-11): six canonical Q4 digests on both
# drivers; BF16 fork f26175202f3dabe9 / 8690dc83246b39f8, stock
# 7fc0f968789b1882 / 46108ad71157cb4d, 1K ff502900d2a179a5 on both. The
# BF16 long pins and the stock BF16 short pin moved on main with the
# native-order dense BF16 decode GEMV (re-pinned digests); the Q4 and 1K
# pins are unchanged from the b6d662a8 matrix.
CANON = {
    "fork": {
        "qwen25-0.5b-4bit:short-decode-32": "7fd25a869ff21678",
        "qwen25-0.5b-4bit:long-decode-128": "4cc08910089477fd",
        "qwen25-0.5b-4bit:longctx-1024-decode-32": "7da83f06ec9f001d",
        "qwen25-0.5b-bf16:short-decode-32": "f26175202f3dabe9",
        "qwen25-0.5b-bf16:long-decode-128": "8690dc83246b39f8",
        "qwen25-0.5b-bf16:longctx-1024-decode-32": "ff502900d2a179a5",
    },
    "stock": {
        "qwen25-0.5b-4bit:short-decode-32": "7fd25a869ff21678",
        "qwen25-0.5b-4bit:long-decode-128": "4cc08910089477fd",
        "qwen25-0.5b-4bit:longctx-1024-decode-32": "7da83f06ec9f001d",
        "qwen25-0.5b-bf16:short-decode-32": "7fc0f968789b1882",
        "qwen25-0.5b-bf16:long-decode-128": "46108ad71157cb4d",
        "qwen25-0.5b-bf16:longctx-1024-decode-32": "ff502900d2a179a5",
    },
}
NATIVE_DIGEST = {
    "qwen25-0.5b-4bit:short-decode-32": "7fd25a869ff21678",
    "qwen25-0.5b-4bit:long-decode-128": "254d73fd93164b98",
    "qwen25-0.5b-4bit:longctx-1024-decode-32": "7da83f06ec9f001d",
    "qwen25-0.5b-bf16:short-decode-32": "7fc0f968789b1882",
    "qwen25-0.5b-bf16:long-decode-128": "407b7624ed1b3b29",
    "qwen25-0.5b-bf16:longctx-1024-decode-32": "ff502900d2a179a5",
}
# Committed base-M1 native macOS baseline (medians),
# receipts/2026-09-10-native-macos-metal-baseline/committed-base-m1/
# native-baseline-baseM1.json (macOS 14.8.9, Apple M1 8 GPU cores, MLX
# 0.32.2 native Metal).
NATIVE = {
    "qwen25-0.5b-4bit:short-decode-32": (150.57, 294.1),
    "qwen25-0.5b-4bit:long-decode-128": (146.77, 1213.0),
    "qwen25-0.5b-4bit:longctx-1024-decode-32": (140.38, 1840.9),
    "qwen25-0.5b-bf16:short-decode-32": (56.43, 232.6),
    "qwen25-0.5b-bf16:long-decode-128": (55.72, 1007.7),
    "qwen25-0.5b-bf16:longctx-1024-decode-32": (54.55, 1655.7),
}
EXPECT_PROMPT = {"short-decode-32": 30, "long-decode-128": 262,
                 "longctx-1024-decode-32": 1053}
EXPECT_GEN = {"short-decode-32": 32, "long-decode-128": 128,
              "longctx-1024-decode-32": 32}
CANONICAL_KEY = {
    "qwen25-0.5b-4bit:short-decode-32": "q4_short",
    "qwen25-0.5b-4bit:long-decode-128": "q4_long",
    "qwen25-0.5b-4bit:longctx-1024-decode-32": "q4_1K",
    "qwen25-0.5b-bf16:short-decode-32": "bf16_short",
    "qwen25-0.5b-bf16:long-decode-128": "bf16_long",
    "qwen25-0.5b-bf16:longctx-1024-decode-32": "bf16_1K",
}
PREV_MED = {  # receipts/2026-09-10-main-parity-12-matrix/verdict.json legs
    "fork:qwen25-0.5b-4bit:short-decode-32": (112.195, 331.502),
    "fork:qwen25-0.5b-4bit:long-decode-128": (108.945, 970.37),
    "fork:qwen25-0.5b-4bit:longctx-1024-decode-32": (96.125, 1112.52),
    "fork:qwen25-0.5b-bf16:short-decode-32": (11.7, 85.96),
    "fork:qwen25-0.5b-bf16:long-decode-128": (11.255, 317.769),
    "fork:qwen25-0.5b-bf16:longctx-1024-decode-32": (10.03, 364.802),
    "stock:qwen25-0.5b-4bit:short-decode-32": (102.525, 170.942),
    "stock:qwen25-0.5b-4bit:long-decode-128": (81.875, 310.06),
    "stock:qwen25-0.5b-4bit:longctx-1024-decode-32": (52.91, 361.671),
    "stock:qwen25-0.5b-bf16:short-decode-32": (11.735, 83.799),
    "stock:qwen25-0.5b-bf16:long-decode-128": (11.29, 210.78),
    "stock:qwen25-0.5b-bf16:longctx-1024-decode-32": (10.125, 220.132),
}
MODELS = ("qwen25-0.5b-4bit", "qwen25-0.5b-bf16")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--verdict", default=None,
                    help="write machine-readable verdict JSON here")
    args = ap.parse_args()
    d = Path(args.dir)
    failures, out = [], {}
    per_rep = {}
    for drv in ("fork", "stock"):
        per_leg = {}
        for rep in range(1, args.reps + 1):
            f = d / f"{drv}-rep{rep:02d}.json"
            try:
                run = json.loads(f.read_text())
            except OSError as exc:
                failures.append(f"{f.name}: unreadable: {exc}")
                continue
            if run.get("measured_legs") != 6:
                failures.append(f"{f.name}: measured_legs "
                                f"{run.get('measured_legs')} != 6")
            if run.get("clean_check", {}).get("status") != "clean":
                failures.append(f"{f.name}: contention gate not clean")
            prov = (run.get("binary_provenance") or {}).get("omarchy") or {}
            if prov.get("verified") != "match":
                failures.append(f"{f.name}: provenance not match: {prov}")
            seen = set()
            for leg in run.get("legs", []):
                if leg.get("status") == "skipped":
                    continue
                lid = f"{leg['model_id']}:{leg['workload_id']}"
                seen.add(lid)
                m = leg.get("metrics") or {}
                if leg.get("status") != "measured":
                    failures.append(f"{f.name} {lid}: status "
                                    f"{leg.get('status')}: "
                                    f"{leg.get('stderr_tail','')[:160]}")
                    continue
                if m.get("prompt_tokens") != EXPECT_PROMPT[
                        leg["workload_id"]]:
                    failures.append(
                        f"{f.name} {lid}: prompt_tokens "
                        f"{m.get('prompt_tokens')} != "
                        f"{EXPECT_PROMPT[leg['workload_id']]}")
                if m.get("generated_ids_n") != EXPECT_GEN[
                        leg["workload_id"]]:
                    failures.append(
                        f"{f.name} {lid}: generated n "
                        f"{m.get('generated_ids_n')} != "
                        f"{EXPECT_GEN[leg['workload_id']]}")
                if m.get("requested_tokens") != EXPECT_GEN[
                        leg["workload_id"]]:
                    failures.append(
                        f"{f.name} {lid}: requested "
                        f"{m.get('requested_tokens')} != "
                        f"{EXPECT_GEN[leg['workload_id']]}")
                if m.get("decode_tokens") != EXPECT_GEN[
                        leg["workload_id"]] - 1:
                    failures.append(
                        f"{f.name} {lid}: decode span "
                        f"{m.get('decode_tokens')} != "
                        f"{EXPECT_GEN[leg['workload_id']] - 1}")
                dig = m.get("generated_ids_sha256_16")
                if dig != CANON[drv][lid]:
                    failures.append(f"{f.name} {lid}: digest {dig} != "
                                    f"pin {CANON[drv][lid]}")
                pline = m.get("provenance_line") or ""
                if "verified=match" not in pline or \
                        "harness=711327c" not in pline:
                    failures.append(f"{f.name} {lid}: provenance line "
                                    f"lacks verified=match harness="
                                    f"711327c: {pline[:120]}")
                per_leg.setdefault(lid, []).append({
                    "rep": rep,
                    "decode_tok_s": m.get("decode_tok_s"),
                    "prefill_tok_s": m.get("prefill_tok_s"),
                    "digest": dig,
                    "prompt_tokens": m.get("prompt_tokens"),
                    "generated_tokens": m.get("generated_ids_n"),
                    "decode_tokens": m.get("decode_tokens"),
                    "requested_tokens": m.get("requested_tokens"),
                    "duration_s": leg.get("duration_s"),
                })
            missing = ({"%s:%s" % (mo, w) for mo in MODELS for w in
                        EXPECT_PROMPT} - seen)
            if missing:
                failures.append(f"{f.name}: legs absent: {sorted(missing)}")
        out[drv] = per_leg
        per_rep[drv] = args.reps

    digests_ok = True
    print("== digests (all reps identical to current pin) ==")
    digest_comparison = {}
    for drv in ("fork", "stock"):
        for lid, vals in sorted(out[drv].items()):
            uniq = sorted({v["digest"] for v in vals})
            ok = uniq == [CANON[drv][lid]]
            digests_ok &= ok
            nat = uniq == [NATIVE_DIGEST[lid]]
            digest_comparison[f"{drv}:{lid}"] = {
                "leg_id": lid, "driver": drv,
                "canonical_key": CANONICAL_KEY[lid],
                "digest": uniq, "n_reps": len(vals),
                "digest_matches_pin": ok, "pin": CANON[drv][lid],
                "native_digest": NATIVE_DIGEST[lid],
                "digest_matches_native": nat,
            }
            print(f"  {drv:5s} {lid:42s} n={len(vals)} {uniq} "
                  f"{'OK' if ok else 'DIGEST-MISMATCH'}"
                  f"{' (=native)' if nat else ''}")

    print("\n== per-leg min / median / max (decode tok/s, prefill tok/s)"
          " ==")
    legs_out = {}
    for drv in ("fork", "stock"):
        for lid, vals in out[drv].items():
            dec = [v["decode_tok_s"] for v in vals]
            pre = [v["prefill_tok_s"] for v in vals]
            legs_out[f"{drv}:{lid}"] = {
                "leg_id": lid, "driver": drv,
                "canonical_key": CANONICAL_KEY[lid],
                "digest": CANON[drv][lid],
                "digest_matches_pin": True,
                "native_digest": NATIVE_DIGEST[lid],
                "digest_matches_native":
                    CANON[drv][lid] == NATIVE_DIGEST[lid],
                "prompt_tokens": EXPECT_PROMPT[lid.split(":")[1]],
                "generated_tokens": EXPECT_GEN[lid.split(":")[1]],
                "decode_tok_s": {"median": round(statistics.median(dec), 3),
                                 "min": min(dec), "max": max(dec)},
                "prefill_tok_s": {"median": round(statistics.median(pre), 3),
                                  "min": min(pre), "max": max(pre)},
                "reps": sorted(vals, key=lambda v: v["rep"]),
            }
    for lid in sorted({k.split(":", 1)[1] for k in legs_out}):
        fk, sk = f"fork:{lid}", f"stock:{lid}"
        if fk not in legs_out or sk not in legs_out:
            continue
        f, s = legs_out[fk], legs_out[sk]
        fd, sd = f["decode_tok_s"]["median"], s["decode_tok_s"]["median"]
        fp, sp = f["prefill_tok_s"]["median"], s["prefill_tok_s"]["median"]
        nat_d, nat_p = NATIVE[lid]
        row = {"leg_id": lid, "canonical_key": CANONICAL_KEY[lid],
               "native_baseline": {"decode_tok_s": nat_d,
                                   "prefill_tok_s": nat_p},
               "fork": {
                   "decode_frac_native": round(fd / nat_d, 3),
                   "prefill_frac_native": round(fp / nat_p, 3),
                   "delta_vs_prev_decode_pct": round(
                       (fd - PREV_MED[fk][0]) / PREV_MED[fk][0] * 100, 1),
                   "delta_vs_prev_prefill_pct": round(
                       (fp - PREV_MED[fk][1]) / PREV_MED[fk][1] * 100, 1)},
               "stock": {
                   "decode_frac_native": round(sd / nat_d, 3),
                   "prefill_frac_native": round(sp / nat_p, 3),
                   "delta_vs_prev_decode_pct": round(
                       (sd - PREV_MED[sk][0]) / PREV_MED[sk][0] * 100, 1),
                   "delta_vs_prev_prefill_pct": round(
                       (sp - PREV_MED[sk][1]) / PREV_MED[sk][1] * 100, 1)},
               "paired_fork_vs_stock_pct": {
                   "decode": round((fd - sd) / sd * 100, 1),
                   "prefill": round((fp - sp) / sp * 100, 1)}}
        legs_out[f"{lid}:summary"] = row
        print(f"{lid}\n"
              f"  fork  decode {min(v['decode_tok_s'] for v in out['fork'][lid]):.2f}/"
              f"{fd:.2f}/{max(v['decode_tok_s'] for v in out['fork'][lid]):.2f}"
              f"  prefill {min(v['prefill_tok_s'] for v in out['fork'][lid]):.1f}/"
              f"{fp:.1f}/{max(v['prefill_tok_s'] for v in out['fork'][lid]):.1f}"
              f"  native {fd / nat_d:.3f}x/{fp / nat_p:.3f}x"
              f"  vs-prev {(fd - PREV_MED[fk][0]) / PREV_MED[fk][0] * 100:+.1f}%/"
              f"{(fp - PREV_MED[fk][1]) / PREV_MED[fk][1] * 100:+.1f}%\n"
              f"  stock decode {min(v['decode_tok_s'] for v in out['stock'][lid]):.2f}/"
              f"{sd:.2f}/{max(v['decode_tok_s'] for v in out['stock'][lid]):.2f}"
              f"  prefill {min(v['prefill_tok_s'] for v in out['stock'][lid]):.1f}/"
              f"{sp:.1f}/{max(v['prefill_tok_s'] for v in out['stock'][lid]):.1f}"
              f"  native {sd / nat_d:.3f}x/{sp / nat_p:.3f}x"
              f"  vs-prev {(sd - PREV_MED[sk][0]) / PREV_MED[sk][0] * 100:+.1f}%/"
              f"{(sp - PREV_MED[sk][1]) / PREV_MED[sk][1] * 100:+.1f}%\n"
              f"  paired fork-vs-stock {(fd - sd) / sd * 100:+.1f}% decode "
              f"{(fp - sp) / sp * 100:+.1f}% prefill")

    print(f"\n== validation: {len(failures)} problem(s) ==")
    for f in failures:
        print("FAIL:", f)
    if failures or not digests_ok:
        print("VERDICT: FAIL")
        rc = 3
    else:
        n = args.reps * 12
        print(f"ALL_OK: {args.reps} reps x 2 drivers x 6 legs = {n} legs, "
              "every digest equals its current pin and every token count "
              "matches")
        print("VERDICT: PASS")
        rc = 0
    if args.verdict:
        Path(args.verdict).write_text(json.dumps({
            "digest_comparison": digest_comparison,
            "legs": legs_out,
            "failures": failures,
            "reps_per_driver": per_rep,
        }, indent=2) + "\n")
    sys.exit(rc)


if __name__ == "__main__":
    main()
