#!/usr/bin/env python3
"""Summarize the canonical-protocol 12-rep paired parity session.

Reads {fork,stock}-rep{01..12}.json (scripts/bench_matrix.py run
outputs), validates every measured leg (status, prompt tokens, generated
token count, decode span, digest) against the canonical per-driver
digests, and prints per-leg min/median/max per driver plus paired deltas.
"""
import json
import statistics
import sys
from pathlib import Path

CANON = {
    "fork": {
        "qwen25-0.5b-4bit:short-decode-32": "7fd25a869ff21678",
        "qwen25-0.5b-4bit:long-decode-128": "4cc08910089477fd",
        "qwen25-0.5b-4bit:longctx-1024-decode-32": "7da83f06ec9f001d",
        "qwen25-0.5b-bf16:short-decode-32": "f26175202f3dabe9",
        "qwen25-0.5b-bf16:long-decode-128": "ad964232ee67fecd",
        "qwen25-0.5b-bf16:longctx-1024-decode-32": "ff502900d2a179a5",
    },
    "stock": {
        "qwen25-0.5b-4bit:short-decode-32": "7fd25a869ff21678",
        "qwen25-0.5b-4bit:long-decode-128": "4cc08910089477fd",
        "qwen25-0.5b-4bit:longctx-1024-decode-32": "7da83f06ec9f001d",
        "qwen25-0.5b-bf16:short-decode-32": "f26175202f3dabe9",
        "qwen25-0.5b-bf16:long-decode-128": "c5be9207833d2a26",
        "qwen25-0.5b-bf16:longctx-1024-decode-32": "ff502900d2a179a5",
    },
}
NATIVE = {
    "qwen25-0.5b-4bit:short-decode-32": "7fd25a869ff21678",
    "qwen25-0.5b-4bit:long-decode-128": "254d73fd93164b98",
    "qwen25-0.5b-4bit:longctx-1024-decode-32": "7da83f06ec9f001d",
    "qwen25-0.5b-bf16:short-decode-32": "7fc0f968789b1882",
    "qwen25-0.5b-bf16:long-decode-128": "407b7624ed1b3b29",
    "qwen25-0.5b-bf16:longctx-1024-decode-32": "ff502900d2a179a5",
}
EXPECT_PROMPT = {"short-decode-32": 30, "long-decode-128": 262,
                 "longctx-1024-decode-32": 1053}
EXPECT_GEN = {"short-decode-32": 32, "long-decode-128": 128,
              "longctx-1024-decode-32": 32}
MODELS = ("qwen25-0.5b-4bit", "qwen25-0.5b-bf16")


def main(d, reps=12):
    d = Path(d)
    failures, out = [], {}
    for drv in ("fork", "stock"):
        per_leg = {}
        for rep in range(1, reps + 1):
            run = json.loads((d / f"{drv}-rep{rep:02d}.json").read_text())
            for leg in run["legs"]:
                lid = leg["leg_id"]
                if leg["model_id"] not in MODELS:
                    continue
                wl = leg["workload_id"]
                if leg["status"] != "measured":
                    failures.append(f"{drv} rep{rep} {lid}: status "
                                    f"{leg['status']} "
                                    f"{leg.get('stderr_tail','')}")
                    continue
                m = leg["metrics"]
                if leg["prompt_tokens"] != EXPECT_PROMPT[wl]:
                    failures.append(f"{drv} rep{rep} {lid}: prompt_tokens "
                                    f"{leg['prompt_tokens']} != "
                                    f"{EXPECT_PROMPT[wl]}")
                if m["generated_ids_n"] != EXPECT_GEN[wl]:
                    failures.append(f"{drv} rep{rep} {lid}: gen n "
                                    f"{m['generated_ids_n']} != "
                                    f"{EXPECT_GEN[wl]}")
                if m["decode_tokens"] != EXPECT_GEN[wl] - 1:
                    failures.append(f"{drv} rep{rep} {lid}: decode span "
                                    f"{m['decode_tokens']}")
                dig = m["generated_ids_sha256_16"]
                canon = CANON[drv][lid]
                if dig != canon:
                    failures.append(f"{drv} rep{rep} {lid}: digest {dig} != "
                                    f"canonical {canon} (native "
                                    f"{NATIVE[lid]})")
                if run["clean_check"]["status"] != "clean":
                    failures.append(f"{drv} rep{rep}: contended")
                if run["power"]["source"] != "AC":
                    failures.append(f"{drv} rep{rep}: not on AC")
                per_leg.setdefault(lid, []).append(
                    (m["decode_tok_s"], m["prefill_tok_s"], dig))
        out[drv] = per_leg

    digests_ok = True
    print("== digests (all reps identical to canonical) ==")
    for drv in ("fork", "stock"):
        for lid, vals in sorted(out[drv].items()):
            digs = {v[2] for v in vals}
            ok = len(digs) == 1 and digs == {CANON[drv][lid]}
            digests_ok &= ok
            print(f"{drv:5s} {lid:45s} n={len(vals):2d} {sorted(digs)} "
                  f"{'OK' if ok else 'DIGEST-MISMATCH'}")
    print("\n== per-leg min / median / max (decode tok/s, prefill tok/s) ==")
    rows = {}
    for drv in ("fork", "stock"):
        for lid, vals in out[drv].items():
            dec = [v[0] for v in vals]
            pre = [v[1] for v in vals]
            rows[lid] = rows.get(lid, {}) | {
                drv: (min(dec), statistics.median(dec), max(dec),
                      min(pre), statistics.median(pre), max(pre))}
    for lid in sorted(rows):
        f = rows[lid]["fork"]
        s = rows[lid]["stock"]
        print(f"{lid}\n  fork  decode {f[0]:.2f}/{f[1]:.2f}/{f[2]:.2f}  "
              f"prefill {f[3]:.1f}/{f[4]:.1f}/{f[5]:.1f}\n"
              f"  stock decode {s[0]:.2f}/{s[1]:.2f}/{s[2]:.2f}  "
              f"prefill {s[3]:.1f}/{s[4]:.1f}/{s[5]:.1f}\n"
              f"  paired delta (median fork vs stock): decode "
              f"{(f[1]-s[1])/s[1]*100:+.1f}%  prefill "
              f"{(f[4]-s[4])/s[4]*100:+.1f}%")
    print(f"\n== validation: {len(failures)} problem(s) ==")
    for f in failures:
        print("FAIL:", f)
    if failures or not digests_ok:
        sys.exit(3)
    print(f"ALL_OK: {reps} reps x 2 drivers x 6 legs = {reps*12} legs, "
          "every digest equals the canonical per-driver value and every "
          "token count matches")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 12)
