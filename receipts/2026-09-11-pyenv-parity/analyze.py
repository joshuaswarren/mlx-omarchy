#!/usr/bin/env python3
"""Aggregate PythonHostParity legs into the per-config table.

bench_matrix legs: legs/<win>/<cfg>-r<rep>-<wl>.json  (legs[].metrics)
hostphases splits: legs/<win>/split-<cfg>-r<rep>.json (flat doc)
microbenches:      legs/<win>/micro-<cfg>.json

Digests are verified against the canonical Q4 pins; any mismatch exits 4.
"""
import glob
import json
import statistics as st
import sys
import os

ROOT = os.path.expanduser("~/src/mlx-HostPathOverhead/receipts/2026-09-11-pyenv-parity")
CANON = {"short-decode-32": "7fd25a869ff21678",
         "long-decode-128": "4cc08910089477fd",
         "longctx-1024-decode-32": "7da83f06ec9f001d"}
WLS = list(CANON)


def med(v):
    return st.median(v) if v else float("nan")


def load(pattern):
    out = []
    for f in sorted(glob.glob(pattern)):
        try:
            out.append(json.load(open(f)))
        except Exception:
            pass
    return out


def main(win, baseline):
    base = os.path.join(ROOT, "legs", win)
    cfgs = sorted({os.path.basename(f).split("-r")[0]
                   for f in glob.glob(base + "/*-r1*.json")})
    cfgs = [c for c in cfgs if not c.startswith("split-")]
    digest_fail = []
    table = {}
    for cfg in cfgs:
        per_wl = {wl: [] for wl in WLS}
        wall, pyo, evms, ecalls = [], [], [], []
        for doc in load(f"{base}/{cfg}-r*.json"):
            for leg in doc.get("legs", [doc]):
                wl = leg.get("workload_id")
                m = leg.get("metrics", {})
                d = m.get("generated_ids_sha256_16")
                if wl in CANON:
                    if d and d != CANON[wl]:
                        digest_fail.append((cfg, wl, d))
                    if m.get("decode_tok_s"):
                        per_wl[wl].append((m["decode_tok_s"],
                                           m.get("decode_mean_per_token_ms")))
        for doc in load(f"{base}/split-{cfg}-r*.json"):
            if doc.get("decode_ms_per_token"):
                d = doc.get("digest")
                if d and d != CANON["short-decode-32"]:
                    digest_fail.append((cfg, "split", d))
                wall.append(doc["decode_ms_per_token"])
                cpp = doc["eval_call_ns_total"] / 1e6 / max(1, doc["n_tokens"] - 1)
                evms.append(cpp)
                pyo.append(doc["decode_ms_per_token"] - cpp)
                ecalls.append(doc["eval_calls"] / max(1, doc["n_tokens"] - 1))
        table[cfg] = {"per_wl": per_wl, "wall": wall, "py": pyo,
                      "cpp": evms, "calls": ecalls}

    print(f"window={win} baseline={baseline}")
    print(f"{'config':<16} " + "".join(f"{wl:>28}" for wl in WLS) +
          " | split(short): wall / py-out / cpp-in / eval-calls")
    for cfg, t in table.items():
        cells = []
        for wl in WLS:
            v = t["per_wl"][wl]
            if v:
                tok = med([x[0] for x in v])
                msv = [x[1] for x in v if x[1]]
                ms = med(msv) if msv else 1000.0 / tok
                cells.append(f"{tok:9.2f}t {ms:7.3f}ms n={len(v)}")
            else:
                cells.append(f"{'--':>28}")
        print(f"{cfg:<16} " + "".join(f"{c:>28}" for c in cells) +
              f" | {med(t['wall']):7.3f} {med(t['py']):7.3f} "
              f"{med(t['cpp']):7.3f} {med(t['calls']):7.1f}")

    print("\npaired medians vs baseline:")
    b = table.get(baseline)
    if b:
        for cfg, t in table.items():
            if cfg == baseline:
                continue
            for wl in WLS:
                bv = [x[0] for x in b["per_wl"][wl]]
                cv = [x[0] for x in t["per_wl"][wl]]
                if bv and cv:
                    n = min(len(bv), len(cv))
                    r = sorted(cv[i] / bv[i] for i in range(n))[n // 2]
                    print(f"  {cfg:<14} {wl:<24} x{r:.4f} ({(r-1)*100:+.1f}%)")
            if b["py"] and t["py"]:
                n = min(len(b["py"]), len(t["py"]))
                d = sorted(t["py"][i] - b["py"][i] for i in range(n))[n // 2]
                print(f"  {cfg:<14} python-out ms/token {d:+.3f} "
                      f"({med(t['py']):.3f} vs {med(b['py']):.3f})")
    for cfg, t in table.items():
        micro = load(f"{base}/micro-{cfg}.json")
        if micro:
            m = micro[0]
            print(f"micro {cfg:<14} ctor={m['ctor_ns_per_op']}ns "
                  f"eval={m['eval_ns_per_op']}ns pycall={m['pyloop_ns_per_op']}ns")
    if digest_fail:
        print("\nDIGEST FAILURES:", digest_fail)
        sys.exit(4)
    print("\ndigests: all canonical")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "cp314hpo")
