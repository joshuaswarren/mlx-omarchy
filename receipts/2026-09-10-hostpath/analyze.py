#!/usr/bin/env python3
"""Analyze HostPathOverhead legs: host share, replay delta, bit-identity.

Reads the legs/ directory produced by window.sh:
  a{rep}-baseline-{wl}.json   bench_matrix legs on the instrumented wheel
  c{rep}-replay-{wl}.json     bench_matrix legs with MLX_OMARCHY_REPLAY=1
  b{rep}-trace.json           hostphases legs (trace on)
  d{rep}-replaytrace.json     hostphases legs with replay
  e{rep}-tiny.json            tiny-model host floor
  e1rel-tiny.json             tiny on the release wheel

Prints:
  1. throughput medians + digest identity per arm
  2. host-phase decomposition per token (instrumented + tiny + replay)
  3. the host-vs-GPU split by three independent methods
"""
import glob
import json
import statistics
import sys
from pathlib import Path

PIN = "a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3"
CANONICAL = {
    "short-decode-32": "7fd25a869ff21678",
    "long-decode-128": "4cc08910089477fd",
    "longctx-1024-decode-32": "7da83f06ec9f001d",
}
CANONICAL_TPS = {
    "short-decode-32": 112.53,
    "long-decode-128": 109.23,
    "longctx-1024-decode-32": 97.30,
}

# host-busy phases (C++ side, exclusive). join_wait is GPU-paced;
# completion_work runs on the device thread (reported separately);
# alloc/free are subsets of backend_eval (reported separately).
BUSY = ["backend_eval", "eval_bookkeeping", "disp_total", "desc_setup",
        "barriers", "vkcmd", "ensure_recording", "submit_total",
        "finalize_ns"]
INFO = ["join_wait", "completion_work", "alloc", "free", "replay_hits",
        "replay_records"]


def load(pattern):
    out = []
    for p in sorted(glob.glob(pattern)):
        try:
            out.append(json.loads(Path(p).read_text()))
        except Exception as e:
            print(f"  WARN: {p}: {e}", file=sys.stderr)
    return out


def bench_legs(out, pattern, arm):
    rows = []
    for d in load(pattern):
        for leg in d.get("legs", []):
            if leg.get("status") != "measured":
                continue
            m = leg["metrics"]
            rows.append({
                "arm": arm,
                "workload": leg["workload_id"],
                "tok_s": m["decode_tok_s"],
                "digest": m.get("generated_ids_sha256_16"),
            })
    return rows


def phase_table(traces, title):
    print(f"\n-- {title}")
    keys = BUSY + INFO
    agg = {k: {"ns": [], "hits": []} for k in keys}
    ntok = []
    for t in traces:
        tr = t.get("trace") or {}
        if not tr:
            continue
        ntok.append(t["n_tokens"])
        for k in keys:
            agg[k]["ns"].append(tr.get(k, {}).get("ns", 0))
            agg[k]["hits"].append(tr.get(k, {}).get("hits", 0))
    if not ntok:
        print("  no traces")
        return
    reps = len(ntok)
    print(f"  legs={reps} tokens_per_leg={statistics.median(ntok):.0f}")
    for k in keys:
        if k in INFO and not any(agg[k]["hits"]):
            continue
        ns = statistics.median(agg[k]["ns"]) / 1e6
        hits = statistics.median(agg[k]["hits"])
        print(f"  {k:18s} {ns:9.3f} ms/token  hits {hits:8.1f}")
    busy = statistics.median([
        sum(t["trace"].get(k, {"ns": 0})["ns"] for k in BUSY) / 1e6
        for t in traces if t.get("trace")
    ])
    evals = statistics.median([t["eval_call_ns_total"] / 1e6 for t in traces])
    wall = statistics.median([t["decode_ms_per_token"] for t in traces])
    print(f"  {'SUM busy(BUSY)':18s} {busy:9.3f} ms/token")
    print(f"  {'python_only':18s} {wall - evals:9.3f} ms/token"
          f"   (wall {wall:.3f} - eval calls {evals:.3f})")
    print(f"  {'host_busy_est':18s} {busy + wall - evals:9.3f} ms/token"
          f"   (cpp busy + python_only)")


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else
               Path(__file__).parent / "legs")
    rows = []
    rows += bench_legs(out, str(out / "a*-baseline-*.json"), "baseline")
    rows += bench_legs(out, str(out / "c*-replay-*.json"), "replay")

    print("== Throughput (decode tok/s medians) and digest identity")
    for arm in ["baseline", "replay"]:
        for wl in CANONICAL:
            xs = [r["tok_s"] for r in rows
                  if r["arm"] == arm and r["workload"] == wl]
            ds = {r["digest"] for r in rows
                  if r["arm"] == arm and r["workload"] == wl}
            if not xs:
                continue
            med = statistics.median(xs)
            ident = "OK" if ds == {CANONICAL[wl]} else f"BAD {ds}"
            delta = 100 * (med - CANONICAL_TPS[wl]) / CANONICAL_TPS[wl]
            print(f"  {arm:9s} {wl:24s} n={len(xs)} {med:7.2f} tok/s"
                  f"  ({delta:+.1f}% vs canonical)  digest {ident}")

    base = [r for r in rows if r["arm"] == "baseline"
            and r["workload"] == "short-decode-32"]
    rep = [r for r in rows if r["arm"] == "replay"
           and r["workload"] == "short-decode-32"]
    if base and rep:
        mb = statistics.median([r["tok_s"] for r in base])
        mr = statistics.median([r["tok_s"] for r in rep])
        print(f"\n== Replay prototype delta (short-decode-32)"
              f"  baseline {1000/mb:.3f} ms/token -> replay {1000/mr:.3f}"
              f" ms/token  = {(mb - mr) / mb * 100:+.1f}%")

    phase_table(load(str(out / "b*-trace.json")), "Eager host phases (B)")
    phase_table(load(str(out / "d*-replaytrace.json")),
                "Replay host phases (D)")
    phase_table(load(str(out / "e[0-9]-tiny.json")), "Tiny model (E)")

    print("\n== Tiny-model host floor (method B)")
    for name, pat in [("tiny@instrumented", "e[0-9]-tiny.json"),
                      ("tiny@release", "e1rel-tiny.json")]:
        xs = load(str(out / pat))
        if xs:
            ms = statistics.median([x["decode_ms_per_token"] for x in xs])
            print(f"  {name:18s} {ms:.3f} ms/token (n={len(xs)})")


if __name__ == "__main__":
    main()
