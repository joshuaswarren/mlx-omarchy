#!/usr/bin/env python3
"""Aggregate the attribution-window probe logs into the receipt table.

Reads receipts/2026-09-12-qmm-staging-attribution/logs/probe-{arm}-{round}.log
(JSON lines from qmm_coop_bench_probe.py), prints per-cell medians per arm,
the base-vs-hoist delta, the nodequant delta, and the implied clk/step at
the dominant gate_up cell under the 240-clk matrix-busy model
(GFLOP/s x clk/step = 515520, the fma-ceiling calibration constant).
"""
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1]) / "logs"
ARMS = ("base", "hoist", "nodequant")
ROUNDS = ("r1", "r2", "r3")


def load(arm, rnd):
    p = root / f"probe-{arm}-{rnd}.log"
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.startswith("{")]
    return {r["shape"]: r for r in rows}


def med(arm, shape):
    vals, digests, tf = [], set(), []
    for rnd in ROUNDS:
        r = load(arm, rnd)[shape]
        vals.append(r["median_ms"])
        digests.add(r["f16_digest"])
        tf.append(r["tflops"])
    return statistics.median(vals), digests, statistics.median(tf)


shapes = sorted(load("base", "r1").keys())
print(f"{'shape':>16} | {'base GF/s':>9} | {'hoist GF/s':>10} | {'hoist %':>7} | "
      f"{'nodeq GF/s':>10} | {'nodeq %':>7} | hoist digest ok")
print("-" * 100)
gate_up = "1053x896x9728"
for s in shapes:
    bm, bdig, btf = med("base", s)
    hm, hdig, htf = med("hoist", s)
    nm, ndig, ntf = med("nodequant", s)
    ok = hdig == bdig
    print(f"{s:>16} | {btf:9.1f} | {htf:10.1f} | {100*(htf/btf-1):+6.1f}% | "
          f"{ntf:10.1f} | {100*(ntf/btf-1):+6.1f}% | {'YES' if ok else 'NO '}"
          f" {sorted(hdig)[0][:8]}")
    if s == gate_up:
        # clock model: GFLOP/s * clk/step = 515520 (fma-ceiling calibration)
        bclk = 515520 / btf
        hclk = 515520 / htf
        nclk = 515520 / ntf
        print(f"  gate_up implied clk/step: base {bclk:.0f} -> hoist {hclk:.0f} "
              f"({bclk-hclk:+.0f}) -> nodequant {nclk:.0f} ({bclk-nclk:+.0f}); "
              f"matrix-busy floor 240, zero-traffic ceiling 359 "
              f"(=515520/1440.4)")
