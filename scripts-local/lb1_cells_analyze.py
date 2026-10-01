#!/usr/bin/env python3
"""Jw16LevelBatch W2 gate-A analyzer: paired cells, digests, logits gates.

Reads /var/tmp/lb1/cells-{ctl,lb}-r{1..5}/ on jw16. Prints per-cell digest
verdicts vs the production pins, paired decode rates with the gain and the
control min-max band, pf512 latencies, and logits-gate outcomes.
"""
import json
import statistics
import sys

OUT = "/var/tmp/lb1"
ROUNDS = [1, 2, 3, 4, 5]
CELLS = ["d64", "d128", "d256", "d512"]
PINS = {
    "d64": "c84b3e7a",
    "d128": "07c515e0",
    "d256": "c6aabbf0",
    "d512": "5c120987",
}
PF512_RECORD = "100a61b62470"


def load(arm, rnd, cell):
    path = f"{OUT}/cells-{arm}-r{rnd}/qwen-gpu-{cell}-n5.json"
    try:
        return json.load(open(path))
    except Exception as e:
        print(f"LOAD-FAIL {path}: {e}")
        return None


def rates(arm, cell):
    out = []
    for r in ROUNDS:
        d = load(arm, r, cell)
        if d:
            out.append(d["decode_tok_rate"]["median"])
    return out


def digests(arm, cell):
    vals = set()
    for r in ROUNDS:
        d = load(arm, r, cell)
        if d:
            vals.add(d["ordered_records_sha256"][:8])
    return vals


print("== decode digests (must equal pins in BOTH arms)")
ok = True
for cell in CELLS:
    dv, dl = digests("ctl", cell), digests("lb", cell)
    pin = PINS[cell]
    good = dv == {pin} and dl == {pin}
    ok &= good
    print(f"  {cell}: ctl={sorted(dv)} lb={sorted(dl)} pin={pin} {'OK' if good else 'FAIL'}")

print("== decode rates (tok/s median of 5-pass cells; 5 rounds)")
med = {}
for cell in CELLS:
    rc, rl = rates("ctl", cell), rates("lb", cell)
    if not rc or not rl:
        print(f"  {cell}: MISSING ctl={rc} lb={rl}")
        ok = False
        continue
    mc, ml = statistics.median(rc), statistics.median(rl)
    gain = (ml - mc) / mc * 100.0
    lo, hi = min(rc), max(rc)
    outside = ml > hi
    med[cell] = (mc, ml, gain, lo, hi)
    print(
        f"  {cell}: ctl={rc} med={mc} | lb={rl} med={ml} | "
        f"gain={gain:+.2f}% ctl_min_max=[{lo},{hi}] outside_band={outside}")

print("== pf512 (prefill latency/rate per round; logits gate per run)")
pf_ctl, pf_lb = [], []
gate_fail = 0
for arm, acc in (("ctl", pf_ctl), ("lb", pf_lb)):
    for r in ROUNDS:
        for i in range(1, 6):
            path = f"{OUT}/cells-{arm}-r{r}/qwen-gpu-pf512-{i}.json"
            try:
                d = json.load(open(path))
            except Exception as e:
                print(f"LOAD-FAIL {path}: {e}")
                ok = False
                continue
            rate = d.get("prefill_tok_rate", {}).get("median")
            if rate:
                acc.append(rate)
        log = f"{OUT}/cells-{arm}-r{r}.log"
        try:
            text = open(log).read()
        except Exception:
            continue
        gate_fail += text.count("LOGITS GATE FAILED")
print(f"  pf512 tok/s: ctl={pf_ctl} med={statistics.median(pf_ctl) if pf_ctl else '-'}")
print(f"  pf512 tok/s: lb ={pf_lb} med={statistics.median(pf_lb) if pf_lb else '-'}")
if pf_ctl and pf_lb:
    drop = (statistics.median(pf_lb) - statistics.median(pf_ctl)) / statistics.median(pf_ctl) * 100
    print(f"  pf512 delta: {drop:+.2f}% (regression if negative beyond control band)")
print(f"  logits gate FAILED lines: {gate_fail}")
ok &= gate_fail == 0

print("== pf512 record digest")
rec = set()
for arm in ("ctl", "lb"):
    for r in ROUNDS:
        log = f"{OUT}/cells-{arm}-r{r}.log"
        try:
            for line in open(log):
                if "ordered_records_sha256" in line:
                    rec.add(line.split()[-1][:12])
        except Exception:
            pass
print(f"  records seen: {sorted(rec)} expect {{{PF512_RECORD}}}")
ok &= rec == {PF512_RECORD}

d64 = med.get("d64")
if d64:
    print(f"== GATE d64 gain {d64[2]:+.2f}% (need >= +3.00 and outside control band): "
          + ("PASS" if d64[2] >= 3.0 and d64[1] > d64[4] else "FAIL"))
    ok &= d64[2] >= 3.0 and d64[1] > d64[4]
print("CELLS-GATE", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
