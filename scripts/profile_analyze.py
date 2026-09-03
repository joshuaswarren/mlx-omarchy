#!/usr/bin/env python3
"""Analyze an MLX_OMARCHY_GPU_PROFILE NDJSON stream.

Usage:
  python3 profile_analyze.py PROFILE.jsonl [--markers markers.jsonl]
      [--compute-h header-path]

Produces: GPU busy fraction, gap distribution (intra- vs inter-submission),
per-kernel totals/mean/median/share, dispatches per submission, host-side
join/submit/record costs, and the dependency proxy (fraction of consecutive
dispatch pairs with disjoint buffer sets = ceiling for what
dependency-gated barriers could remove).

Kernel names come from the declaration order of the ComputeKernel enum in
overlay/mlx/backend/omarchy/compute.h, the same order the C++ harness
records. GPU ticks convert with meta.period_ns; unwrapping uses
meta.valid_bits. Phases come from an optional markers file written by
scripts/profile_generate.py (CLOCK_MONOTONIC ns, same clock as the harness
host timestamps).
"""

import argparse
import json
import re
import sys


def parse_kernel_names(header_path):
    names = []
    pattern = re.compile(r"^\s{2}(\w+),\s*$")
    inside = False
    with open(header_path, "r", encoding="utf-8") as f:
        for line in f:
            if "enum class ComputeKernel" in line:
                inside = True
                continue
            if inside:
                m = pattern.match(line)
                if m:
                    names.append(m.group(1))
                elif "};" in line:
                    break
    return names


def unwrap(ticks, valid_bits):
    """Unwrap device ticks into a monotonically increasing series."""
    if not ticks:
        return []
    if valid_bits >= 64 or valid_bits == 0:
        return list(ticks)
    m = 1 << valid_bits
    half = m >> 1
    out = [ticks[0]]
    delta = 0
    for prev_raw, raw in zip(ticks, ticks[1:]):
        d = (raw - prev_raw) % m
        if d > half:
            d -= m
        delta += d
        out.append(ticks[0] + delta)
    return out


def pct(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    i = min(len(sorted_vals) - 1, int(p * (len(sorted_vals) - 1) + 0.5))
    return sorted_vals[i]


def fmt_ns(v):
    if v >= 1e6:
        return f"{v / 1e6:.3f}ms"
    if v >= 1e3:
        return f"{v / 1e3:.1f}us"
    return f"{v:.0f}ns"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile")
    ap.add_argument("--markers")
    ap.add_argument("--compute-h", default=None)
    args = ap.parse_args()

    meta = None
    dispatches = []
    joins = []
    submits = []
    with open(args.profile, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            kind = rec.get("k")
            if kind == "meta":
                meta = rec
            elif kind == "d":
                dispatches.append(rec)
            elif kind == "j":
                joins.append(rec)
            elif kind == "s":
                submits.append(rec)

    if meta is None:
        print("no meta line; not a profile file", file=sys.stderr)
        sys.exit(1)

    names = {}
    if args.compute_h:
        for i, n in enumerate(parse_kernel_names(args.compute_h)):
            names[i] = n

    def kname(e):
        return names.get(e, f"kernel_{e}")

    period = meta["period_ns"]
    valid_bits = meta["valid_bits"]
    has_gpu = any("t0" in d for d in dispatches)

    markers = []
    if args.markers:
        with open(args.markers, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    markers.append(json.loads(line))

    join_by_s = {j["s"]: j for j in joins}

    def phase_of_host(t):
        if not markers:
            return None
        cur = markers[0]["p"]
        for m in markers:
            if m["t"] > t:
                break
            cur = m["p"]
        return cur

    for d in dispatches:
        j = join_by_s.get(d["s"])
        d["phase"] = phase_of_host(j["t"]) if j else None

    out = []

    def say(s):
        out.append(s)

    say(f"== profile {args.profile}")
    say(f"   device={meta['device']} period={period}ns "
        f"valid_bits={valid_bits}")
    say(f"   dispatches={len(dispatches)} submissions={len(submits)} "
        f"joins={len(joins)}")
    if markers:
        phases = []
        for m in markers:
            if m["p"] not in phases:
                phases.append(m["p"])
        say(f"   phases={phases}")

    if has_gpu:
        ordered = sorted(
            (d for d in dispatches if "t0" in d), key=lambda d: d["t0"])
        ticks = unwrap([d["t0"] for d in ordered] +
                       [d["t1"] for d in ordered], valid_bits)
        # unwrap preserves lengths and total order of the concatenated
        # (t0..., t1...) list; t1 entries interleave correctly because the
        # concatenation is re-sorted below via paired indices.
        n = len(ordered)
        t0s = ticks[:n]
        t1s = ticks[n:]
        span = max(t1s) - min(t0s)
        busy = 0.0
        kernels_ns = []
        for d, u0, u1 in zip(ordered, t0s, t1s):
            dur = max(0.0, (u1 - u0)) * period
            busy += dur
            kernels_ns.append((dur, d))
        busy_frac = busy / (span * period) if span else 0.0
        say("")
        say(f"== GPU busy fraction: {busy_frac * 100:.2f}% "
            f"(busy {fmt_ns(busy)} / span {fmt_ns(span * period)})")

        # gaps between consecutive kernels on the GPU timeline
        gap_pairs = []
        for a, b in zip(ordered, ordered[1:]):
            ia = ordered.index(a)
            gap = 0.0
            gap_pairs.append((gap, 0))
        # the pair loop above is replaced below with unwrapped values
        gap_pairs = []
        ends = {id(d): u1 for d, u1 in zip(ordered, t1s)}
        starts = {id(d): u0 for d, u0 in zip(ordered, t0s)}
        for a, b in zip(ordered, ordered[1:]):
            gap = (starts[id(b)] - ends[id(a)]) * period
            gap_pairs.append((gap, 1 if a["s"] == b["s"] else 0))
        for tag, sel in (("intra-submission (barrier+driver)",
                          lambda gi: gi[1] == 1),
                         ("inter-submission (host round trip)",
                          lambda gi: gi[1] == 0)):
            vals = sorted(g for g, i in gap_pairs if sel(gi=(g, i)))
            if not vals:
                continue
            say(f"   gaps {tag}: n={len(vals)} total={fmt_ns(sum(vals))} "
                f"p50={fmt_ns(pct(vals, .5))} p90={fmt_ns(pct(vals, .9))} "
                f"p99={fmt_ns(pct(vals, .99))} max={fmt_ns(vals[-1])}")
        hist = {}
        for g, _ in gap_pairs:
            if g <= 0:
                key = "<=0"
            else:
                b = 1
                while g > 10 ** (b + 1):
                    b += 1
                key = f"1e{b + 1}"
            hist[key] = hist.get(key, 0) + 1
        say("   gap histogram (ns decade: count): " + str(
            dict(sorted(hist.items(), key=lambda kv: kv[0]))))

        agg = {}
        for dur, d in kernels_ns:
            a = agg.setdefault(d["e"], {"n": 0, "total": 0.0, "durs": []})
            a["n"] += 1
            a["total"] += dur
            a["durs"].append(dur)
        say("")
        say("== top kernels by cumulative GPU time")
        for e, a in sorted(agg.items(),
                           key=lambda kv: -kv[1]["total"])[:15]:
            durs = sorted(a["durs"])
            say(f"   {kname(e):<26} n={a['n']:<6} "
                f"total={fmt_ns(a['total']):<10} "
                f"share={a['total'] / busy * 100:5.1f}% "
                f"mean={fmt_ns(a['total'] / a['n']):<9} "
                f"p50={fmt_ns(pct(durs, .5))}")

        def bufs(d):
            return {b[0] for b in d.get("b", [])}

        for tag, sel in (("decode", lambda d: d.get("phase") == "decode"),
                         ("all", lambda d: True)):
            ds = [d for d in ordered if sel(d)]
            if len(ds) < 2:
                continue
            indep = sum(1 for a, b in zip(ds, ds[1:])
                        if bufs(a).isdisjoint(bufs(b)))
            say(f"   dependency proxy [{tag}]: {indep}/{len(ds) - 1} "
                f"consecutive pairs fully disjoint "
                f"({indep / (len(ds) - 1) * 100:.1f}% "
                f"barrier-free ceiling)")

    waits = sorted(j["wait"] for j in joins)
    invals = sorted(j["inval"] for j in joins)
    subs = sorted(s["dur"] for s in submits)
    recs = sorted(d["h"] for d in dispatches)
    say("")
    say("== host-side costs (SINGLE-CORE host: upper bounds)")
    if waits:
        say(f"   join wait    n={len(waits)} total={fmt_ns(sum(waits))} "
            f"p50={fmt_ns(pct(waits, .5))} p90={fmt_ns(pct(waits, .9))} "
            f"max={fmt_ns(waits[-1])}")
    if invals:
        say(f"   join inval   n={len(invals)} total={fmt_ns(sum(invals))} "
            f"p50={fmt_ns(pct(invals, .5))} max={fmt_ns(invals[-1])}")
    if subs:
        say(f"   submit()     n={len(subs)} total={fmt_ns(sum(subs))} "
            f"p50={fmt_ns(pct(subs, .5))} max={fmt_ns(subs[-1])}")
    if recs:
        say(f"   dispatch record (host) n={len(recs)} "
            f"total={fmt_ns(sum(recs))} p50={fmt_ns(pct(recs, .5))} "
            f"mean={fmt_ns(sum(recs) / len(recs))}")

    per_join = {}
    for d in dispatches:
        per_join[d["s"]] = per_join.get(d["s"], 0) + 1
    say("")
    say("== dispatches per submission (in submit order)")
    say("   " + str([per_join.get(s["s"], 0) for s in submits]))

    if markers and has_gpu:
        by_phase = {}
        for d in dispatches:
            by_phase.setdefault(d["phase"], []).append(d)
        say("")
        say("== per phase")
        for ph in sorted(by_phase, key=lambda p: (p is None, p or "")):
            ds = by_phase[ph]
            dur = sum((d["t1"] - d["t0"]) * period
                      for d in ds if "t0" in d)
            joins_ph = [j for j in joins if phase_of_host(j["t"]) == ph]
            say(f"   {ph}: dispatches={len(ds)} gpu_busy={fmt_ns(dur)} "
                f"joins={len(joins_ph)} join_wait_total="
                f"{fmt_ns(sum(j['wait'] for j in joins_ph))}")
        n_tok = sum(1 for m in markers if m["p"] == "tok")
        decode_ds = len(by_phase.get("decode", by_phase.get("tok", [])))
        decode_subs = [s for s in submits
                       if phase_of_host(s["t"]) in ("decode", "tok")]
        if n_tok:
            say(f"   dispatches/decode-token: {decode_ds / n_tok:.1f} "
                f"({decode_ds} over {n_tok} tokens)")
            say(f"   submissions/decode-token: "
                f"{len(decode_subs) / n_tok:.1f} "
                f"({len(decode_subs)} over {n_tok} tokens)")
            jw = sum(j["wait"] for j in joins
                     if phase_of_host(j["t"]) in ("decode", "tok"))
            say(f"   join wait per decode-token: {fmt_ns(jw / n_tok)}")

    print("\n".join(out))


if __name__ == "__main__":
    main()
