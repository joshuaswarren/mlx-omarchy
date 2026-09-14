#!/usr/bin/env python3
"""Decode-phase census and gap decomposition from archived GPU profiles.

Host-only. Reads an MLX_OMARCHY_GPU_PROFILE capture, isolates the decode
submissions, and reports:

  - dispatches per decode token, per kernel, with the launch grid of each
    (counts and grids are exact: they are recorded, not timed)
  - the distinct (buffer, offset, range) binding footprint per token
  - host-side cost per dispatch: the record cost ("h") and the per-submission
    vkQueueSubmit duration regressed against dispatch count
  - bracketed GPU durations and inter-dispatch gaps, reported ONLY to show
    that they overprice the token (bracketed busy exceeds the clean-run wall),
    never as a price

and, from clean-run rates alone, the two-term decode-deficit decomposition:
a context-independent per-dispatch term and a KV-length-dependent term.

Usage:
  ./decode_census.py PROFILE.jsonl --compute-h compute-b41e2b74.h
      [--decode-subs 7,8,9,10,11,12]
  ./decode_census.py --deficits-only
"""

import argparse
import collections
import gzip
import json
import re
import statistics
import sys

# Clean-run medians, all from existing receipts. Linux jwm1:
# receipts/2026-09-14-jwm1-gpu-parity-rerun.md. Linux jw16 (Honeykrisp fork,
# same driver as jwm1): receipts/2026-09-14-jw16-honeykrisp-mesa-parity.
# Native divisors: receipts/native-baseline-2026-09-06 summary, as pinned by
# receipts/2026-09-13-jwm1-jw16-gpu-parity.md.
LEGS = {
    ("jwm1", "30/32"): (107.285, 150.57),
    ("jwm1", "1053/32"): (95.0352, 140.38),
    ("jw16", "30/32"): (169.6769, 286.96),
    ("jw16", "1053/32"): (143.1857, 283.79),
}
DISPATCHES_PER_TOKEN = 249

# Qwen2.5-0.5B-Instruct-4bit, affine 4-bit / group 64, tied embeddings.
HIDDEN, LAYERS, INTER = 896, 24, 4864
KV_WIDTH, HEADS, HEAD_DIM, VOCAB = 128, 14, 64, 151936
BYTES_PER_ELT = 0.5 + 2 * 2 / 64  # 4-bit weight + f16 scale + f16 bias per 64
# Vendor specification, not measured here. fp32 FMA peak follows the
# convention of receipts/2026-09-14-jw16-max-gpu-attribution.md:
# cores x 128 lanes x 2 x 1.296 GHz.
PARTS = {"jwm1": (68.25, 2654.0), "jw16": (400.0, 10617.0)}


def parse_kernel_names(path):
    names, inside = [], False
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if "enum class ComputeKernel" in line:
                inside = True
                continue
            if inside:
                m = re.match(r"^\s{2}(\w+),\s*$", line)
                if m:
                    names.append(m.group(1))
                elif "};" in line:
                    break
    return {i: n for i, n in enumerate(names)}


def load(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def traffic_and_flops():
    per_layer = (
        HIDDEN * HIDDEN            # q
        + KV_WIDTH * HIDDEN * 2    # k, v
        + HIDDEN * HIDDEN          # o
        + INTER * HIDDEN * 2       # gate, up
        + HIDDEN * INTER           # down
    )
    weights = per_layer * LAYERS * BYTES_PER_ELT
    lm_head = VOCAB * HIDDEN * BYTES_PER_ELT
    kv = 1053 * KV_WIDTH * 2 * 2 * LAYERS       # K and V, f16, every layer
    norms = (LAYERS * 2 * HIDDEN + HIDDEN) * 2
    flops = 2 * (per_layer * LAYERS + VOCAB * HIDDEN)
    flops += 2 * 2 * 1053 * HEAD_DIM * HEADS * LAYERS  # QK and PV at ctx 1053
    return per_layer, weights + lm_head + kv + norms, flops


def deficits():
    ms = {k: (1000.0 / lin, 1000.0 / nat) for k, (lin, nat) in LEGS.items()}
    print("== clean-run wall per decode token (existing receipts)")
    print(f"   {'leg':<16}{'linux ms':>10}{'native ms':>11}"
          f"{'deficit ms':>12}{'parity':>9}")
    for (host, leg), (lin, nat) in ms.items():
        rate_l, rate_n = LEGS[(host, leg)]
        print(f"   {host + ' ' + leg:<16}{lin:>10.5f}{nat:>11.5f}"
              f"{lin - nat:>12.5f}{rate_l / rate_n * 100:>8.2f}%")

    print("\n== two-term decomposition (per-token, from the rates above)")
    print("   term A = deficit at 30-token context: context-independent, and")
    print("            the dispatch count per token does not vary with")
    print("            context (see the grid census), so A is per-dispatch.")
    print("   term B = deficit growth from 30 to 1053 context: the KV stream.")
    for host in ("jwm1", "jw16"):
        a = ms[(host, "30/32")][0] - ms[(host, "30/32")][1]
        t = ms[(host, "1053/32")][0] - ms[(host, "1053/32")][1]
        print(f"   {host}: A={a:.4f} ms ({a / t * 100:4.1f}% of deficit) "
              f"= {a / DISPATCHES_PER_TOKEN * 1e3:5.2f} us/dispatch | "
              f"B={t - a:.4f} ms ({(t - a) / t * 100:4.1f}%)")
    d1 = ms[("jwm1", "1053/32")][0] - ms[("jwm1", "1053/32")][1]
    d2 = ms[("jw16", "1053/32")][0] - ms[("jw16", "1053/32")][1]
    print(f"   1053 deficit jwm1/jw16 = {d1 / d2:.4f}  <- device-invariance:")
    print("   the GPU is 3.30x faster (measured prefill GPU-busy ratio,")
    print("   receipts/2026-09-14-jw16-max-gpu-attribution.md) yet the")
    print("   absolute deficit is the same.")

    per_layer, total_bytes, flops = traffic_and_flops()
    print(f"\n== required traffic and arithmetic per token (from the model "
          f"config; {per_layer} weight elements per layer)")
    print(f"   {total_bytes / 1e6:.2f} MB, {flops / 1e9:.3f} GFLOP")
    for host, (bw_spec, fl_spec) in PARTS.items():
        for tag, t in (("linux", ms[(host, "1053/32")][0]),
                       ("native", ms[(host, "1053/32")][1])):
            bw = total_bytes / 1e9 / (t / 1e3)
            fl = flops / 1e9 / (t / 1e3)
            print(f"   {host} {tag:<7}{bw:7.2f} GB/s = {bw / bw_spec * 100:5.1f}%"
                  f" of spec | {fl:7.1f} GFLOP/s ="
                  f" {fl / fl_spec * 100:5.2f}% of fp32 peak")

    inc = (1068.5 - 45.5) * KV_WIDTH * 2 * 2 * LAYERS
    print(f"\n== term B as a stream: {inc / 1e6:.2f} MB of extra KV per token")
    print("   (mean KV length 45.5 on the 30/32 leg, 1068.5 on the 1053/32 leg)")
    for host in ("jwm1", "jw16"):
        dl = ms[(host, "1053/32")][0] - ms[(host, "30/32")][0]
        dn = ms[(host, "1053/32")][1] - ms[(host, "30/32")][1]
        print(f"   {host}: linux +{dl:.4f} ms -> {inc / 1e9 / (dl / 1e3):7.2f}"
              f" GB/s | native +{dn:.4f} ms -> {inc / 1e9 / (dn / 1e3):7.2f} GB/s")


# Dispatches removable per token by folding each elementwise consumer into
# the GEMV store, from the grid census: FastRopeF16 48 + SwigluF16 24,
# then FastRmsNormF16 49, then every remaining <=14-workgroup dispatch.
FUSION_STEPS = (
    (72, "RoPE + SwiGLU folded"),
    (121, "+ RMS norm folded"),
    (151, "every <=14-wg dispatch gone (ceiling)"),
)


def pricing():
    """Price dispatch removal at term A's own measured per-dispatch rate."""
    print("\n== proposal pricing (term A rate = 30-token deficit / 249)")
    for host in ("jwm1", "jw16"):
        lin, nat = LEGS[(host, "1053/32")]
        base = 1000.0 / lin
        short_lin, short_nat = LEGS[(host, "30/32")]
        rate = (1000.0 / short_lin - 1000.0 / short_nat) / DISPATCHES_PER_TOKEN
        print(f"   {host}: {rate * 1e3:.2f} us/dispatch; "
              f"today {DISPATCHES_PER_TOKEN} disp, {base:.4f} ms, "
              f"{lin:.2f} tok/s, {lin / nat * 100:.2f}% of native")
        for n, label in FUSION_STEPS:
            t = base - n * rate
            print(f"     -{n:>3} disp -> {DISPATCHES_PER_TOKEN - n:>3} disp, "
                  f"{t:.4f} ms, {1000 / t:7.2f} tok/s, "
                  f"{(1000 / t) / nat * 100:5.2f}% of native   [{label}]")
    print("   Conservative: removing a dispatch removes the whole per-dispatch")
    print("   cost, not only the part exceeding native. Optimistic: term A's")
    print("   rate averages 1-workgroup and 1216-workgroup dispatches alike.")


def census(path, compute_h, decode_subs):
    names = parse_kernel_names(compute_h)
    recs = load(path)
    meta = next(r for r in recs if r["k"] == "meta")
    disp = [r for r in recs if r["k"] == "d"]
    subs = {r["s"]: r for r in recs if r["k"] == "s"}
    queues = {r["s"]: r for r in recs if r["k"] == "q"}
    counts = collections.Counter(r["s"] for r in disp)
    print(f"== {path}")
    print(f"   device={meta['device']} dispatches={len(disp)} "
          f"submissions={len(subs)}")
    print(f"   dispatches per submission: {[counts[s] for s in sorted(counts)]}")

    print("\n== host vkQueueSubmit duration vs dispatch count")
    print(f"   {'sub':>4}{'disp':>6}{'vkQueueSubmit us':>18}{'us/disp':>9}")
    rows = []
    for s in sorted(counts):
        q = queues[s]
        us = (q["queue_t1"] - q["queue_t0"]) / 1e3
        rows.append((s, counts[s], us))
        print(f"   {s:>4}{counts[s]:>6}{us:>18.1f}{us / counts[s]:>9.2f}")
    big = [r for r in rows if r[1] > 100]
    small = [r for r in rows if r[1] < 100]
    if big and small:
        nb = statistics.median(r[1] for r in big)
        ns = statistics.median(r[1] for r in small)
        mb = statistics.median(r[2] for r in big)
        msm = statistics.median(r[2] for r in small)
        slope = (mb - msm) / (nb - ns)
        print(f"   median-pair slope over all submissions: {slope:.2f} "
              f"us/dispatch, intercept {msm - slope * ns:+.1f} us/submission")

    if not decode_subs:
        return
    step = sorted(decode_subs)[-2:]
    ds = [x for x in disp if x["s"] in step]
    print(f"\n== last decode token (submissions {step}), {len(ds)} dispatches")
    agg = collections.defaultdict(lambda: [0, 0])
    for x in ds:
        k = names[x["e"]]
        agg[k][0] += 1
        agg[k][1] += x["t1"] - x["t0"]
    brk = sum(v[1] for v in agg.values())
    print(f"   {'kernel':<28}{'n':>4}{'/layer':>8}{'grids (wg)':>22}"
          f"{'bracketed ms':>14}{'share':>8}")
    grids = collections.defaultdict(collections.Counter)
    for x in ds:
        grids[names[x["e"]]][x["gx"] * x["gy"] * x["gz"]] += 1
    for k, (n, tt) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        g = ",".join(f"{w}x{c}" for w, c in sorted(grids[k].items()))
        print(f"   {k:<28}{n:>4}{n / LAYERS:>8.3f}{g:>22}"
              f"{tt / 1e6:>14.3f}{tt / brk * 100:>7.1f}%")
    tiny = [x for x in ds if x["gx"] * x["gy"] * x["gz"] <= 14]
    one = [x for x in ds if x["gx"] * x["gy"] * x["gz"] == 1]
    print(f"   dispatches launching <=14 workgroups: {len(tiny)}/{len(ds)} "
          f"({len(tiny) / len(ds) * 100:.1f}%); exactly 1 workgroup: {len(one)}")

    foot = {}
    for x in ds:
        for buf, off, rng in x["b"]:
            foot[(buf, off, rng)] = rng
    print(f"   distinct binding footprint: {sum(foot.values()) / 1e6:.2f} MB "
          f"over {len(foot)} (buffer,offset,range) triples")

    hs = sorted(x["h"] for x in ds)
    print(f"   host dispatch-record cost: {sum(hs) / 1e6:.3f} ms/token, "
          f"p50 {statistics.median(hs) / 1e3:.2f} us")
    order = sorted(ds, key=lambda x: x["t0"])
    gaps = [b["t0"] - a["t1"] for a, b in zip(order, order[1:])
            if a["s"] == b["s"]]
    busy = sum(x["t1"] - x["t0"] for x in ds) / 1e6
    span = (max(x["t1"] for x in ds) - min(x["t0"] for x in ds)) / 1e6
    print(f"   bracketed busy {busy:.3f} ms, span {span:.3f} ms, intra gaps "
          f"n={len(gaps)} total {sum(gaps) / 1e6:.3f} ms "
          f"p50 {statistics.median(gaps) / 1e3:.1f} us")
    print("   NOT A PRICE: bracketed busy and span both exceed the clean-run")
    print("   wall for this leg, so no per-dispatch duration here is usable.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile", nargs="?")
    ap.add_argument("--compute-h")
    ap.add_argument("--decode-subs", default="")
    ap.add_argument("--deficits-only", action="store_true")
    args = ap.parse_args()
    if args.profile:
        if not args.compute_h:
            print("--compute-h is required with a profile", file=sys.stderr)
            return 2
        subs = [int(s) for s in args.decode_subs.split(",") if s]
        census(args.profile, args.compute_h, subs)
        print()
    if args.deficits_only or not args.profile:
        deficits()
        pricing()
    return 0


if __name__ == "__main__":
    sys.exit(main())
