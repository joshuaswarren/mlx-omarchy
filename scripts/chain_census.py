#!/usr/bin/env python3
"""Elementwise-chain census over an MLX_OMARCHY_GPU_PROFILE NDJSON stream.

Answers three questions for one decode run:
1. How many GPU dispatches does a decode token issue, split by the tape
   path (dispatches recorded inside eval_compiled_tape, event field
   "tp":1) and the eager path ("tp":0)?
2. What is the per-kernel composition of each path?
3. What elementwise chains actually occur: maximal runs of consecutive
   elementwise-class dispatches where each dispatch writes the buffer
   the next one reads, at the same element count. Reports the chain
   length histogram, per-length dispatch totals, and the distinct op
   sequences.

Ordering: per-dispatch events ("k":"d") are flushed at submission
completion and carry no host timestamp, so each is ordered by the host
timestamp of its submission ("k":"s" events), then by file order within
the submission. Token windows come from the markers file written by
scripts/profile_generate.py.

Usage:
  python3 chain_census.py PROFILE.jsonl --compute-h compute.h \
      [--markers markers.jsonl]
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict


def kernel_names(header_path):
    names = []
    pat = re.compile(r"^\s*([A-Za-z0-9_]+)\s*,?\s*$")
    inside = False
    with open(header_path, encoding="utf-8") as fh:
        for line in fh:
            if "enum class ComputeKernel" in line:
                inside = True
                continue
            if inside:
                if "}" in line:
                    break
                m = pat.match(line)
                if m and m.group(1) != "Count":
                    names.append(m.group(1))
    return names


ELEMENTWISE_CLASSES = (
    "Elementwise",
    "Cast",
    "CopyGeneral",
    "Select",
    "Compare",
    "LogicalOr",
    "Fill",
)


def is_elementwise(name):
    return name.startswith(ELEMENTWISE_CLASSES)


# Ops that fuse into one elementwise kernel (MLX primitive names as they
# appear in the tape census dump). Views record zero dispatches and ride
# along free; they cannot break a chain.
FUSABLE_OPS = frozenset((
    "Abs", "Add", "ArcCos", "ArcSin", "ArcTan", "AsType", "BitwiseAnd",
    "BitwiseOr", "BitwiseXor", "Ceil", "Cos", "Divide", "Equal", "Erf",
    "ErfInv", "Exp", "Expm1", "Floor", "FloorDivide", "Greater",
    "GreaterEqual", "Less", "LessEqual", "Log", "Log1p", "Log2", "Log10",
    "LogicalAnd", "LogicalNot", "LogicalOr", "Maximum", "Minimum",
    "Multiply", "Negative", "NotEqual", "Power", "Remainder", "Round",
    "Select", "Sigmoid", "Sign", "Sin", "Sinh", "Square", "Sqrt",
    "Subtract", "Tan", "Tanh", "Where",
))
# Zero-dispatch views (reshape/transpose/expand recorded as Broadcast).
VIEW_OPS = frozenset(("Broadcast",))


def parse_tape_dump(path):
    """Parse MLX_OMARCHY_TAPE_CENSUS lines into per-eval node lists."""
    tapes = []
    pat = re.compile(
        r"n(\d+)=(\S+) (\S+) uses=(\d+) in=\[([^\]]*)\]( OUT)?")
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.startswith("CENSUS "):
                continue
            nodes = []
            for seg in line.strip().split(" | ")[1:]:
                m = pat.match(seg)
                if not m:
                    sys.exit("unparsable census segment: " + seg)
                ins = [int(x) for x in m.group(5).split(",") if x and x != "x"]
                nodes.append({
                    "idx": int(m.group(1)), "op": m.group(2),
                    "kind": m.group(3), "uses": int(m.group(4)),
                    "ins": ins, "out": bool(m.group(6)),
                })
            tapes.append(nodes)
    return tapes


def analyze_tape(nodes):
    """Chain census inside one tape DAG.

    A chain is a maximal run of fusable nodes where every interior node
    is consumed only by the next node of the run (sole-consumer rule).
    View nodes (zero dispatches) ride along. Also computes the
    multi-consumer-capable whole-region ceiling.
    """
    consumers = defaultdict(list)
    for nd in nodes:
        for i in nd["ins"]:
            consumers[i].append(nd["idx"])

    def cost(nd):
        return 0 if nd["op"] in VIEW_OPS else 1

    def fusable(nd):
        return nd["op"] in FUSABLE_OPS or nd["op"] in VIEW_OPS

    # v may extend the chain of u iff u is fusable and u's only
    # consumer is v. The definition constrains INTERIOR nodes only,
    # so a chain end may be fed by several outside nodes. Ties
    # (several sole-consumer preds) attach v to the first in input
    # order; the other preds' chains end at them.
    pred_of = {}
    for nd in nodes:
        if not fusable(nd):
            continue
        preds = [i for i in nd["ins"]
                 if fusable(nodes[i]) and len(consumers[i]) == 1
                 and consumers[i][0] == nd["idx"]]
        if preds:
            pred_of[nd["idx"]] = preds[0]

    chain_of = {}
    chains = []
    for nd in nodes:
        if not fusable(nd):
            continue
        p = pred_of.get(nd["idx"])
        if p is not None:
            chain_of[nd["idx"]] = chain_of[p]
        else:
            chain_of[nd["idx"]] = len(chains)
            chains.append([])
        chains[chain_of[nd["idx"]]].append(nd)

    chain_costs = Counter()
    chain_views = Counter()
    for ch in chains:
        c = sum(cost(nd) for nd in ch)
        chain_costs[c] += 1
        chain_views[c] += sum(1 for nd in ch if nd["op"] in VIEW_OPS)
    sole_savings = sum(max(c - 1, 0) for c in chain_costs.elements())

    # Multi-consumer-capable rule: one kernel per connected fusable
    # region, interior use counts unrestricted.
    seen = set()
    region_stats = []
    for nd in nodes:
        if not fusable(nd) or nd["idx"] in seen:
            continue
        stack, comp = [nd["idx"]], set()
        while stack:
            u = stack.pop()
            if u in comp:
                continue
            comp.add(u)
            stack.extend(i for i in nodes[u]["ins"] if fusable(nodes[i]))
            stack.extend(v for v in consumers[u] if fusable(nodes[v]))
        seen |= comp
        c = sum(cost(nodes[i]) for i in comp)
        interior_multi = any(
            len(consumers[i]) >= 2 for i in comp
            if not nodes[i]["out"] and all(v in comp for v in consumers[i]))
        region_stats.append((c, interior_multi))
    region_savings = sum(max(c - 1, 0) for c, _ in region_stats)
    region_blocked = sum(1 for _, m in region_stats if m)
    return {
        "nodes": len(nodes),
        "dispatch_nodes": sum(cost(nd) for nd in nodes),
        "ops": Counter(nd["op"] for nd in nodes),
        "chain_costs": chain_costs,
        "chain_views": chain_views,
        "sole_savings": sole_savings,
        "regions": region_stats,
        "region_savings": region_savings,
        "region_blocked": region_blocked,
    }


def self_test():
    """Assert analyze_tape on hand-built tapes with known answers."""
    def tape(*nodes):
        return [{"op": p[0], "ins": list(p[2]),
                 "out": len(p) > 3 and p[3], "idx": i, "kind": "f",
                 "uses": 0} for i, p in enumerate(nodes)]

    # 7-node standalone swiglu: views with uses=2 block sole-consumer.
    r = analyze_tape(tape(
        ("Sigmoid", 0, (), False), ("Broadcast", 0, (0,), False),
        ("Broadcast", 0, (0,), False), ("Multiply", 0, (1, 2), False),
        ("Broadcast", 0, (3,), False), ("Broadcast", 0, (3,), False),
        ("Multiply", 0, (4, 5), True)))
    assert r["chain_costs"] == Counter({1: 3, 0: 2}), r["chain_costs"]
    assert r["sole_savings"] == 0
    assert r["region_blocked"] == 1 and r["region_savings"] == 2

    # 3-node compiled swiglu: clean sole-consumer chain of length 3.
    r = analyze_tape(tape(
        ("Sigmoid", 0, (), False), ("Multiply", 0, (0,), False),
        ("Multiply", 0, (1,), True)))
    assert r["chain_costs"] == Counter({3: 1})
    assert r["sole_savings"] == 2 and r["region_blocked"] == 0

    # 5-node mask: LessEqual->Select chain survives two extra feeders
    # into the chain end.
    r = analyze_tape(tape(
        ("Broadcast", 0, (), False), ("LessEqual", 0, (0,), False),
        ("Broadcast", 0, (), False), ("Broadcast", 0, (), False),
        ("Select", 0, (1, 2, 3), True)))
    assert r["chain_costs"] == Counter({2: 1, 0: 2})
    assert r["sole_savings"] == 1

    # Multi-consumer interior: B feeds C and D. A->B still fuses (A's
    # only consumer is B); the block starts at B's fan-out, and the
    # region rule sees one blocked region covering all four nodes.
    r = analyze_tape(tape(
        ("Add", 0, (), False), ("Add", 0, (0,), False),
        ("Add", 0, (1,), True), ("Add", 0, (1,), True)))
    assert r["chain_costs"] == Counter({2: 1, 1: 2}), r["chain_costs"]
    assert r["sole_savings"] == 1
    assert r["region_blocked"] == 1 and r["region_savings"] == 3

    # Diamond: two sole-consumer preds into one node - the node joins
    # the first chain, the second ends alone.
    r = analyze_tape(tape(
        ("Add", 0, (), False), ("Add", 0, (), False),
        ("Multiply", 0, (0, 1), True)))
    assert r["chain_costs"] == Counter({2: 1, 1: 1})
    assert r["sole_savings"] == 1
    print("tape-census self-test: all assertions pass")


def tape_census_main(args):
    tapes = parse_tape_dump(args.tapes)
    if not tapes:
        sys.exit("no CENSUS lines in " + args.tapes)
    uniq = defaultdict(int)
    for nodes in tapes:
        key = tuple((nd["op"], nd["kind"], tuple(nd["ins"]), nd["out"])
                    for nd in nodes)
        uniq[key] += 1
    print(f"tape evaluations: {len(tapes)}, unique tapes: {len(uniq)}")
    agg_chains = Counter()
    agg = Counter()
    for key, evals in sorted(uniq.items(), key=lambda kv: -len(kv[0])):
        nodes = [{"op": op, "kind": kind, "ins": list(ins), "out": out,
                  "idx": i, "uses": 0}
                 for i, (op, kind, ins, out) in enumerate(key)]
        r = analyze_tape(nodes)
        print(f"\n== tape nodes={r['nodes']} dispatch_nodes="
              f"{r['dispatch_nodes']} evals={evals} ==")
        print("  ops: " + " ".join(
            f"{op}x{n}" for op, n in r["ops"].most_common()))
        print("  sole-consumer chains by dispatch-recording length:")
        for c, cnt in sorted(r["chain_costs"].items()):
            print(f"    len={c:3d}: count={cnt:4d} (view nodes inside:"
                  f" {r['chain_views'][c]})")
        print(f"  sole-consumer savings/eval: {r['sole_savings']}"
              f" of {r['dispatch_nodes']} dispatch nodes")
        print(f"  regions: {len(r['regions'])}"
              f" blocked-by-multi-consumer: {r['region_blocked']}"
              f" region savings/eval: {r['region_savings']}")
        agg_chains.update({c: cnt * evals for c, cnt in
                           r["chain_costs"].items()})
        agg["evals"] += evals
        agg["dispatch_nodes"] += r["dispatch_nodes"] * evals
        agg["sole_savings"] += r["sole_savings"] * evals
        agg["region_savings"] += r["region_savings"] * evals
    print("\n== all tapes aggregated ==")
    print(f"evals: {agg['evals']}, dispatch nodes total:"
          f" {agg['dispatch_nodes']}")
    print("chain-length histogram over all evals (dispatch-recording"
          " nodes per chain):")
    for c, cnt in sorted(agg_chains.items()):
        print(f"  len={c:3d}: {cnt}")
    print(f"sole-consumer fusable ceiling: {agg['sole_savings']} dispatches"
          f" removable of {agg['dispatch_nodes']}"
          f" ({100.0 * agg['sole_savings'] / max(agg['dispatch_nodes'], 1):.1f}%)")
    print(f"region-rule ceiling (multi-consumer capable):"
          f" {agg['region_savings']} dispatches"
          f" ({100.0 * agg['region_savings'] / max(agg['dispatch_nodes'], 1):.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile", nargs="?")
    ap.add_argument("--compute-h")
    ap.add_argument("--markers")
    ap.add_argument("--detail-tokens", type=int, default=2,
                    help="tokens to dump full dispatch sequences for")
    ap.add_argument("--tapes",
                    help="MLX_OMARCHY_TAPE_CENSUS dump: analyze chain"
                         " structure inside tapes instead of the"
                         " dispatch stream")
    ap.add_argument("--self-test", action="store_true",
                    help="assert analyze_tape on hand-built tapes")
    args = ap.parse_args()

    if args.tapes:
        tape_census_main(args)
        return

    if args.self_test:
        self_test()
        return

    if not args.profile:
        sys.exit("profile path required unless --tapes is given")

    if not args.compute_h:
        sys.exit("--compute-h required for dispatch-stream analysis")

    names = kernel_names(args.compute_h)
    if not names:
        sys.exit("no kernel names parsed from " + args.compute_h)

    # Pass 1: submission host timestamps, in submission order.
    sub_t = {}
    with open(args.profile, encoding="utf-8") as fh:
        for line in fh:
            ev = json.loads(line)
            if ev.get("k") == "s":
                sub_t[ev["s"]] = ev["t"]

    # Pass 2: dispatch events ordered by (submission time, file order).
    events = []
    with open(args.profile, encoding="utf-8") as fh:
        for line in fh:
            ev = json.loads(line)
            if ev.get("k") == "d":
                events.append(ev)
    events.sort(key=lambda e: (sub_t.get(e["s"], 0)))

    # Decode token windows from markers (host monotonic ns).
    token_bounds = []
    if args.markers:
        toks = []
        with open(args.markers, encoding="utf-8") as fh:
            for line in fh:
                ev = json.loads(line)
                if ev["p"] in ("decode_start", "tok", "decode_done"):
                    toks.append(ev)
        starts = [e["t"] for e in toks if e["p"] == "decode_start"]
        tok_ts = [e["t"] for e in toks if e["p"] == "tok"]
        if starts:
            token_bounds.append((0, starts[0], tok_ts[0] if tok_ts else None))
        for i, t in enumerate(tok_ts):
            hi = tok_ts[i + 1] if i + 1 < len(tok_ts) else None
            token_bounds.append((i + 1, t, hi))

    def token_of(ev):
        t = sub_t.get(ev["s"], 0)
        if not token_bounds:
            return None
        for idx, lo, hi in token_bounds:
            if t >= lo and (hi is None or t < hi):
                return idx
        return None

    tok_total = Counter()
    tok_tape = Counter()
    tok_kernel = defaultdict(Counter)
    tok_kernel_tape = defaultdict(Counter)
    total = Counter()
    tape_total = Counter()
    for ev in events:
        k = names[ev["e"]] if ev["e"] < len(names) else f"K{ev['e']}"
        tp = ev.get("tp", 0)
        total[k] += 1
        if tp:
            tape_total[k] += 1
        t = token_of(ev)
        if t is not None:
            tok_total[t] += 1
            tok_kernel[t][k] += 1
            if tp:
                tok_tape[t] += 1
                tok_kernel_tape[t][k] += 1

    def counter_block(title, counter, grand):
        print(f"\n{title}")
        for k, n in counter.most_common():
            print(f"  {k:28s} {n:7d}  {100.0 * n / max(grand, 1):5.1f}%")

    print(f"dispatch events parsed: {len(events)}")
    print(f"decode tokens: {len(tok_total)}")
    if tok_total:
        counts = sorted(tok_total.values())
        print(f"dispatches/token: min={counts[0]} median="
              f"{counts[len(counts) // 2]} max={counts[-1]}")
        tap = sorted(tok_tape.values())
        if tap:
            print(f"tape-path dispatches/token: min={tap[0]} median="
                  f"{tap[len(tap) // 2]} max={tap[-1]}")

    counter_block("== whole-run by kernel ==", total, sum(total.values()))
    counter_block("== whole-run TAPE-path by kernel ==", tape_total,
                  sum(total.values()))
    print("\n== median decode token: kernel composition ==")
    if tok_kernel:
        med = sorted(tok_total, key=lambda t: tok_total[t])[
            len(tok_total) // 2]
        print(f"(token {med}: {tok_total[med]} dispatches,"
              f" {tok_tape[med]} tape-path)")
        counter_block("-- all paths --", tok_kernel[med], tok_total[med])
        counter_block("-- tape path --", tok_kernel_tape[med],
                      tok_total[med])

    # Chain census within tokens: runs of consecutive elementwise
    # dispatches linked by buffer flow (writer -> reader, same count).
    def chains_for(seq):
        chains = []
        run = []
        for ev in seq:
            k = names[ev["e"]] if ev["e"] < len(names) else f"K{ev['e']}"
            if not is_elementwise(k):
                if len(run) >= 2:
                    chains.append(run)
                run = []
                continue
            linked = False
            if run:
                prev = run[-1]
                pbs = prev.get("b", [])
                cbs = ev.get("b", [])
                if prev["n"] == ev["n"] and len(pbs) >= 3 and cbs:
                    out_binding = pbs[2]
                    linked = any(c == out_binding for c in cbs[:2])
                if not linked:
                    if len(run) >= 2:
                        chains.append(run)
                    run = []
            run.append(ev)
        if len(run) >= 2:
            chains.append(run)
        return chains

    tok_chains = defaultdict(list)
    for t in sorted(tok_total):
        tok_chains[t] = chains_for(
            [ev for ev in events if token_of(ev) == t])

    hist = Counter()
    disp_in_chains = Counter()
    op_seqs = Counter()
    for chains in tok_chains.values():
        for run in chains:
            hist[len(run)] += 1
            disp_in_chains[len(run)] += len(run)
            sig = "+".join(
                f"{names[e['e']] if e['e'] < len(names) else '?'}"
                f"/op{e['op']}" for e in run)
            op_seqs[sig] += 1
    print("\n== chain census, decode tokens, all paths ==")
    print(f"chains>=2: {sum(hist.values())}, dispatches inside chains:"
          f" {sum(disp_in_chains.values())}")
    for length in sorted(hist):
        print(f"  len={length:2d}: count={hist[length]:5d} dispatches="
              f"{disp_in_chains[length]:5d}")
    for sig, n in op_seqs.most_common(15):
        print(f"  x{n:4d} {sig[:150]}")

    for t in sorted(tok_total)[: args.detail_tokens]:
        print(f"\n== token {t} dispatch sequence ==")
        for ev in [e for e in events if token_of(e) == t]:
            k = names[ev["e"]] if ev["e"] < len(names) else f"K{ev['e']}"
            print(f"  {'T' if ev.get('tp', 0) else 'e'} {k:24s} op="
                  f"{ev['op']:3d} n={ev['n']:6d}")


if __name__ == "__main__":
    main()
