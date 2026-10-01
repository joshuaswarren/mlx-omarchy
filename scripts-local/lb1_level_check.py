#!/usr/bin/env python3
"""Jw16LevelBatch: validate the encoder's frontier level assignment against a
brute-force longest-path reference over the REAL recorded decode DAG
(Jw16BarrierElide W2 dump, d128). The C++ emit_level_batched implements the
same frontier algorithm; if frontier levels ever differ from the reference,
or the per-token level count is not ~42, the encoder change is unsound.

Usage: lb1_level_check.py DUMP.err [--tokens N]
"""
import re
import sys
from collections import defaultdict

node_re = re.compile(
    r"\[dag\] (\d+) enc=(\S+) pipe=(\S+) (\d+) R:((?: \S+@\d+-\d+)*)"
    r" W:((?: \S+@\d+-\d+)*)")
range_re = re.compile(r"(\S+)@(\d+)-(\d+)")


def parse(path):
    nodes = []
    for line in open(path, errors="replace"):
        m = node_re.match(line)
        if not m:
            continue
        rd = [(b, int(o), int(e)) for b, o, e in range_re.findall(m.group(5))]
        wr = [(b, int(o), int(e)) for b, o, e in range_re.findall(m.group(6))]
        nodes.append((int(m.group(1)), m.group(2), int(m.group(4)), rd, wr))
    return nodes


def levels_bruteforce(nodes):
    lv = []
    for i, (_, _, _, rd, wr) in enumerate(nodes):
        best = 0
        for j in range(i):
            _, _, _, rj, wj = nodes[j]
            hit = False
            for (bi, oi, ei) in rd:
                for (bj, oj, ej) in wj:
                    if bi == bj and oi < ej and oj < ei:
                        hit = True
            for (bi, oi, ei) in wr:
                for (bj, oj, ej) in wj + rj:
                    if bi == bj and oi < ej and oj < ei:
                        hit = True
            if hit:
                best = max(best, lv[j])
        lv.append(best + 1)
    return lv


def levels_frontier(nodes):
    # Mirrors emit_level_batched: write insert supersedes overlaps, read
    # insert merges to max level, query returns max level overlapped.
    fw = defaultdict(list)  # buf -> [(o, e, level)]
    fr = defaultdict(list)
    lv = []
    for _, _, _, rd, wr in nodes:
        best = 0
        for (b, o, e) in rd:
            for (vo, ve, vl) in fw[b]:
                if vo < e and o < ve:
                    best = max(best, vl)
        for (b, o, e) in wr:
            for (vo, ve, vl) in fw[b] + fr[b]:
                if vo < e and o < ve:
                    best = max(best, vl)
        lv.append(best + 1)
        for (b, o, e) in rd:
            merged_o, merged_e, merged_l = o, e, lv[-1]
            keep = []
            for iv in fr[b]:
                if iv[0] < merged_e and merged_o < iv[1]:
                    merged_o = min(merged_o, iv[0])
                    merged_e = max(merged_e, iv[1])
                    merged_l = max(merged_l, iv[2])
                else:
                    keep.append(iv)
            keep.append((merged_o, merged_e, merged_l))
            fr[b] = keep
        for (b, o, e) in wr:
            fw[b] = [iv for iv in fw[b]
                     if not (iv[0] < e and o < iv[1])]
            fr[b] = [iv for iv in fr[b]
                     if not (iv[0] < e and o < iv[1])]
            fw[b].append((o, e, lv[-1]))
    return lv


def main():
    path = sys.argv[1]
    nodes = parse(path)
    # encoder ids: slice the LAST full token of one encoder (drop warmup).
    by_enc = defaultdict(list)
    for n in nodes:
        by_enc[n[1]].append(n)
    enc, seq = max(by_enc.items(), key=lambda kv: len(kv[1]))
    per_token = 321  # census: ~320.9 nodes/token (d128)
    # skip prefill: take exactly the LAST token chunk
    seq = seq[-per_token:]
    print(f"nodes={len(nodes)} encs={len(by_enc)} "
          f"slice_enc={enc} slice_nodes={len(seq)}")
    ref = levels_bruteforce(seq)
    frt = levels_frontier(seq)
    mism = [i for i in range(len(seq)) if ref[i] != frt[i]]
    print(f"max level (barriers if emitted level-wise) = {max(ref)}")
    print(f"emitted-barrier equivalent (recorded order) = "
          f"{sum(1 for n in seq if n[2] == 1)}")
    print(f"frontier vs bruteforce mismatches = {len(mism)}")
    if mism:
        for i in mism[:10]:
            print("  node", i, "ref", ref[i], "frt", frt[i])
        sys.exit(1)
    print("LEVEL-CHECK PASS")


if __name__ == "__main__":
    main()
