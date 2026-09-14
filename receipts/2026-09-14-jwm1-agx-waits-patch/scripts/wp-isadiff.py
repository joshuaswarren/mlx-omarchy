#!/usr/bin/env python3
"""Extract the coopmat kernel's final packed ISA from two AGX dumps and
locate exactly where the wait counts differ.

"23 -> 19 waits" is only meaningful if it says WHERE. A wait removed from
the preamble costs nothing per step; a wait removed from the steady-state
k-loop body is the one that could pay.
"""
import re
import sys

LINE = re.compile(r"^\s*([0-9a-f]+):\s+[0-9a-f]+\s+(\S+)(.*)$")


def programs(path):
    progs, current, last = [], [], -1
    for line in open(path, errors="replace"):
        m = LINE.match(line)
        if not m:
            continue
        addr, op, rest = int(m.group(1), 16), m.group(2), m.group(3)
        if addr <= last and current:
            progs.append(current)
            current = []
        current.append((addr, op, rest.strip()))
        last = addr
    if current:
        progs.append(current)
    return progs


def kernel(path):
    best = None
    for p in programs(path):
        if any(op.startswith("simd_matrix") for _, op, _ in p):
            if best is None or len(p) > len(best):
                best = p
    return best


def loop_bounds(k):
    """Steady-state k-loop body: from the last `while` backward-branch
    target region. Use the span containing the matrix ops, delimited by the
    enclosing `while`/`jmp_any` back edge."""
    ops = [op for _, op, _ in k]
    mat = [i for i, op in enumerate(ops) if op.startswith("simd_matrix")]
    lo, hi = mat[0], mat[-1]
    # walk back to the nearest barrier preceding the first matrix op, then
    # back to the preceding `while` (loop head)
    head = max((i for i, op in enumerate(ops[:lo]) if op == "while"), default=0)
    tail = min((i for i, op in enumerate(ops) if i > hi and op == "jmp_any"),
               default=len(ops) - 1)
    return head, tail


def report(path):
    k = kernel(path)
    ops = [op for _, op, _ in k]
    head, tail = loop_bounds(k)
    print("==", path)
    print("   total instructions: %d   total waits: %d"
          % (len(ops), ops.count("wait")))
    print("   loop body [%d..%d] = %d instructions, waits inside: %d"
          % (head, tail, tail - head + 1, ops[head:tail + 1].count("wait")))
    print("   waits before loop head: %d   waits after loop tail: %d"
          % (ops[:head].count("wait"), ops[tail + 1:].count("wait")))
    return k, ops, head, tail


def main():
    a, b = sys.argv[1], sys.argv[2]
    ka, oa, ha, ta = report(a)
    kb, ob, hb, tb = report(b)

    print("\n== per-position opcode diff (aligned by index)")
    if len(oa) == len(ob):
        diffs = [(i, x, y) for i, (x, y) in enumerate(zip(oa, ob)) if x != y]
        print("   %d positions differ out of %d" % (len(diffs), len(oa)))
        for i, x, y in diffs:
            region = ("LOOP" if ha <= i <= ta else
                      "pre" if i < ha else "post")
            ctx = " | ".join(oa[max(0, i - 2):i])
            print("   [%4d] %-5s %-12s -> %-12s   after: %s"
                  % (i, region, x, y, ctx))
    else:
        print("   lengths differ: %d vs %d" % (len(oa), len(ob)))

    print("\n== full opcode order, run-length encoded")
    for label, ops, h, t in ((a, oa, ha, ta), (b, ob, hb, tb)):
        seq = [op for op in ops
               if op in ("load", "wait", "barrier", "lstore", "lload",
                         "while", "jmp_any", "if", "else", "pop_exec")
               or op.startswith("simd_matrix")]
        runs, prev, n = [], None, 0
        for op in seq:
            if op == prev:
                n += 1
            else:
                if prev is not None:
                    runs.append("%s x%d" % (prev, n) if n > 1 else prev)
                prev, n = op, 1
        runs.append("%s x%d" % (prev, n) if n > 1 else prev)
        print("--", label)
        print("   " + " ".join(runs))


if __name__ == "__main__":
    main()
