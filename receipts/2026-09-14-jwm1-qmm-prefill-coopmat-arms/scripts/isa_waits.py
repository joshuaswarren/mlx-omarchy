"""Report where the AGX scheduler placed scoreboard waits in the coopmat
kernel: distance from each global load to the next wait, and whether a
wait sits immediately before a barrier (the H2 discriminator from
receipts/2026-09-12-agx-qmm-codegen).
"""
import re
import sys

LINE = re.compile(r"^\s*([0-9a-f]+):\s+[0-9a-f]+\s+(\S+)")


def kernel(path):
    progs, current, last = [], [], -1
    for line in open(path, errors="replace"):
        m = LINE.match(line)
        if not m:
            continue
        addr, op = int(m.group(1), 16), m.group(2)
        if addr <= last and current:
            progs.append(current)
            current = []
        current.append((addr, op))
        last = addr
    if current:
        progs.append(current)
    best = None
    for p in progs:
        if any(o.startswith("simd_matrix") for _, o in p):
            if best is None or len(p) > len(best):
                best = p
    return best


def main() -> int:
    for path in sys.argv[1:]:
        k = kernel(path)
        ops = [op for _, op in k]
        print("==", path, f"({len(ops)} instructions)")
        gaps = []
        for i, op in enumerate(ops):
            if op != "load":
                continue
            nxt = next((j for j in range(i + 1, len(ops))
                        if ops[j] == "wait"), None)
            gaps.append("none" if nxt is None else str(nxt - i))
        print("   load -> next wait, instruction gaps:", ", ".join(gaps))
        before_barrier = sum(
            1 for i, op in enumerate(ops)
            if op == "wait" and "barrier" in ops[i + 1:i + 3])
        print("   waits within 2 instructions before a barrier:",
              before_barrier, "of", ops.count("wait"))
        matrix = [i for i, op in enumerate(ops) if op.startswith("simd_matrix")
                  or op == "<unknown"]
        if matrix:
            lo, hi = matrix[0], matrix[-1]
            window = ops[lo:hi + 1]
            print("   matrix window:", hi - lo + 1, "instructions,",
                  "waits inside:", window.count("wait"),
                  "lloads inside:", window.count("lload"))
        seq = [op for op in ops
               if op in ("load", "wait", "barrier", "lstore", "lload")
               or op.startswith("simd_matrix") or op == "<unknown"]
        runs, prev, n = [], None, 0
        for op in seq:
            if op == prev:
                n += 1
            else:
                if prev is not None:
                    runs.append(f"{prev}x{n}" if n > 1 else prev)
                prev, n = op, 1
        runs.append(f"{prev}x{n}" if n > 1 else prev)
        print("   order:", " ".join(runs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
