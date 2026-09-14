"""Count final-ISA instruction classes for the coopmat kernel in an
AGX_MESA_DEBUG=shaders dump.

Packed disassembly lines look like "  808: <hex>  <mnemonic> ...".
Programs are separated by an address reset, not by blank lines, because
the dump interleaves pass headers inside one program's listing. The
coopmat kernel is the only shader here that issues simd_matrix_fmadd32.
"""
import collections
import re
import sys

LINE = re.compile(r"^\s*([0-9a-f]+):\s+[0-9a-f]+\s+(\S+)")
CLASSES = (
    "simd_matrix_fmadd32", "wait", "load", "lload", "lstore", "barrier",
    "if", "pop_exec", "iadd", "imadd", "bfeil", "shr", "and", "csel",
    "ffma", "fadd", "u32_to_f", "mov", "ldimm", "stop",
)


def programs(path):
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
    return progs


def main() -> int:
    for path in sys.argv[1:]:
        kernel = None
        for p in programs(path):
            if any(op.startswith("simd_matrix") for _, op in p):
                if kernel is None or len(p) > len(kernel):
                    kernel = p
        if kernel is None:
            print(path, "no matrix program found")
            continue
        counts = collections.Counter(op for _, op in kernel)
        print("==", path)
        print("   packed instructions:", len(kernel))
        for op in CLASSES:
            if counts.get(op):
                print(f"   {op}: {counts[op]}")
        print("   other:", ", ".join(
            f"{op}={n}" for op, n in counts.most_common()
            if op not in CLASSES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
