#!/usr/bin/env python3
"""Per-section qmm kernel ISA extraction (the isa_by_blake approach that
worked): find shader sections whose packed disasm contains the coopmat
madd; save the largest one per arm and print structure counts."""
import re
import sys

D = sys.argv[1]
LINE = re.compile(r"^\s+([0-9a-f]+):\s+([0-9a-f]{4,})\s+([a-z_0-9]+)\s*(.*)$")
for arm in ("base", "hoist", "nodequant"):
    txt = open(f"{D}/dump-{arm}-stdout.log", errors="replace").read()
    sections = txt.split("shader: MESA_SHADER_COMPUTE")[1:]
    best = None
    for sec in sections:
        dis = [m.groups() for m in
               (LINE.match(l) for l in sec.splitlines()) if m]
        madds = sum(1 for a, h, o, r in dis if o == "simd_matrix_fmadd32")
        if madds >= 16 and (best is None or len(dis) > len(best[0])):
            best = (dis, madds)
    if best is None:
        print(arm, "NO QMM SECTION")
        continue
    dis, madds = best
    ops = {}
    for a, h, o, r in dis:
        ops[o] = ops.get(o, 0) + 1
    ints = sum(ops.get(k, 0) for k in
               ["iadd", "and", "shr", "imadd", "bfeil", "shl", "isub", "ior"])
    whiles = sum(v for k, v in ops.items() if k.startswith("while"))
    jmps = sum(v for k, v in ops.items() if k.startswith("jmp"))
    print(f"{arm:10s} n={len(dis)} madds={madds} whiles={whiles} jmps={jmps} "
          f"barriers={ops.get('barrier', 0)} int={ints} ffma={ops.get('ffma', 0)} "
          f"cvt={ops.get('u32_to_f', 0)} lload={ops.get('lload', 0)} "
          f"lstore={ops.get('lstore', 0)} load={ops.get('load', 0)} "
          f"wait={ops.get('wait', 0)}")
    open(f"/tmp/qmm-attr/final2-{arm}.isa", "w").write(
        "\n".join(f"{a}: {h} {o} {r}" for a, h, o, r in dis))
