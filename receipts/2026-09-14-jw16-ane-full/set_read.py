#!/usr/bin/env python3
"""Read-only Apple PMGR SET0..SET5. PROT_READ mmap of /dev/mem, no writes."""
from __future__ import annotations

import json
import mmap
import os
import sys

PAGE = 16384
PHYS = 0x28E08C000
NAMES = ("set0", "base", "set1", "set2", "set3", "set4", "set5")


def main() -> int:
    fd = os.open("/dev/mem", os.O_RDONLY | os.O_SYNC)
    off = PHYS % PAGE
    base = PHYS - off
    length = off + 8 * len(NAMES)
    mm = mmap.mmap(fd, length, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ, offset=base)
    out = {}
    for i, name in enumerate(NAMES):
        o = off + i * 8
        raw = int.from_bytes(mm[o : o + 4], "little")
        rec = {
            "raw": f"0x{raw:08x}",
            "ACTUAL": f"0x{(raw >> 4) & 0xF:x}",
            "TARGET": f"0x{raw & 0xF:x}",
        }
        if i == 0:
            rec["phys"] = hex(PHYS)
        out[name] = rec
    mm.close()
    os.close(fd)
    json.dump(out, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
