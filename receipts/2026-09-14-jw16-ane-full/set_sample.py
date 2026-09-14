#!/usr/bin/env python3
"""Tight PROT_READ SET0 sampler. No writes to PMGR."""
from __future__ import annotations

import json
import mmap
import os
import sys
import time
from pathlib import Path

PAGE = 16384
PHYS = 0x28E08C000


def main() -> int:
    out_path = Path(sys.argv[1])
    stop_path = Path(sys.argv[2])
    fd = os.open("/dev/mem", os.O_RDONLY | os.O_SYNC)
    off = PHYS % PAGE
    base = PHYS - off
    mm = mmap.mmap(fd, off + 8, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ, offset=base)
    t0 = time.monotonic()
    samples: list[dict] = []
    last: tuple[int, int, int] | None = None
    last_hb = t0
    try:
        while not stop_path.exists() and len(samples) < 20000:
            raw = int.from_bytes(mm[off : off + 4], "little")
            actual = (raw >> 4) & 0xF
            target = raw & 0xF
            rec = (raw, actual, target)
            now = time.monotonic()
            if rec != last or now - last_hb >= 0.02:
                samples.append(
                    {
                        "t_ms": round((now - t0) * 1000, 3),
                        "set0": {
                            "raw": f"0x{raw:08x}",
                            "ACTUAL": f"0x{actual:x}",
                            "TARGET": f"0x{target:x}",
                            "phys": hex(PHYS),
                        },
                    }
                )
                last = rec
                if now - last_hb >= 0.02:
                    last_hb = now
    finally:
        mm.close()
        os.close(fd)
    out_path.write_text(json.dumps(samples) + "\n")
    os.chmod(out_path, 0o644)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
