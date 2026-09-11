#!/usr/bin/env python3
"""Post-hoc canonical digest gate for the bf16-alpha-fix S1 window.

Reads RECEIPT_DIR/matrix/r{1,2,3}-{fork,stock}-alpha/matrix.json and
verifies every measured leg against the canonical digests: all six Q4
digests (driver-independent) and the per-driver BF16 pins. Writes
digest-gates.json and exits nonzero on any violation.
"""

import json
import sys
from pathlib import Path

CANONICAL = {
    "qwen25-0.5b-4bit:short-decode-32": "7fd25a869ff21678",
    "qwen25-0.5b-4bit:long-decode-128": "4cc08910089477fd",
    "qwen25-0.5b-4bit:longctx-1024-decode-32": "7da83f06ec9f001d",
    "qwen25-0.5b-bf16:short-decode-32": {"fork": "f26175202f3dabe9",
                                          "stock": "7fc0f968789b1882"},
    "qwen25-0.5b-bf16:long-decode-128": {"fork": "8690dc83246b39f8",
                                          "stock": "46108ad71157cb4d"},
    "qwen25-0.5b-bf16:longctx-1024-decode-32": "ff502900d2a179a5",
}


def main(d):
    d = Path(d)
    report = {"cells": {}, "all_held": True}
    for run_dir in sorted((d / "matrix").glob("r*-*-alpha")):
        mj = run_dir / "matrix.json"
        if not mj.exists():
            continue
        label = run_dir.name
        rep, driver = label.split("-", 1)
        driver = driver.removesuffix("-alpha")
        run = json.loads(mj.read_text())
        cell = {}
        for leg in run["legs"]:
            if leg["status"] != "measured":
                cell[leg["leg_id"]] = {"status": leg["status"]}
                report["all_held"] = False
                continue
            got = leg["metrics"]["generated_ids_sha256_16"]
            want = CANONICAL[leg["leg_id"]]
            if isinstance(want, dict):
                want = want[driver]
            held = got == want
            cell[leg["leg_id"]] = {
                "digest": got, "expected": want, "held": held}
            report["all_held"] &= held
        report["cells"][label] = cell
    out = d / "digest-gates.json"
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps({"all_held": report["all_held"],
                      "cells": sorted(report["cells"])}, indent=1))
    return 0 if report["all_held"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
