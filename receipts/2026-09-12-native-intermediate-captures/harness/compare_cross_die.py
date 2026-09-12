#!/usr/bin/env python3
"""Compare regenerated (macstudio) capture arrays against the committed
BULK-MANIFEST.json (16m1mbp origin) file by file.

Classification per file:
  identical   - same sha256 across dies
  absent      - not regenerated on macstudio (e.g. trail dumps that were
                known-broken at capture time)
  different   - present but different bytes (die-dependent or harness
                drift - listed explicitly)
Expects: --manifest BULK-MANIFEST.json, --old <16m1mbp data root>,
         --new <macstudio data root>.
"""

import argparse
import hashlib
import json
import os


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    manifest = json.load(open(args.manifest))["files"]
    result = {"identical": [], "different": {}, "absent": [],
              "new_files": []}
    for rel, meta in sorted(manifest.items()):
        path = os.path.join(args.new, rel)
        if not os.path.isfile(path):
            result["absent"].append(rel)
            continue
        got = sha256(path)
        if got == meta["sha256"]:
            result["identical"].append(rel)
        else:
            result["different"][rel] = {"expected": meta["sha256"],
                                        "got": got}
    for root, _, files in os.walk(args.new):
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), args.new)
            if rel not in manifest:
                result["new_files"].append(rel)
    result["summary"] = {
        "identical": len(result["identical"]),
        "different": len(result["different"]),
        "absent": len(result["absent"]),
        "new_files": len(result["new_files"]),
    }
    json.dump(result, open(args.out, "w"), indent=1)
    print(json.dumps(result["summary"], indent=1))


if __name__ == "__main__":
    main()
