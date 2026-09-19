#!/usr/bin/env python3
"""Print the schema identity embedded in src/routes.ts as SCHEMA_IDENTITY.

Run after any change to schema/payload-v1.schema.json. Copy the printed
JSON into the SCHEMA_IDENTITY constant. The check script
(scripts/check_schema_identity.py) compares the live worker against the
repo file so a stale deploy is caught at the wire instead of silently
422-ing every submission that carries a field the worker does not know
about.
"""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATHS = sorted((ROOT / "schema").glob("payload-v1*.schema.json"))


def main() -> int:
    raws = [p.read_bytes() for p in SCHEMA_PATHS]
    fields = sorted(set().union(
        *(json.loads(raw)["properties"].keys() for raw in raws)))
    fields_hash = hashlib.sha256("\n".join(fields).encode()).hexdigest()
    schema_hash = hashlib.sha256(b"".join(raws)).hexdigest()
    out = {
        "schema_version": json.loads(raws[0])["properties"]["schema_version"]["const"],
        "fields_sha256": fields_hash,
        "schema_sha256": schema_hash,
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
