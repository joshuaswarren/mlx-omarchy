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
SCHEMA_PATH = ROOT / "schema" / "payload-v1.schema.json"


def main() -> int:
    raw = SCHEMA_PATH.read_bytes()
    parsed = json.loads(raw)
    fields = sorted(parsed["properties"].keys())
    fields_hash = hashlib.sha256("\n".join(fields).encode()).hexdigest()
    schema_hash = hashlib.sha256(raw).hexdigest()
    out = {
        "schema_version": parsed["properties"]["schema_version"]["const"],
        "fields_sha256": fields_hash,
        "schema_sha256": schema_hash,
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
