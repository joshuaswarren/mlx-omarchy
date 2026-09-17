#!/usr/bin/env python3
"""Compare the live worker's schema identity to the repo file.

Usage:
  scripts/check_schema_identity.py [--url URL]

URL defaults to https://mlx-omarchy-community-data.joshua-s-warren.workers.dev.

Exits non-zero if the live worker reports a different fields_sha256 or
schema_sha256 than the bundled schema. That catches the failure mode
behind the 2026-09-17 ane_port 422: the worker was deployed from a
revision that pre-dated the schema field, so the JSON validator silently
rejected every current-collector submit. Wire it into pre-deploy and
cron health checks.
"""

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = ROOT / "schema" / "payload-v1.schema.json"
DEFAULT_URL = (
    "https://mlx-omarchy-community-data.joshua-s-warren.workers.dev"
)

USER_AGENT = "mlx-omarchy-schema-check/1"


def expected() -> dict:
    raw = SCHEMA_PATH.read_bytes()
    parsed = json.loads(raw)
    fields = sorted(parsed["properties"].keys())
    return {
        "schema_version": parsed["properties"]["schema_version"]["const"],
        "fields_sha256": hashlib.sha256(
            "\n".join(fields).encode()).hexdigest(),
        "schema_sha256": hashlib.sha256(raw).hexdigest(),
    }


def live(url: str) -> dict:
    req = urllib.request.Request(
        f"{url}/v1/schema", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default=DEFAULT_URL,
                    help="worker URL (default: %(default)s)")
    args = ap.parse_args()
    url = args.url
    exp = expected()
    got = live(url)
    mismatches = []
    for key in ("schema_version", "fields_sha256", "schema_sha256"):
        if exp[key] != got.get(key):
            mismatches.append(
                f"  {key}: expected={exp[key]} live={got.get(key)}")
    if mismatches:
        print(
            f"[schema-identity] STALE DEPLOY at {url}\n"
            + "\n".join(mismatches),
            file=sys.stderr,
        )
        return 2
    print(f"[schema-identity] OK live={url} fields_sha256={got['fields_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
