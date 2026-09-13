#!/usr/bin/env python3
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""mlx-omarchy-parakeet — pinned Parakeet reference downloader.

Plan section 13: `mlx-omarchy-parakeet download` fetches the pinned
public reference (lock: ``overlay/tools/coreml/parakeet-reference.lock``)
into the shared cache, verifying every file's SHA-256 before and after.
Nothing is trusted on presence or mtime; the receipt (``--json``)
records the exact bytes on disk.
"""

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

_TOOLS = str(Path(__file__).resolve().parents[1])
_COREML = str(Path(__file__).resolve().parents[1] / "coreml")
for _entry in (_TOOLS, _COREML):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from coreml import fetch_parakeet_reference as fetch  # noqa: E402
from coreml.reference import (  # noqa: E402
    ReferenceError,
    ReferenceLock,
    default_cache_root,
    model_cache_dir,
)

RECEIPT_SCHEMA = "mlx-omarchy.parakeet-download-receipt.v1"


def _cache_dir(lock: ReferenceLock) -> Path:
    return model_cache_dir(
        default_cache_root(), lock.model_repo, lock.model_revision
    )


def _receipt(lock: ReferenceLock, cache_dir: Path, elapsed_ms: int) -> dict:
    return {
        "schema": RECEIPT_SCHEMA,
        "model_repo": lock.model_repo,
        "model_revision": lock.model_revision,
        "reference_commit": lock.reference_commit,
        "cache_dir": str(cache_dir),
        "files": [
            {
                "path": f.path,
                "size": f.size,
                "sha256": f.sha256,
                "verified": (cache_dir / f.path).is_file()
                and fetch.sha256_file(cache_dir / f.path) == f.sha256,
            }
            for f in lock.files
        ],
        "audio_fixture": {
            "url": lock.audio.url,
            "sha256": lock.audio.sha256,
            "size": lock.audio.size,
        },
        "elapsed_ms": elapsed_ms,
    }


def _download(args) -> int:
    lock = ReferenceLock.load()
    cache_dir = _cache_dir(lock)
    started = time.monotonic()
    # Human-readable progress goes to stderr so --json stdout stays
    # parseable.
    with contextlib.redirect_stdout(sys.stderr):
        code = fetch.cmd_download(lock, cache_dir, force=args.force)
    elapsed = int((time.monotonic() - started) * 1000)
    if args.json:
        print(json.dumps(_receipt(lock, cache_dir, elapsed), indent=2))
    return code


def _verify(args) -> int:
    lock = ReferenceLock.load()
    return fetch.cmd_verify(lock, _cache_dir(lock))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="mlx-omarchy-parakeet")
    sub = parser.add_subparsers(dest="command", required=True)

    download = sub.add_parser(
        "download", help="fetch and verify the pinned reference"
    )
    download.add_argument(
        "--force", action="store_true", help="re-fetch mismatched files"
    )
    download.add_argument(
        "--json", action="store_true", help="emit a machine-readable receipt"
    )
    download.set_defaults(handler=_download)

    verify = sub.add_parser("verify", help="verify the cache against the lock")
    verify.set_defaults(handler=_verify)

    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReferenceError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
