# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Host tests for the mlx-omarchy-parakeet downloader CLI (section 13)."""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

try:
    from _bootstrap import _TOOLS  # plain-script execution
except ImportError:  # package discovery (omarchy.coreml.*)
    from ._bootstrap import _TOOLS  # noqa: F401

from coreml.reference import ReferenceLock, default_cache_root, model_cache_dir

_CLI = (
    Path(__file__).resolve().parents[3]
    / "tools"
    / "mlx-omarchy-parakeet"
    / "mlx_omarchy_parakeet.py"
)


class ParakeetCliTest(unittest.TestCase):
    def test_download_on_verifying_cache_emits_true_receipt(self):
        lock = ReferenceLock.load()
        cache_dir = model_cache_dir(
            default_cache_root(), lock.model_repo, lock.model_revision
        )
        if not cache_dir.is_dir():
            self.skipTest("parakeet reference cache not present (offline)")

        run = subprocess.run(
            [sys.executable, str(_CLI), "download", "--json"],
            capture_output=True,
            text=True,
            timeout=600,
        )
        self.assertEqual(run.returncode, 0, run.stderr)
        receipt = json.loads(run.stdout)
        self.assertEqual(
            receipt["schema"], "mlx-omarchy.parakeet-download-receipt.v1"
        )
        self.assertEqual(receipt["model_revision"], lock.model_revision)
        self.assertEqual(
            receipt["reference_commit"], lock.reference_commit
        )
        # Every pinned file reports as verified against its lock hash.
        self.assertTrue(receipt["files"])
        for entry in receipt["files"]:
            self.assertTrue(entry["verified"], entry["path"])
        paths = {entry["path"] for entry in receipt["files"]}
        self.assertIn("tokenizer.json", paths)
        for package in ("encoder", "decoder", "joint"):
            self.assertTrue(
                any(p.startswith(f"{package}.mlpackage/") for p in paths)
            )

    def test_verify_reports_ok(self):
        lock = ReferenceLock.load()
        cache_dir = model_cache_dir(
            default_cache_root(), lock.model_repo, lock.model_revision
        )
        if not cache_dir.is_dir():
            self.skipTest("parakeet reference cache not present (offline)")
        run = subprocess.run(
            [sys.executable, str(_CLI), "verify"],
            capture_output=True,
            text=True,
            timeout=600,
        )
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("verified", run.stdout)


if __name__ == "__main__":
    unittest.main()
