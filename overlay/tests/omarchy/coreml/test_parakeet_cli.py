# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Host tests for the mlx-omarchy-parakeet downloader CLI (section 13)."""

import hashlib
import importlib.machinery
import importlib.util
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


class ParakeetAssetVerifySessionTest(unittest.TestCase):
    """Resident-session asset verification: bytes consumed by a live
    resident worker are verified at its load and re-used without a
    re-hash; any reopen/reload re-hashes the new consumed bytes, so a
    mutated bundle is always caught before it is executed."""

    def _cli(self):
        loader = importlib.machinery.SourceFileLoader(
            "mlx_omarchy_parakeet_under_test", str(_CLI)
        )
        spec = importlib.util.spec_from_loader(
            "mlx_omarchy_parakeet_under_test", loader
        )
        mod = importlib.util.module_from_spec(spec)
        loader.exec_module(mod)
        return mod

    _PAYLOAD = b"\x00\x01immutable-whole-encoder-bytes-for-pin-verification"

    def _make_share(self, root: Path):
        share = root / "share" / "mlx-omarchy" / "parakeet-1"
        bundle = share / "bundles" / "whole"
        bundle.mkdir(parents=True)
        (share / "libane").mkdir()
        payload = self._PAYLOAD
        (bundle / "program-0.anec").write_bytes(payload)
        (share / "libane" / "libane-strict.so").write_bytes(payload)
        mod = self._cli()
        pin = {
            "assets": {
                "bundles": {
                    "whole": {
                        "program-0.anec": mod._sha256_file(
                            bundle / "program-0.anec"
                        )
                    }
                },
                "libane": {
                    "libane-strict.so": mod._sha256_file(
                        share / "libane" / "libane-strict.so"
                    )
                },
            }
        }
        (share / "parakeet-runtime-pin.json").write_text(json.dumps(pin))
        return share, pin, mod

    def test_correct_bytes_pass_and_mutation_is_caught(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            share, pin, mod = self._make_share(root)
            mod._share_dir = lambda: share
            mod._verify_assets(pin)
            payload = (share / "bundles" / "whole" / "program-0.anec")
            mutated = bytearray(payload.read_bytes())
            mutated[0] ^= 0xFF
            payload.write_bytes(bytes(mutated))
            with self.assertRaises(mod.TranscribeRefusal):
                mod._verify_assets(pin)

    def test_resident_reuse_skips_rehash_and_respawn_rehashes(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            share, pin, mod = self._make_share(root)
            mod._share_dir = lambda: share
            pin_key = hashlib.sha256(
                json.dumps(pin, sort_keys=True).encode()
            ).hexdigest()
            calls = {"hash": 0}
            real = mod._sha256_file

            def counting(path):
                calls["hash"] += 1
                return real(path)

            mod._sha256_file = counting
            # Run 1: nothing resident -> verify (hashes the consumed
            # bytes), then the pipeline records the now-live resident
            # session that consumed those verified bytes.
            self.assertTrue(mod._asset_verify_needed(None, pin_key))
            mod._verify_assets(pin)
            mod._ASSET_VERIFY_STATE = (("ident", 4242), pin_key)
            self.assertFalse(mod._asset_verify_needed(("ident", 4242), pin_key))
            first = calls["hash"]
            self.assertGreater(first, 0)
            self.assertFalse(
                mod._asset_verify_needed(("ident", 4242), pin_key)
            )
            self.assertEqual(calls["hash"], first)  # resident reuse: no re-hash
            self.assertTrue(
                mod._asset_verify_needed(("ident", 4243), pin_key)
            )  # worker respawn re-verifies the consumed bytes
            self.assertTrue(mod._asset_verify_needed(("other", 1), pin_key))
