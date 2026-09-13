#!/usr/bin/env python3
"""Host-only tests for the explicit H13 ANEC package adapter."""

import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
ADAPTER_PATH = REPO / "overlay/tools/ane-export/h13_package_to_bundle.py"
FIXTURE = REPO / "receipts/fixtures/h13-explicit-chain-add-mul"
SPEC = importlib.util.spec_from_file_location("h13_package_to_bundle", ADAPTER_PATH)
ADAPTER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ADAPTER)

IDENTITY = {
    "name": "h13-first-run-chain-add-mul",
    "graph_hash": "5584d0fd8d40027229890408e924e6f7930cd5f02516466a442193408482ce76",
    "compiler_host_build": "Linux x86_64 explicit ANEC fixture",
    "compiler_toolchain": "mil-hwxc aa688df66cbc2110e0df94f0d50fb72c7fa30a18",
    "source_repo": "joshuaswarren/mlx-omarchy",
    "source_commit": "309cd745d40117b689e8936d3ef62fcea262b232",
    "exported_at": "2026-09-12",
    "model": "h13-first-run-chain-add-mul",
    "model_sha256": "5584d0fd8d40027229890408e924e6f7930cd5f02516466a442193408482ce76",
}


class AdapterTest(unittest.TestCase):
    def test_real_two_program_package_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "bundle"
            manifest = ADAPTER.adapt(FIXTURE, output, IDENTITY)
            self.assertEqual(manifest["manifest_version"], 3)
            self.assertEqual(manifest["driver_abi_major"], 1)
            self.assertEqual(manifest["dispatch_plan"], [0, 1])
            self.assertEqual([p["operation"] for p in manifest["programs"]], ["add", "mul"])
            self.assertEqual(manifest["programs"][0]["inputs"][0]["channel"], 5)
            self.assertEqual(manifest["programs"][1]["outputs"][0]["allocation_bytes"], 16384)
            self.assertEqual(
                manifest["payloads"][1]["sha256"],
                "62595e4a61db24066b83a4f7c077d459c69a6a740109098ee0122b4b46dcd3cc",
            )
            self.assertEqual((output / "program-0.anec").read_bytes(),
                             (FIXTURE / "program-0.anec").read_bytes())

    def test_malformed_allocation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "package"
            shutil.copytree(FIXTURE, package)
            source = json.loads((package / "manifest.json").read_text())
            source["programs"][0]["inputs"][0]["allocationBytes"] = 128
            (package / "manifest.json").write_text(json.dumps(source))
            with self.assertRaisesRegex(ADAPTER.AdapterError, "allocationBytes"):
                ADAPTER.adapt(package, Path(directory) / "bundle", IDENTITY)

    def test_malformed_channel_mapping_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "package"
            shutil.copytree(FIXTURE, package)
            source = json.loads((package / "manifest.json").read_text())
            source["programs"][0]["inputs"][0]["index"] = 6
            source["programs"][0]["inputs"][1]["index"] = 5
            (package / "manifest.json").write_text(json.dumps(source))
            with self.assertRaisesRegex(ADAPTER.AdapterError, "channel mapping"):
                ADAPTER.adapt(package, Path(directory) / "bundle", IDENTITY)

    def test_non_anec_package_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "package"
            shutil.copytree(FIXTURE, package)
            source = json.loads((package / "manifest.json").read_text())
            source["artifactFormat"] = "hwx"
            (package / "manifest.json").write_text(json.dumps(source))
            with self.assertRaisesRegex(ADAPTER.AdapterError, "artifactFormat"):
                ADAPTER.adapt(package, Path(directory) / "bundle", IDENTITY)


if __name__ == "__main__":
    unittest.main()
