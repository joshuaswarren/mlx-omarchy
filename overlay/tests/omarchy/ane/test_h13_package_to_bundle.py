#!/usr/bin/env python3
"""Host-only tests for the explicit H13 ANEC package adapter."""

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
ADAPTER_PATH = REPO / "overlay/tools/ane-export/h13_package_to_bundle.py"
sys.path.insert(0, str(ADAPTER_PATH.parent))
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
            encoded = json.dumps(
                sorted(manifest["payloads"], key=lambda payload: payload["path"]),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
            self.assertEqual(
                manifest["release_asset"]["model_sha256"],
                hashlib.sha256(encoded).hexdigest(),
            )

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

    def test_top_level_offset_is_independent_of_physical_slice_span(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "package"
            shutil.copytree(FIXTURE, package)
            source = json.loads((package / "manifest.json").read_text())
            source["tensors"]["a"]["shape"] = [1, 1024, 1, 1]
            source["tensors"]["a"]["logicalBytes"] = 2048
            binding = source["programs"][0]["inputs"][0]
            binding["shape"] = [1, 384, 1, 1]
            binding["nchw"] = [1, 512, 1, 1, 32, 32]
            binding["logicalBytes"] = 768
            binding["slice"] = {
                "tensor": "a",
                "elementOffset": 512,
                "elementCount": 384,
                "physicalElements": 512,
            }
            (package / "manifest.json").write_text(json.dumps(source))
            manifest = ADAPTER.adapt(package, Path(directory) / "bundle", IDENTITY)
            self.assertEqual(
                manifest["programs"][0]["inputs"][0]["element_offset"], 512)

    def test_output_ranges_require_complete_union_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "package"
            shutil.copytree(FIXTURE, package)
            source = json.loads((package / "manifest.json").read_text())
            source["tensors"]["y"]["shape"] = [1, 128, 1, 1]
            source["tensors"]["y"]["logicalBytes"] = 256
            output = source["programs"][1]["outputs"][0]
            output["nchw"][1] = 128
            output["slice"] = {
                "tensor": "y",
                "elementOffset": 0,
                "elementCount": 64,
                "physicalElements": 128,
            }
            (package / "manifest.json").write_text(json.dumps(source))
            with self.assertRaisesRegex(ADAPTER.AdapterError, "not fully written"):
                ADAPTER.adapt(package, Path(directory) / "gap", IDENTITY)

            source["intermediates"] = []
            del source["tensors"]["sum"]
            for index, program in enumerate(source["programs"]):
                program["inputs"][0]["name"] = "a"
                program["inputs"][0].pop("role", None)
                output = program["outputs"][0]
                output["name"] = "y"
                output.pop("role", None)
                output["nchw"][1] = 128
                output["slice"] = {
                    "tensor": "y",
                    "elementOffset": index * 64,
                    "elementCount": 64,
                    "physicalElements": 128,
                }
            (package / "manifest.json").write_text(json.dumps(source))
            manifest = ADAPTER.adapt(package, Path(directory) / "complete", IDENTITY)
            self.assertEqual(
                [program["outputs"][0]["element_offset"]
                 for program in manifest["programs"]],
                [0, 64],
            )

    def test_generation_receipt_binds_compiler_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "package"
            output = Path(directory) / "bundle"
            shutil.copytree(FIXTURE, package)
            source = json.loads((package / "manifest.json").read_text())
            source["programs"][1]["inputs"][1]["name"] = "a"
            (package / "manifest.json").write_text(json.dumps(source))
            result = subprocess.run(
                [
                    sys.executable,
                    str(ADAPTER_PATH),
                    str(package),
                    "--out-dir", str(output),
                    "--graph-source", str(package / "model.mil"),
                    "--compiler-source", str(REPO),
                    "--name", "mutated-package",
                    "--source-repo", "mil-hwxc",
                    "--source-commit", "aa688df66cbc2110e0df94f0d50fb72c7fa30a18",
                    "--model", "mutated-package",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn(
                "compiler receipt compiler_manifest_sha256 does not match compiler manifest",
                result.stderr,
            )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
