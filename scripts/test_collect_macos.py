#!/usr/bin/env python3
"""Offline macOS collector contracts, runnable on Linux without MLX."""

import base64
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch, Mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import collect_common as cc
import collect_deep as cd
import collect_macos as cm
import collect_quick as cq
import mlx_provenance as prov


class MacHostTests(unittest.TestCase):
    def test_hardware_does_not_require_mlx_and_keeps_only_selected_fields(self):
        facts = {"machine": "arm64", "chip": "Apple M3", "cores": "8",
                 "memsize_bytes": 24 * 1024**3, "os": "macOS 15.0 (24A335)",
                 "gpu": {"chipset": "Apple M3", "gpu_cores": "10", "metal": "Metal 3"},
                 "serial_number": "DO-NOT-KEEP"}
        with patch.object(cm.bench_matrix, "host_facts", return_value=facts), \
                patch.object(cm, "run_tool", side_effect=[
                    {"stdout": "Mac15,12", "exit_code": 0},
                    {"stdout": "8", "exit_code": 0}]):
            host = cm.probe_host(cc.Redactor())
        self.assertEqual(host["model"], "Mac15,12")
        self.assertEqual(host["memory_total_mib"], 24576)
        self.assertEqual(host["cpu_online"], 8)
        self.assertEqual(host["cpu"]["present"], 8)
        self.assertEqual(host["gpu"]["gpu_cores"], "10")
        self.assertNotIn("DO-NOT-KEEP", json.dumps(host))
        self.assertNotIn("devicetree", host)

    def test_missing_system_tools_leave_unknown_values(self):
        with patch.object(cm.bench_matrix, "host_facts", return_value={"machine": "arm64"}), \
                patch.object(cm, "run_tool", return_value={"stdout": "", "exit_code": None}):
            host = cm.probe_host(cc.Redactor())
        self.assertTrue(host["available"])
        for key in ("model", "chip", "memory_total_mib", "cpu_online", "gpu"):
            self.assertIsNone(host[key])

    def test_linux_diagnostics_are_not_run_on_mac(self):
        with patch("platform.system", return_value="Darwin"), \
                patch.object(cq, "run_tool", side_effect=AssertionError("Linux command")):
            for probe in (cq.probe_mesa, cq.probe_mesa_package, cq.probe_ane):
                self.assertEqual(probe(cc.Redactor()), cm.not_applicable())
            self.assertEqual(cd.section_profile(cc.Redactor(), "/missing", "/missing"),
                             cm.not_applicable())

    def test_context_excludes_process_names_ids_and_raw_power_output(self):
        with patch.object(cm.bench_matrix, "power_state", return_value={
                "raw": "private battery identity", "source": "Battery", "percent": 76,
                "charging": False}), patch.object(cm.bench_matrix, "clean_check", return_value={
                    "status": "contended", "scanned": 400,
                    "matched": [{"pid": "123", "comm": "private-service"}]}):
            context = cm.measurement_context()
        self.assertEqual(context["model_processes"]["matched_count"], 1)
        self.assertEqual(context["power"]["percent"], 76)
        for secret in ("private", "123", "comm", "raw"):
            self.assertNotIn(secret, json.dumps(context))

    def test_native_payload_uses_existing_schema_and_null_linux_fields(self):
        quick = {"host": {"system": "Darwin", "model": "Mac14,2", "chip": "Apple M2",
                          "kernel_release": "25.6.0", "os": "macOS 26.6.2 (25G83)",
                          "cpu_online": 8, "cpu": {"present": 8}},
                 "mlx": {"mlx_version": "0.32.1", "metal_available": True,
                         "default_device": "Device(gpu, 0)"}}
        payload = cc.build_payload("deep", quick, {}, benchmark=[{"n": 256, "median_ms": 1.2}])
        schema = json.loads((Path(cd.REPO) / "services/community-data/schema/payload-v1.schema.json").read_text())
        self.assertEqual(set(payload), set(schema["properties"]))
        self.assertEqual(payload["model"], "Mac14,2")
        self.assertEqual(payload["chip"], "Apple M2")
        self.assertIn("macOS", payload["kernel"])
        self.assertIn("Metal", payload["mlx_device"])
        self.assertEqual(payload["benchmark"][0]["median_ms"], 1.2)
        for key in ("mesa_driver", "mesa_device", "ane_dt_node", "ane_dt_compatible",
                    "boot_chain", "cmdline", "hotplug_control"):
            self.assertIsNone(payload[key], key)
        text = cd.build_submission({}, {"quick.json": cc.json_bytes(quick)}, "mac.tar.gz")
        self.assertIn("Native MLX: 0.32.1", text)
        self.assertIn("does not prove Linux support", text)
        self.assertNotIn("mlx-omarchy: not installed", text)

    def test_nested_observations_are_redacted_before_writing(self):
        red = cc.Redactor(hostname="private-host", username="private-user", home="/Users/private-user")
        with tempfile.TemporaryDirectory() as ws, patch.object(cd, "Redactor", return_value=red), \
                patch.object(cd, "section_environment", return_value={
                    "available": True, "nested": [{"path": "/Users/private-user/data", "host": "private-host"}]}):
            cd.section_child("environment", ws, cd.REPO)
            output = (Path(ws) / "environment.json").read_text()
        self.assertNotIn("private-user", output)
        self.assertNotIn("private-host", output)
        self.assertIn("[home]/data", output)


class NativeProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.site = Path(self.tmp.name).resolve()
        self.extension = self.site / "core.cpython-314-darwin.so"
        self.library = self.site / "libmlx.dylib"
        self.extension.write_bytes(b"extension")
        self.library.write_bytes(b"library")
        self.mx = types.ModuleType("mlx.core")
        self.mx.__version__ = "0.32.1"
        self.mx.metal = Mock()
        self.mx.metal.is_available.return_value = True
        self.mx.gpu = "gpu"
        self.mx.set_default_device = Mock()
        mlx = types.ModuleType("mlx")
        mlx.core = self.mx
        patcher = patch.dict(sys.modules, {"mlx": mlx, "mlx.core": self.mx})
        patcher.start()
        self.addCleanup(patcher.stop)
        binaries = patch.object(prov, "_loaded_macos_binaries", return_value=[self.extension, self.library])
        binaries.start()
        self.addCleanup(binaries.stop)

    def distribution(self, files, version="0.32.1"):
        entries = []
        for path in files:
            # PackagePath carries the same RECORD metadata as importlib.metadata.
            from importlib.metadata import PackagePath, FileHash
            entry = PackagePath(path.name)
            digest = base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest()).decode().rstrip("=")
            entry.hash = FileHash("sha256=" + digest)
            entries.append(entry)
        return types.SimpleNamespace(version=version, files=entries,
                                     locate_file=lambda name: self.site / name,
                                     read_text=lambda name: "pip" if name == "INSTALLER" else "record")

    def inspect(self, distributions):
        def lookup(name):
            if name not in distributions:
                raise prov.importlib.metadata.PackageNotFoundError(name)
            return distributions[name]
        with patch.object(prov.importlib.metadata, "distribution", side_effect=lookup):
            return prov.native_provenance()

    def test_split_mlx_and_metal_wheels_verify_loaded_binaries(self):
        result = self.inspect({"mlx": self.distribution([self.extension]),
                               "mlx-metal": self.distribution([self.library])})
        self.assertEqual(result["verified"], "match")
        self.assertEqual(len(result["files"]), 2)
        self.assertNotIn(str(self.site), json.dumps(result))

    def test_homebrew_without_record_keeps_fingerprints_unverified(self):
        dist = self.distribution([])
        dist.read_text = lambda name: "brew" if name == "INSTALLER" else None
        result = self.inspect({"mlx": dist})
        self.assertEqual(result["verified"], "unverified")
        self.assertEqual(result["packages"]["mlx"]["installer"], "brew")
        self.assertFalse(result["packages"]["mlx"]["record_available"])
        self.assertEqual(len(result["files"]), 2)

    def test_hash_mismatch_cannot_be_hidden_by_later_unverified_file(self):
        dist = self.distribution([self.extension])
        self.extension.write_bytes(b"changed extension")
        result = self.inspect({"mlx": dist})
        self.assertEqual(result["verified"], "mismatch")
        self.assertIn(self.extension.name, result["mismatch"])

    def test_compiled_version_mismatch_refuses(self):
        result = self.inspect({"mlx": self.distribution([], version="0.30.0")})
        self.assertEqual(result["verified"], "mismatch")

    def test_source_install_has_hashes_but_no_verification_claim(self):
        result = self.inspect({})
        self.assertEqual(result["verified"], "unverified")
        self.assertEqual(len(result["files"]), 2)

    def test_probe_requires_metal_and_selects_gpu(self):
        with patch("platform.system", return_value="Darwin"), \
                patch.object(prov, "native_provenance", return_value={"verified": "unverified"}), \
                patch.object(prov, "provenance_line", return_value="provenance: test"), \
                patch("sys.stderr", new=io.StringIO()):
            cd.prepare_probe(self.mx)
            self.mx.set_default_device.assert_called_once_with("gpu")
            self.mx.metal.is_available.return_value = False
            with self.assertRaisesRegex(RuntimeError, "Metal GPU unavailable"):
                cd.prepare_probe(self.mx)

    def test_both_probes_refuse_mismatch_before_tensor_work(self):
        self.mx.default_device = lambda: "Device(gpu, 0)"
        for source, result_key in ((cd.CORRECTNESS_PROBE, "ops"), (cd.BENCH_PROBE, "matmul")):
            with self.subTest(probe=result_key), \
                    patch.object(cd, "prepare_probe", return_value={
                        "verified": "mismatch", "mismatch": "changed library"}), \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                with self.assertRaises(SystemExit):
                    exec(source.replace("__SCRIPTS_DIR__", repr(cd.SCRIPTS_DIR)), {})
                result = json.loads(output.getvalue())
            self.assertFalse(result["available"])
            self.assertEqual(result[result_key], [])
            self.assertIn("changed library", result["error"])

    def test_linux_provenance_gate_is_unchanged(self):
        with patch("platform.system", return_value="Linux"), \
                patch.object(prov, "installed_provenance", return_value={"verified": "mismatch"}) as installed, \
                patch.object(prov, "native_provenance") as native, \
                patch.object(prov, "provenance_line", return_value="provenance: test"), \
                patch("sys.stderr", new=io.StringIO()):
            self.assertEqual(cd.prepare_probe(self.mx)["verified"], "mismatch")
        installed.assert_called_once_with()
        native.assert_not_called()
        self.mx.set_default_device.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
