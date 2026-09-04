#!/usr/bin/env python3
"""Focused tests for the contributor collectors.

Covers the four contracts the collectors promise: PII redaction,
deterministic structure, unavailable-tool behavior, and archive integrity,
plus a static guard that the collector entry points import no network
module. Standard library only:

  python3 scripts/test_collect.py
"""

import gzip
import hashlib
import io
import json
import os
import re
import sys
import tarfile
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import collect_common as cc
import collect_quick as cq
import collect_deep as cd


class RedactionStripsPII(unittest.TestCase):
    SAMPLE = (
        "user joshuawarren on host jwm1\n"
        "model path /home/joshuawarren/models/Qwen\n"
        "gateway 192.168.3.66 link-local fe80::1234:56ff:fe78:9abc\n"
        "mac f0:18:98:12:34:56\n"
        "uuid 01234567-89ab-cdef-0123-456789abcdef\n"
        "serial-number: C02XYZ123456\n"
        '  "serial_number": "FVFXC02X"\n'
        "API_KEY=sk-live-abcdef0123456789abcdef\n"
        "token ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ123456\n"
        "Authorization: Bearer eyJhbGciOiJI.eyJzY29wZSIsInN1YiI.sIGN4TuR3\n"
        "safe: Apple M1 (G13G B1) apiVersion 1.4.354 Mesa 26.1.7\n"
    )

    def redactor(self):
        return cc.Redactor(hostname="jwm1", username="joshuawarren",
                           home="/home/joshuawarren")

    def test_no_pii_survives(self):
        out = self.redactor().apply(self.SAMPLE)
        for secret in ("joshuawarren", "jwm1", "/home/", "192.168.3.66",
                       "fe80::", "f0:18:98", "01234567-89ab",
                       "C02XYZ123456", "FVFXC02X", "sk-live-", "ghp_ABCDEF",
                       "eyJhbGciOiJI"):
            self.assertNotIn(secret, out, f"{secret!r} leaked: {out!r}")

    def test_typed_placeholders_appear(self):
        out = self.redactor().apply(self.SAMPLE)
        for mark in ("[user]", "[host]", "[home]", "[redacted-ip4]",
                     "[redacted-ip6]", "[redacted-mac]", "[redacted-uuid]",
                     "[redacted]"):
            self.assertIn(mark, out, f"{mark} missing: {out!r}")

    def test_safe_values_survive(self):
        out = self.redactor().apply(self.SAMPLE)
        for keep in ("Apple M1 (G13G B1)", "1.4.354", "26.1.7"):
            self.assertIn(keep, out, f"safe value {keep!r} was clobbered")

    def test_counts_recorded_per_kind(self):
        red = self.redactor()
        red.apply(self.SAMPLE)
        for kind in ("home_path", "ipv4", "mac", "uuid", "credential",
                     "hostname", "username"):
            self.assertGreaterEqual(red.counts.get(kind, 0), 1, kind)


class QuickReportStructure(unittest.TestCase):
    FAKE = {
        "host": lambda red: {"available": True, "arch": "aarch64",
                             "kernel_release": "7.1.6-1-ARCH"},
        "mesa": lambda red: {"available": True, "gpu": {
            "deviceName": "Apple M1 (G13G B1)",
            "driverName": "Mesa Honeykrisp", "apiVersion": "1.4.354"}},
        "broken": lambda red: (_ for _ in ()).throw(RuntimeError("boom")),
    }

    def test_deterministic_and_complete(self):
        one = cq.collect(probes=dict(self.FAKE))
        two = cq.collect(probes=dict(self.FAKE))
        self.assertEqual(one, two)
        self.assertEqual(cc.dump_json(one), cc.dump_json(two))
        self.assertEqual(one["schema_version"], cc.SCHEMA_VERSION)
        self.assertEqual(one["report"], "mlx-omarchy-quick")
        for section in ("host", "mesa", "mesa_package", "ane", "mlx"):
            self.assertIn(section, one)
        self.assertEqual(one["host"]["gpu"] if "gpu" in one["host"]
                         else one["mesa"]["gpu"]["driverName"],
                         "Mesa Honeykrisp")

    def test_probe_exception_is_recorded_not_raised(self):
        report = cq.collect(probes={"broken": self.FAKE["broken"]})
        self.assertFalse(report["broken"]["available"])
        self.assertIn("RuntimeError", report["broken"]["error"])


class UnavailableToolBehavior(unittest.TestCase):
    def test_missing_binary_recorded(self):
        rec = cc.run_tool(["definitely-not-a-real-tool-xyz"],
                          cc.Redactor(), label="probe")
        self.assertFalse(rec["available"])
        self.assertEqual(rec["error"], "not-found")
        self.assertIsNone(rec["exit_code"])

    def test_real_binary_captured(self):
        rec = cc.run_tool(["echo", "hello"], cc.Redactor(), label="echo")
        self.assertTrue(rec["available"])
        self.assertEqual(rec["exit_code"], 0)
        self.assertEqual(rec["stdout"], "hello")

    def test_deep_section_preserves_unavailability(self):
        with tempfile.TemporaryDirectory() as ws:
            cd.section_child("correctness", ws, os.getcwd())
            with open(os.path.join(ws, "correctness.json")) as fh:
                data = json.load(fh)
        self.assertIn("available", data)
        self.assertFalse(data["available"])
        self.assertTrue(data.get("import_error") or data.get("probe"))


class ArchiveIntegrityAndDeterminism(unittest.TestCase):
    def build_files(self, ws):
        red = cc.Redactor()
        with open(os.path.join(ws, "quick.json"), "wb") as fh:
            fh.write(cc.json_bytes(cq.collect(probes={
                "host": lambda r: {"available": True, "arch": "aarch64"}})))
        for name in ("environment", "correctness", "benchmark", "profile"):
            with open(os.path.join(ws, f"{name}.json"), "wb") as fh:
                fh.write(cc.json_bytes({
                    "available": False, "error": "unavailable",
                    "_redaction": {"ipv4": 2} if name == "profile" else {}}))
        return cd.assemble_files(ws, os.getcwd(), [
            {"zone": "thermal_zone0", "type": "soc", "phase": "start",
             "temp_mc": 40123}])

    def test_two_builds_are_byte_identical(self):
        with tempfile.TemporaryDirectory() as ws:
            files, unavailable, redaction = self.build_files(ws)
            name = "mlx-omarchy-deep.tar.gz"
            m1, data1, _ = cd.finalize(dict(files), unavailable, redaction,
                                       name, os.getcwd())
            m2, data2, _ = cd.finalize(dict(files), unavailable, redaction,
                                       name, os.getcwd())
        self.assertEqual(data1, data2)
        self.assertEqual(hashlib.sha256(data1).hexdigest(),
                         hashlib.sha256(data2).hexdigest())
        self.assertEqual(cc.dump_json(m1), cc.dump_json(m2))

    def test_manifest_hashes_match_members(self):
        with tempfile.TemporaryDirectory() as ws:
            files, unavailable, redaction = self.build_files(ws)
            manifest, data, _ = cd.finalize(dict(files), unavailable,
                                            redaction, "a.tar.gz",
                                            os.getcwd())
        self.assertEqual(manifest["schema_version"], cc.SCHEMA_VERSION)
        listed = {entry["path"]: entry for entry in manifest["files"]}
        self.assertNotIn("manifest.json", listed)
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
            members = {m.name: tf.extractfile(m).read() for m in tf.getmembers()}
        self.assertEqual(set(members),
                         set(listed) | set(cd.SUBMISSION_MEMBERS))
        for path, entry in listed.items():
            self.assertEqual(len(members[path]), entry["bytes"])
            self.assertEqual(hashlib.sha256(members[path]).hexdigest(),
                             entry["sha256"])
            self.assertEqual(members[path], files[path])

    def test_manifest_embedded_in_archive(self):
        with tempfile.TemporaryDirectory() as ws:
            files, unavailable, redaction = self.build_files(ws)
            manifest, data, _ = cd.finalize(dict(files), unavailable,
                                            redaction, "a.tar.gz",
                                            os.getcwd())
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
            embedded = json.load(tf.extractfile("manifest.json"))
        self.assertTrue(embedded["no_network"])
        self.assertIn("correctness", embedded["sections_unavailable"])

    def test_submission_is_paste_ready_and_ingestion_free(self):
        with tempfile.TemporaryDirectory() as ws:
            files, unavailable, redaction = self.build_files(ws)
            work = dict(files)
            manifest, data, _ = cd.finalize(work, unavailable, redaction,
                                            "a.tar.gz", os.getcwd())
        text = work["submission.md"].decode("utf-8")
        self.assertIn("mlx-omarchy hardware report", text)
        self.assertIn("```json", text)
        self.assertNotIn("pull request", text.lower())
        self.assertNotIn("fork", text.lower())
        for entry in manifest["files"]:
            self.assertIn(entry["sha256"], text)
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
            embedded = tf.extractfile("submission.md").read()
        self.assertEqual(embedded, work["submission.md"])


class SubmitProtocol(unittest.TestCase):
    """The chunked resumable v1 wire protocol, against a scripted fake."""

    class FakeResponse:
        def __init__(self, status, payload):
            self.status = status
            self._payload = payload

        def read(self):
            return json.dumps(self._payload).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class FakeServer:
        """Routes by URL shape, records every request and chunk body."""

        def __init__(self, probe=(404, {}), initiate=None, chunk=None,
                     complete=None):
            import collect_submit as cs
            self.probe = probe
            self.initiate = list(initiate or
                                 [(200, {"status": "awaiting_chunks",
                                         "missing_chunks": [0, 1, 2]})])
            self.chunk = chunk or (200, {"status": "stored"})
            self.complete = complete or (200, {"status": "stored",
                                               "receipt_url": "http://r/1"})
            self.requests = []
            self.chunk_bodies = {}
            self.archive = bytes(range(256)) * (cs.CHUNK_BYTES // 256 * 2 + 1)

        def open(self, req, timeout=None):
            url = req.get_full_url()
            method = req.get_method()
            self.requests.append(req)
            if method == "GET":
                status, payload = self.probe
            elif url.endswith("/v1/submit"):
                status, payload = self.initiate.pop(0) if len(
                    self.initiate) > 1 else self.initiate[0]
            elif "/chunk/" in url:
                idx = int(url.rsplit("/", 1)[1])
                self.chunk_bodies[idx] = req.data
                status, payload = self.chunk
                payload = dict(payload, idx=idx)
            elif url.endswith("/complete"):
                status, payload = self.complete
            else:
                raise AssertionError(f"unexpected URL {url}")
            return SubmitProtocol.FakeResponse(status, payload)

    def run_submit(self, server, **kwargs):
        import collect_submit as cs
        payload = kwargs.get("payload", {
            "schema_version": 1,
            "kind": "deep",
            "generated_at": "2026-09-03T12:00:00Z",
        })
        # Production takes a urlopen CALLABLE, not an opener object.
        result = cs.submit("http://endpoint.example", server.archive,
                           payload, urlopen=server.open)
        return result, server

    def initiate_bodies(self, server):
        import collect_submit as cs
        return [json.loads(req.data.decode("utf-8"))
                for req in server.requests
                if req.get_method() == "POST"
                and req.get_full_url().endswith("/v1/submit")]

    def test_full_multi_chunk_upload(self):
        import collect_submit as cs
        result, server = self.run_submit(SubmitProtocol.FakeServer())
        self.assertEqual(-(-len(server.archive) // cs.CHUNK_BYTES), 3)
        self.assertEqual(result["status"], 200)
        self.assertEqual(sorted(server.chunk_bodies), [0, 1, 2])
        expected = dict((idx, piece) for idx, piece, _ in
                        cs.chunk_archive(server.archive))
        for idx, piece in expected.items():
            self.assertEqual(server.chunk_bodies[idx], piece)
        self.assertTrue(server.requests[-1].get_full_url()
                        .endswith("/complete"))

    def test_initiate_carries_pow_and_per_chunk_hashes(self):
        import collect_submit as cs
        _, server = self.run_submit(SubmitProtocol.FakeServer())
        body = self.initiate_bodies(server)[0]
        digest = cs.sha256_hex(server.archive)
        self.assertEqual(body["content_sha256"], digest)
        self.assertEqual(body["schema_version"], 1)
        self.assertEqual(body["kind"], "deep")
        archive = body["archive"]
        self.assertEqual(archive["chunk_count"], 3)
        self.assertEqual(archive["total_bytes"], len(server.archive))
        self.assertEqual(archive["chunk_sha256"],
                         [c[2] for c in cs.chunk_archive(server.archive)])
        pow_token = body["pow"]
        self.assertEqual(pow_token["difficulty"], cs.POW_DIFFICULTY)
        check = hashlib.sha256(
            f"{digest}:{pow_token['nonce']}".encode()).hexdigest()
        self.assertGreaterEqual(cs.leading_zero_bits(check),
                                cs.POW_DIFFICULTY)

    def test_custom_user_agent_on_every_request(self):
        import collect_submit as cs
        _, server = self.run_submit(SubmitProtocol.FakeServer())
        self.assertGreaterEqual(len(server.requests), 5)
        for req in server.requests:
            self.assertEqual(req.headers.get("User-agent"), cs.USER_AGENT)

    def test_resume_sends_only_missing_chunks(self):
        server = SubmitProtocol.FakeServer(
            initiate=[(200, {"status": "awaiting_chunks",
                             "missing_chunks": [1, 2]})])
        self.run_submit(server)
        self.assertEqual(sorted(server.chunk_bodies), [1, 2])

    def test_dedup_probe_short_circuits_before_any_upload(self):
        server = SubmitProtocol.FakeServer(
            probe=(200, {"status": "duplicate",
                         "receipt_url": "http://r/1"}))
        result, server = self.run_submit(server)
        self.assertTrue(result["deduplicated"])
        self.assertEqual(result["url"], "http://r/1")
        self.assertEqual(len(server.requests), 1)

    def test_pow_invalid_bumps_difficulty_and_retries(self):
        server = SubmitProtocol.FakeServer(initiate=[
            (403, {"error": "pow_invalid",
                   "detail": {"min_difficulty": 20}}),
            (200, {"status": "awaiting_chunks", "missing_chunks": [0, 1, 2]}),
        ])
        self.run_submit(server)
        bodies = self.initiate_bodies(server)
        self.assertEqual(bodies[0]["pow"]["difficulty"], 18)
        self.assertEqual(bodies[1]["pow"]["difficulty"], 20)

    def test_chunk_failure_raises_and_mentions_resume(self):
        import collect_submit as cs
        server = SubmitProtocol.FakeServer(
            chunk=(500, {"error": "storage_error"}))
        with self.assertRaises(cs.SubmitError) as ctx:
            self.run_submit(server)
        self.assertIn("resume", str(ctx.exception))

    def test_incomplete_complete_raises(self):
        import collect_submit as cs
        server = SubmitProtocol.FakeServer(
            complete=(409, {"error": "incomplete", "missing_chunks": [2]}))
        with self.assertRaises(cs.SubmitError):
            self.run_submit(server)

    def test_oversize_archive_never_touches_network(self):
        import collect_submit as cs
        server = SubmitProtocol.FakeServer()
        big = b"x" * (cs.MAX_ARCHIVE_BYTES + 1)
        with self.assertRaises(cs.SubmitError):
            cs.submit("http://endpoint.example", big,
                      {"schema_version": 1, "kind": "deep"}, urlopen=server)
        self.assertEqual(server.requests, [])


class PowSolving(unittest.TestCase):
    def test_known_zero_bit_counts(self):
        import collect_submit as cs
        self.assertEqual(cs.leading_zero_bits("00ff"), 8)
        self.assertEqual(cs.leading_zero_bits("0fff"), 4)
        self.assertEqual(cs.leading_zero_bits("10ff"), 3)
        self.assertEqual(cs.leading_zero_bits("ff"), 0)

    def test_solved_nonce_verifies(self):
        import collect_submit as cs
        digest = "b" * 64
        nonce = cs.solve_pow(digest, 12)
        check = hashlib.sha256(f"{digest}:{nonce}".encode()).hexdigest()
        self.assertGreaterEqual(cs.leading_zero_bits(check), 12)


class ChunkArchiving(unittest.TestCase):
    def test_boundaries_and_hashes(self):
        import collect_submit as cs
        data = bytes(range(10))
        chunks = cs.chunk_archive(data, 4)
        self.assertEqual([c[1] for c in chunks], [b"\x00\x01\x02\x03",
                                                  b"\x04\x05\x06\x07",
                                                  b"\x08\x09"])
        self.assertEqual([c[0] for c in chunks], [0, 1, 2])
        for idx, piece, digest in chunks:
            self.assertEqual(digest, hashlib.sha256(piece).hexdigest())


class BuildPayload(unittest.TestCase):
    QUICK = {
        "host": {
            "arch": "aarch64",
            "kernel_release": "6.9.1-asahi",
            "devicetree": {"model": "Apple Mac mini",
                           "compatible": ["apple,t8103", "apple,arm"]},
            "cpu": {"present": 8, "possible": 64, "online": 1,
                    "offline": 7, "hotplug_control": False},
            "boot": {"m1n1_stage2": "v1.5.2",
                     "iboot2": "iBoot-8422.141.2"},
            "cmdline": "root=UUID=[redacted-uuid] quiet",
            "core_shortfall": {"present": 8, "online": 1},
        },
        "ane": {"devicetree": {"node": False, "compatible": None}},
        "mesa": {"gpu": {"driverName": "Asahi Vulkan",
                         "deviceName": "Apple M1"}},
        "mlx": {"distributions": {"mlx-omarchy": "0.3.2"},
                "default_device": "gpu"},
    }
    MANIFEST = {
        "source_commit": "f" * 40,
        "repo_dirty": False,
        "redaction_summary": {"mac": 1},
        "files": [{"path": "quick.json", "bytes": 5, "sha256": "a" * 64,
                   "internal": "dropped"}],
    }

    def test_exact_schema_key_set(self):
        payload = cc.build_payload("deep", self.QUICK, self.MANIFEST)
        self.assertEqual(sorted(payload), sorted([
            "schema_version", "kind", "generated_at", "arch", "model",
            "chip", "kernel", "mesa_driver", "mesa_device", "mlx_version",
            "mlx_device", "source_commit", "repo_dirty", "cpu_online",
            "cpu_present", "hotplug_control", "ane_dt_node",
            "ane_dt_compatible", "boot_chain", "cmdline", "core_shortfall",
            "benchmark", "redaction_summary", "files",
        ]))

    def test_benchmark_rows_ride_in_the_summary(self):
        rows = [{"n": 512, "tflops": 0.157, "median_ms": 1.71,
                 "reps": 8, "min_ms": 1.597}]
        payload = cc.build_payload("deep", self.QUICK, self.MANIFEST,
                                   benchmark=rows)
        self.assertEqual(payload["benchmark"],
                         [{"n": 512, "tflops": 0.157, "median_ms": 1.71}])

    def test_benchmark_defaults_to_empty_and_drops_junk(self):
        self.assertEqual(
            cc.build_payload("quick", self.QUICK, self.MANIFEST)["benchmark"],
            [])
        junk = ["nope", {"tflops": 1.0}, {"n": "512"}]
        self.assertEqual(
            cc.build_payload("deep", self.QUICK, self.MANIFEST,
                             benchmark=junk)["benchmark"], [])

    def test_benchmark_is_capped(self):
        rows = [{"n": i + 1, "tflops": 1.0, "median_ms": 1.0}
                for i in range(40)]
        payload = cc.build_payload("deep", self.QUICK, self.MANIFEST,
                                   benchmark=rows)
        self.assertEqual(len(payload["benchmark"]), 16)

    def test_cpu_online_is_carried(self):
        quick = json.loads(json.dumps(self.QUICK))
        quick["host"]["cpu_online"] = 1
        payload = cc.build_payload("deep", quick, self.MANIFEST)
        self.assertEqual(payload["cpu_online"], 1)
        self.assertIsNone(
            cc.build_payload("deep", self.QUICK, self.MANIFEST)["cpu_online"])

    def test_fleet_gap_fields_are_carried(self):
        quick = json.loads(json.dumps(self.QUICK))
        quick["host"]["cpu_online"] = 1
        payload = cc.build_payload("deep", quick, self.MANIFEST)
        self.assertEqual(payload["cpu_present"], 8)
        self.assertIs(payload["hotplug_control"], False)
        self.assertIs(payload["core_shortfall"], True)
        self.assertIs(payload["ane_dt_node"], False)
        self.assertIsNone(payload["ane_dt_compatible"])
        self.assertEqual(payload["boot_chain"],
                         "iboot2=iBoot-8422.141.2 m1n1_stage2=v1.5.2")
        self.assertEqual(payload["cmdline"],
                         "root=UUID=[redacted-uuid] quiet")

    def test_shortfall_false_when_running_full_core_count(self):
        quick = json.loads(json.dumps(self.QUICK))
        quick["host"]["cpu_online"] = 8
        quick["host"]["cpu"]["online"] = 8
        quick["host"]["core_shortfall"] = None
        payload = cc.build_payload("deep", quick, self.MANIFEST)
        self.assertIs(payload["core_shortfall"], False)

    def test_gap_fields_null_when_report_lacks_them(self):
        payload = cc.build_payload("quick", {}, {})
        self.assertIsNone(payload["cpu_present"])
        self.assertIsNone(payload["hotplug_control"])
        self.assertIsNone(payload["ane_dt_node"])
        self.assertIsNone(payload["ane_dt_compatible"])
        self.assertIsNone(payload["boot_chain"])
        self.assertIsNone(payload["cmdline"])
        self.assertIsNone(payload["core_shortfall"])

    def test_ane_compatible_list_becomes_searchable_blob(self):
        quick = json.loads(json.dumps(self.QUICK))
        quick["ane"]["devicetree"] = {"node": True,
                                      "compatible": ["apple,t8103-ane"]}
        payload = cc.build_payload("deep", quick, self.MANIFEST)
        self.assertIs(payload["ane_dt_node"], True)
        self.assertEqual(payload["ane_dt_compatible"], "apple,t8103-ane")

    def test_values_extracted_from_report(self):
        payload = cc.build_payload("deep", self.QUICK, self.MANIFEST)
        self.assertEqual(payload["chip"], "apple,t8103")
        self.assertEqual(payload["kernel"], "6.9.1-asahi")
        self.assertEqual(payload["mesa_driver"], "Asahi Vulkan")
        self.assertEqual(payload["mlx_version"], "0.3.2")
        self.assertEqual(payload["repo_dirty"], False)

    def test_file_entries_are_trimmed_to_wire_shape(self):
        payload = cc.build_payload("deep", self.QUICK, self.MANIFEST)
        self.assertEqual(payload["files"],
                         [{"path": "quick.json", "bytes": 5,
                           "sha256": "a" * 64}])

    def test_missing_sections_become_none(self):
        payload = cc.build_payload("quick", {}, {})
        self.assertIsNone(payload["chip"])
        self.assertIsNone(payload["mlx_version"])
        self.assertEqual(payload["kind"], "quick")


class CpuTopology(unittest.TestCase):
    @staticmethod
    def _sysfs(tmp, files=(), cpu_dirs=()):
        base = os.path.join(tmp, "cpu")
        os.makedirs(base)
        for name, content in files:
            with open(os.path.join(base, name), "w",
                      encoding="utf-8") as fh:
                fh.write(content)
        for n in cpu_dirs:
            os.makedirs(os.path.join(base, f"cpu{n}"))
        return base

    def test_counts_and_hotplug_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = self._sysfs(tmp, files=[
                ("present", "0-7\n"), ("possible", "0-63\n"),
                ("online", "0-7\n"), ("offline", "\n")],
                cpu_dirs=(0, 1, 3))
            with open(os.path.join(base, "cpu1", "online"), "w",
                      encoding="utf-8"):
                pass
            cpu = cq._cpu_topology(base)
        self.assertEqual(cpu["present"], 8)
        self.assertEqual(cpu["possible"], 64)
        self.assertEqual(cpu["online"], 8)
        self.assertEqual(cpu["offline"], 0)
        self.assertIsNone(cpu["offline_list"])
        self.assertEqual(cpu["present_list"], "0-7")
        self.assertTrue(cpu["hotplug_control"])

    def test_missing_sysfs_is_all_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            cpu = cq._cpu_topology(os.path.join(tmp, "absent"))
        self.assertIsNone(cpu["present"])
        self.assertIsNone(cpu["present_list"])
        self.assertIsNone(cpu["hotplug_control"])

    def test_spin_table_has_no_hotplug_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = self._sysfs(tmp, files=[
                ("present", "0-7\n"), ("possible", "0-7\n"),
                ("online", "0\n"), ("offline", "1-7\n")],
                cpu_dirs=tuple(range(8)))
            cpu = cq._cpu_topology(base)
        self.assertFalse(cpu["hotplug_control"])
        self.assertEqual(cpu["offline"], 7)
        self.assertEqual(cpu["online"], 1)


class BootChainIdentity(unittest.TestCase):
    PROPS = {
        "asahi,m1n1-stage1-version": "v1.5.2\x00",
        "asahi,m1n1-stage2-version": "v1.5.2\x00",
        "asahi,iboot1-version": "iBoot-8422.100.1\x00",
        "asahi,iboot2-version": "iBoot-8422.141.2\x00",
        "asahi,system-fw-version": "iBoot-20712.1.2.0.0\x00",
        "asahi,os-fw-version": "iBoot-24.1.0\x00",
    }

    def test_chosen_properties_are_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            chosen = os.path.join(tmp, "chosen")
            os.makedirs(chosen)
            for name, value in self.PROPS.items():
                with open(os.path.join(chosen, name), "wb") as fh:
                    fh.write(value.encode())
            boot = cq._boot_chain(cc.Redactor(), base=tmp)
        self.assertEqual(boot["m1n1_stage1"], "v1.5.2")
        self.assertEqual(boot["m1n1_stage2"], "v1.5.2")
        self.assertEqual(boot["iboot1"], "iBoot-8422.100.1")
        self.assertEqual(boot["iboot2"], "iBoot-8422.141.2")
        self.assertEqual(boot["system_fw"], "iBoot-20712.1.2.0.0")
        self.assertEqual(boot["os_fw"], "iBoot-24.1.0")

    def test_missing_chosen_is_all_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            boot = cq._boot_chain(cc.Redactor(), base=tmp)
        self.assertEqual(sorted(boot), sorted([
            "m1n1_stage1", "m1n1_stage2", "iboot1", "iboot2",
            "system_fw", "os_fw"]))
        self.assertTrue(all(value is None for value in boot.values()))


class KernelCmdline(unittest.TestCase):
    def test_uuid_and_home_are_redacted(self):
        red = cc.Redactor(username="zoe", hostname="box", home="/home/zoe")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cmdline")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("root=UUID=1b3c9d2e-4f5a-6b7c-8d9e-0f1a2b3c4d5e "
                         "init=/home/zoe/overlay quiet\n")
            out = cq._kernel_cmdline(red, path=path)
        self.assertEqual(out, "root=UUID=[redacted-uuid] init=[home]/overlay quiet")

    def test_missing_cmdline_is_none(self):
        self.assertIsNone(
            cq._kernel_cmdline(cc.Redactor(), path="/no/such/cmdline"))


class CoreShortfall(unittest.TestCase):
    def test_unexplained_gap_is_recorded_with_numbers(self):
        self.assertEqual(
            cq._core_shortfall({"present": 8, "online": 1}, "quiet"),
            {"present": 8, "online": 1})

    def test_full_machine_is_not_flagged(self):
        self.assertIsNone(
            cq._core_shortfall({"present": 8, "online": 8}, ""))

    def test_maxcpus_and_nosmp_explain_the_gap(self):
        self.assertIsNone(cq._core_shortfall(
            {"present": 8, "online": 1}, "maxcpus=1 quiet"))
        self.assertIsNone(cq._core_shortfall(
            {"present": 8, "online": 1}, "nosmp"))

    def test_unknown_counts_are_not_flagged(self):
        self.assertIsNone(cq._core_shortfall({}, "quiet"))


class AneDevicetreeProbe(unittest.TestCase):
    def test_ane_node_is_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            node = os.path.join(tmp, "ane@26a000000")
            os.makedirs(node)
            with open(os.path.join(node, "compatible"), "wb") as fh:
                fh.write(b"apple,t8103-ane\x00apple,ane\x00")
            out = cq._ane_devicetree(tmp)
        self.assertTrue(out["node"])
        self.assertEqual(out["compatible"], ["apple,ane", "apple,t8103-ane"])

    def test_stock_tree_has_no_ane_node(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "cpus"))
            with open(os.path.join(tmp, "compatible"), "wb") as fh:
                fh.write(b"apple,t8103\x00apple,arm-platform\x00")
            out = cq._ane_devicetree(tmp)
        self.assertFalse(out["node"])
        self.assertIsNone(out["compatible"])

    def test_ane_compatible_on_oddly_named_node_is_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            node = os.path.join(tmp, "engine@26a000000")
            os.makedirs(node)
            with open(os.path.join(node, "compatible"), "wb") as fh:
                fh.write(b"apple,t6000-ane\x00")
            out = cq._ane_devicetree(tmp)
        self.assertTrue(out["node"])
        self.assertEqual(out["compatible"], ["apple,t6000-ane"])

    def test_absent_devicetree_is_clean(self):
        out = cq._ane_devicetree("/no/such/tree")
        self.assertFalse(out["node"])
        self.assertIsNone(out["compatible"])


class PayloadSchemaContract(unittest.TestCase):
    """build_payload and the pinned schema must agree on the key set."""

    def test_payload_keys_equal_schema_properties(self):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            os.pardir, "services", "community-data",
                            "schema", "payload-v1.schema.json")
        with open(path, "r", encoding="utf-8") as fh:
            schema = json.load(fh)
        payload = cc.build_payload("quick", {}, {})
        self.assertEqual(sorted(payload), sorted(schema["properties"]))

class SingleNetworkModule(unittest.TestCase):
    def test_only_collect_submit_imports_urllib(self):
        base = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(base, "collect_submit.py")) as fh:
            self.assertIn("import urllib.request", fh.read())


class NoNetworkImports(unittest.TestCase):
    BANNED = re.compile(
        r"^\s*(?:import|from)\s+(?:socket|urllib|http|requests|ftplib|"
        r"smtplib|telnetlib)\b", re.MULTILINE)

    def assert_no_network(self, path):
        with open(path) as fh:
            source = fh.read()
        hits = self.BANNED.findall(source)
        self.assertEqual(hits, [], f"{path} imports a network module: {hits}")

    def test_collectors_have_no_network_imports(self):
        base = os.path.dirname(os.path.abspath(__file__))
        for name in ("collect_quick.py", "collect_deep.py"):
            self.assert_no_network(os.path.join(base, name))

    def test_common_has_no_http_clients(self):
        base = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(base, "collect_common.py")) as fh:
            source = fh.read()
        hits = re.findall(r"^\s*(?:import|from)\s+(?:urllib|http|requests|"
                          r"ftplib|smtplib)\b", source, re.MULTILINE)
        self.assertEqual(hits, [])


class VersionQuadSurvivesRedaction(unittest.TestCase):
    """A dotted version is data; a dotted address is not.

    Measured on jwm1-linux 2026-09-03: `conformanceVersion = 1.4.0.0`
    came back as `[redacted-ip4]`, destroying real Vulkan data in every
    submission from Apple hardware.
    """

    def test_conformance_version_is_kept(self):
        red = cc.Redactor()
        text = "\tconformanceVersion = 1.4.0.0"
        self.assertIn("1.4.0.0", red.apply(text))
        self.assertEqual(red.counts.get("ipv4", 0), 0)

    def test_lowercase_and_colon_version_forms_are_kept(self):
        red = cc.Redactor()
        self.assertIn("10.0.0.1", red.apply("driver version: 10.0.0.1"))
        self.assertEqual(red.counts.get("ipv4", 0), 0)

    def test_real_address_is_still_redacted(self):
        red = cc.Redactor()
        out = red.apply("inet 192.168.3.66 netmask 255.255.254.0")
        self.assertNotIn("192.168.3.66", out)
        self.assertNotIn("255.255.254.0", out)
        self.assertEqual(red.counts.get("ipv4", 0), 2)

    def test_address_on_a_later_line_is_still_redacted(self):
        red = cc.Redactor()
        out = red.apply("conformanceVersion = 1.4.0.0\ninet 10.1.2.3\n")
        self.assertIn("1.4.0.0", out)
        self.assertNotIn("10.1.2.3", out)


    def test_boot_firmware_version_chain_is_kept(self):
        red = cc.Redactor()
        out = red.apply("asahi,system-fw-version=iBoot-20712.1.2.0.0")
        self.assertIn("iBoot-20712.1.2.0.0", out)
        self.assertEqual(red.counts.get("ipv4", 0), 0)

    def test_mid_chain_quad_is_kept_but_bare_quad_is_not(self):
        red = cc.Redactor()
        out = red.apply("fw 20712.1.2.0.0 host at 10.1.2.3")
        self.assertIn("20712.1.2.0.0", out)
        self.assertNotIn("10.1.2.3", out)


class PrimaryGpuSelection(unittest.TestCase):
    """Honeykrisp must win over llvmpipe.

    `vulkaninfo --summary` on an Apple host lists the real GPU as GPU0
    and llvmpipe as GPU1. Keeping the last block reported llvmpipe as
    the machine's GPU, which makes the submission useless for driver
    work. Sample text is verbatim jwm1-linux output.
    """

    SUMMARY = (
        "Devices:\n"
        "========\n"
        "GPU0:\n"
        "\tapiVersion         = 1.4.354\n"
        "\tdriverVersion      = 26.1.7\n"
        "\tdeviceType         = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU\n"
        "\tdeviceName         = Apple M1 (G13G B1)\n"
        "\tdriverID           = DRIVER_ID_MESA_HONEYKRISP\n"
        "\tdriverName         = Honeykrisp\n"
        "\tconformanceVersion = 1.4.0.0\n"
        "GPU1:\n"
        "\tapiVersion         = 1.4.354\n"
        "\tdeviceType         = PHYSICAL_DEVICE_TYPE_CPU\n"
        "\tdeviceName         = llvmpipe (LLVM 22.1.8, 128 bits)\n"
        "\tdriverID           = DRIVER_ID_MESA_LLVMPIPE\n"
        "\tdriverName         = llvmpipe\n"
    )

    def test_both_devices_are_parsed(self):
        devices = cq._device_blocks(self.SUMMARY)
        self.assertEqual(len(devices), 2)
        self.assertEqual(devices[0]["driverName"], "Honeykrisp")
        self.assertEqual(devices[1]["driverName"], "llvmpipe")

    def test_honeykrisp_is_primary(self):
        primary = cq._primary_device(
            cq._device_blocks(self.SUMMARY))
        self.assertEqual(primary["driverName"], "Honeykrisp")
        self.assertEqual(primary["deviceName"], "Apple M1 (G13G B1)")

    def test_non_cpu_wins_when_driver_is_unknown(self):
        devices = [
            {"driverName": "llvmpipe",
             "deviceType": "PHYSICAL_DEVICE_TYPE_CPU"},
            {"driverName": "futurevk",
             "deviceType": "PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU"},
        ]
        self.assertEqual(
            cq._primary_device(devices)["driverName"], "futurevk")

    def test_cpu_only_host_still_reports_something(self):
        devices = [{"driverName": "llvmpipe",
                    "deviceType": "PHYSICAL_DEVICE_TYPE_CPU"}]
        self.assertEqual(
            cq._primary_device(devices)["driverName"], "llvmpipe")
        self.assertEqual(cq._primary_device([]), {})


class SocGrouping(unittest.TestCase):
    """Submissions group by SoC, not by board model."""

    def test_soc_compatible_wins(self):
        quick = {"host": {"devicetree": {
            "model": "Apple MacBook Pro (13-inch, M1, 2020)",
            "compatible": ["apple,j293", "apple,t8103", "apple,arm-platform"],
        }}}
        payload = cc.build_payload("quick", quick, {})
        self.assertEqual(payload["chip"], "apple,t8103")
        self.assertEqual(payload["model"],
                         "Apple MacBook Pro (13-inch, M1, 2020)")

    def test_first_entry_used_when_no_soc_present(self):
        quick = {"host": {"devicetree": {"compatible": ["vendor,board"]}}}
        self.assertEqual(cc.build_payload("quick", quick, {})["chip"],
                         "vendor,board")

    def test_missing_devicetree_is_null_not_an_error(self):
        self.assertIsNone(cc.build_payload("quick", {}, {})["chip"])


class PayloadOnlySubmit(unittest.TestCase):
    """The quick report publishes without an archive, in one round trip."""

    class Fake:
        def __init__(self, probe=(404, {}), initiate=None):
            self.probe = probe
            self.initiate = initiate or (200, {
                "status": "stored", "receipt_url": "http://r/q"})
            self.requests = []

        def open(self, req, timeout=None):
            self.requests.append(req)
            if req.get_method() == "GET":
                status, body = self.probe
            else:
                status, body = self.initiate
            return SubmitProtocol.FakeResponse(status, body)

    PAYLOAD = {"schema_version": 1, "kind": "quick",
               "generated_at": "2026-09-03T16:40:00Z", "chip": "apple,t8103"}

    def test_initiate_sends_null_archive_and_quick_kind(self):
        import collect_submit as cs
        fake = self.Fake()
        receipt = cs.submit_payload("http://e.example", self.PAYLOAD,
                                    urlopen=fake.open)
        posts = [r for r in fake.requests if r.get_method() == "POST"]
        self.assertEqual(len(posts), 1)
        body = json.loads(posts[0].data.decode("utf-8"))
        self.assertIsNone(body["archive"])
        self.assertEqual(body["kind"], "quick")
        self.assertEqual(body["payload"], self.PAYLOAD)
        self.assertIn("nonce", body["pow"])
        self.assertEqual(receipt["url"], "http://r/q")
        self.assertFalse(receipt["deduplicated"])

    def test_content_hash_is_the_canonical_payload(self):
        import collect_submit as cs
        fake = self.Fake()
        cs.submit_payload("http://e.example", self.PAYLOAD, urlopen=fake.open)
        body = json.loads(
            [r for r in fake.requests
             if r.get_method() == "POST"][0].data.decode("utf-8"))
        expected = cs.sha256_hex(json.dumps(
            self.PAYLOAD, sort_keys=True, separators=(",", ":")).encode())
        self.assertEqual(body["content_sha256"], expected)

    def test_dedup_hit_sends_no_post(self):
        import collect_submit as cs
        fake = self.Fake(probe=(200, {"status": "duplicate",
                                      "receipt_url": "http://r/dup"}))
        receipt = cs.submit_payload("http://e.example", self.PAYLOAD,
                                    urlopen=fake.open)
        self.assertTrue(receipt["deduplicated"])
        self.assertEqual([r.get_method() for r in fake.requests], ["GET"])

    def test_oversize_report_refused_before_any_request(self):
        import collect_submit as cs
        fake = self.Fake()
        big = dict(self.PAYLOAD, model="M" * (cs.MAX_PAYLOAD_BYTES + 10))
        with self.assertRaises(cs.SubmitError):
            cs.submit_payload("http://e.example", big, urlopen=fake.open)
        self.assertEqual(fake.requests, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
