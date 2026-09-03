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
        self.assertIn("never", embedded["upload"])
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
        },
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
            "mlx_device", "source_commit", "repo_dirty",
            "redaction_summary", "files",
        ]))

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
