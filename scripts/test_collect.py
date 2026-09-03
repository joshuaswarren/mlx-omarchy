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
            m1, data1 = cd.finalize(dict(files), unavailable, redaction,
                                    name, os.getcwd())
            m2, data2 = cd.finalize(dict(files), unavailable, redaction,
                                    name, os.getcwd())
        self.assertEqual(data1, data2)
        self.assertEqual(hashlib.sha256(data1).hexdigest(),
                         hashlib.sha256(data2).hexdigest())
        self.assertEqual(cc.dump_json(m1), cc.dump_json(m2))

    def test_manifest_hashes_match_members(self):
        with tempfile.TemporaryDirectory() as ws:
            files, unavailable, redaction = self.build_files(ws)
            manifest, data = cd.finalize(dict(files), unavailable, redaction,
                                         "a.tar.gz", os.getcwd())
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
            manifest, data = cd.finalize(dict(files), unavailable, redaction,
                                         "a.tar.gz", os.getcwd())
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
            embedded = json.load(tf.extractfile("manifest.json"))
        self.assertTrue(embedded["no_network"])
        self.assertIn("never", embedded["upload"])
        self.assertIn("correctness", embedded["sections_unavailable"])

    def test_submission_is_paste_ready_and_ingestion_free(self):
        with tempfile.TemporaryDirectory() as ws:
            files, unavailable, redaction = self.build_files(ws)
            work = dict(files)
            manifest, data = cd.finalize(work, unavailable, redaction,
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

    class FakeOpener:
        def __init__(self, responses):
            self.responses = list(responses)
            self.requests = []

        def open(self, req, timeout=None):
            self.requests.append(req)
            status, payload = self.responses.pop(0)
            return SubmitProtocol.FakeResponse(status, payload)

    def submit(self, responses, data=b"archive-bytes"):
        import collect_submit as cs
        opener = self.FakeOpener(responses)
        return cs.submit("http://endpoint.example", data, urlopen=opener), \
            opener

    def test_dedup_hit_sends_nothing(self):
        (result, opener), = [self.submit([(200, {"url": "http://r/1"})])]
        self.assertTrue(result["deduplicated"])
        self.assertEqual(result["url"], "http://r/1")
        self.assertEqual(len(opener.requests), 1)

    def test_dedup_miss_posts_redacted_archive(self):
        result, opener = self.submit(
            [(404, {}), (201, {"url": "http://r/2"})])
        self.assertFalse(result["deduplicated"])
        post = opener.requests[1]
        self.assertEqual(post.method, "POST")
        self.assertEqual(post.data, b"archive-bytes")
        self.assertEqual(post.headers.get("Content-sha256"),
                         hashlib.sha256(b"archive-bytes").hexdigest())

    def test_failure_raises_submit_error(self):
        import collect_submit as cs
        with self.assertRaises(cs.SubmitError):
            self.submit([(500, {})])
        with self.assertRaises(cs.SubmitError):
            self.submit([(404, {}), (201, {"missing": "url"})])

    def test_failed_submit_keeps_local_output(self):
        import collect_submit as cs
        with tempfile.TemporaryDirectory() as tmp:
            local = os.path.join(tmp, "a.tar.gz")
            with open(local, "wb") as fh:
                fh.write(b"archive-bytes")
            with self.assertRaises(cs.SubmitError):
                self.submit([(503, {})])
            with open(local, "rb") as fh:
                self.assertEqual(fh.read(), b"archive-bytes")


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
