"""Tests for the community-data query CLI and the mirror workflow.

Stdlib unittest only, no network. Fixtures are built in a temp
directory and shaped like the real collector payloads.
"""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import query_community_data as query  # noqa: E402
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "community-data.yml"


SHA_DEEP_M1 = "b7e34c02" + "a" * 56
SHA_QUICK_M2 = "1c9d" + "f" * 28 + "e" * 32
SHA_PARTIAL = "9a03" + "7" * 60
SHA_FUTURE = "5d" * 32

MALFORMED_LINE = '{"content_sha256": "oops", no closing brace'


def deep_m1_record():
    return {
        "schema_version": 1,
        "kind": "deep",
        "content_sha256": SHA_DEEP_M1,
        "payload": {
            "report": "mlx-omarchy-deep",
            "host": {"available": True, "arch": "aarch64",
                     "kernel_release": "7.1.6-1-ARCH",
                     "devicetree": {"model": "Apple M1"}},
            "mesa": {"available": True, "gpu": {
                "deviceName": "Apple M1 (G13G B1)",
                "driverName": "Mesa Honeykrisp",
                "apiVersion": "1.4.354"}},
            "mesa_package": {"pacman": {
                "available": True, "exit_code": 0,
                "stdout": "mesa 1:25.2.3-1\n"}},
            "mlx": {"available": True, "mlx_version": "0.29.3",
                    "distributions": {"mlx-omarchy": "0.3.2"}},
            "benchmark": {"available": True, "matmul": [
                {"n": 256, "reps": 8, "median_ms": 0.512, "min_ms": 0.500,
                 "max_ms": 0.600, "tflops": 65.536},
                {"n": 512, "reps": 8, "median_ms": 3.0, "min_ms": 2.9,
                 "max_ms": 3.4, "tflops": 89.4784},
                {"n": 1024, "reps": 8, "median_ms": 22.0, "min_ms": 21.5,
                 "max_ms": 24.0, "tflops": 97.6129}],
            },
        },
    }


def quick_m2_record():
    return {
        "schema_version": 1,
        "kind": "quick",
        "content_sha256": SHA_QUICK_M2,
        "payload": {
            "report": "mlx-omarchy-quick",
            "host": {"available": True, "arch": "aarch64",
                     "kernel_release": "7.2.0-1-ARCH",
                     "devicetree": {"model": "Apple M2"}},
            "mesa": {"available": True, "gpu": {
                "deviceName": "Apple M2 (G14G)",
                "driverName": "Mesa Honeykrisp",
                "apiVersion": "1.4.400"}},
            "mlx": {"available": True,
                    "distributions": {"mlx-omarchy": "0.3.1"}},
        },
    }


def partial_record():
    """Older schema: flat fields, no devicetree, no benchmark."""
    return {
        "schema_version": 0,
        "kind": "quick",
        "content_sha256": SHA_PARTIAL,
        "chip": "Apple M1 Pro",
        "kernel": "6.9.1-1-ARCH",
        "payload": {"mlx_omarchy_version": "0.2.0"},
    }


def future_record():
    """Unknown schema_version with an empty payload must still load."""
    return {
        "schema_version": 99,
        "kind": "deep",
        "content_sha256": SHA_FUTURE,
        "payload": {},
    }


def snapshot_bytes():
    lines = [json.dumps(deep_m1_record()),
             json.dumps(quick_m2_record()),
             MALFORMED_LINE,
             json.dumps(partial_record()),
             "",
             json.dumps(future_record())]
    return ("\n".join(lines) + "\n").encode("utf-8")


class SnapshotTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.snapshot = Path(self._tmp.name)
        (self.snapshot / "latest.jsonl").write_bytes(snapshot_bytes())
        (self.snapshot / "index.json").write_text(json.dumps(
            {"generated_at": "2026-09-02T12:00:00Z",
             "schema_version": 1, "count": 4, "truncated": False}),
            encoding="utf-8")

    def run_cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), \
                contextlib.redirect_stderr(err):
            rc = query.main(list(argv))
        return rc, out.getvalue(), err.getvalue()

    def json_list(self, *extra):
        rc, out, err = self.run_cli("--source", "local",
                                    "--snapshot", str(self.snapshot),
                                    "--json", *extra, "list")
        self.assertEqual(rc, 0, err)
        return json.loads(out)


class QueryCliTests(SnapshotTestCase):
    def test_local_snapshot_parses_and_skips_malformed(self):
        data = self.json_list()
        self.assertEqual(data["count"], 4)
        self.assertEqual(data["skipped_malformed"], 1)
        shas = [r["content_sha256"] for r in data["results"]]
        self.assertEqual(shas, [SHA_DEEP_M1, SHA_QUICK_M2,
                                SHA_PARTIAL, SHA_FUTURE])

    def test_human_list_has_one_line_per_record(self):
        rc, out, err = self.run_cli("--source", "local",
                                    "--snapshot", str(self.snapshot),
                                    "list")
        self.assertEqual(rc, 0)
        lines = [x for x in out.splitlines() if x.strip()]
        self.assertEqual(len(lines), 4)
        self.assertIn(SHA_DEEP_M1[:12], lines[0])
        self.assertIn("1 malformed line(s) skipped", err)

    def test_filter_kind(self):
        self.assertEqual(
            {r["content_sha256"] for r in self.json_list(
                "--kind", "quick")["results"]},
            {SHA_QUICK_M2, SHA_PARTIAL})
        self.assertEqual(
            {r["content_sha256"] for r in self.json_list(
                "--kind", "deep")["results"]},
            {SHA_DEEP_M1, SHA_FUTURE})

    def test_filter_chip(self):
        self.assertEqual(
            {r["content_sha256"] for r in self.json_list(
                "--chip", "M1")["results"]},
            {SHA_DEEP_M1, SHA_PARTIAL})
        self.assertEqual(
            {r["content_sha256"] for r in self.json_list(
                "--chip", "M1 Pro")["results"]},
            {SHA_PARTIAL})
        self.assertEqual(
            {r["content_sha256"] for r in self.json_list(
                "--chip", "M2")["results"]},
            {SHA_QUICK_M2})

    def test_filter_kernel(self):
        self.assertEqual(
            {r["content_sha256"] for r in self.json_list(
                "--kernel", "7.1.6")["results"]},
            {SHA_DEEP_M1})
        self.assertEqual(
            {r["content_sha256"] for r in self.json_list(
                "--kernel", "6.9.1")["results"]},
            {SHA_PARTIAL})

    def test_filter_mesa(self):
        self.assertEqual(
            {r["content_sha256"] for r in self.json_list(
                "--mesa", "25.2.3")["results"]},
            {SHA_DEEP_M1})
        self.assertEqual(
            {r["content_sha256"] for r in self.json_list(
                "--mesa", "honeykrisp")["results"]},
            {SHA_DEEP_M1, SHA_QUICK_M2})

    def test_filter_mlx_version(self):
        self.assertEqual(
            {r["content_sha256"] for r in self.json_list(
                "--mlx-version", "0.3.1")["results"]},
            {SHA_QUICK_M2})
        self.assertEqual(
            {r["content_sha256"] for r in self.json_list(
                "--mlx-version", "0.2.0")["results"]},
            {SHA_PARTIAL})

    def test_filters_combine(self):
        self.assertEqual(
            {r["content_sha256"] for r in self.json_list(
                "--kind", "deep", "--chip", "M1")["results"]},
            {SHA_DEEP_M1})

    def test_no_match_is_clean_zero_exit(self):
        rc, out, err = self.run_cli("--source", "local",
                                    "--snapshot", str(self.snapshot),
                                    "--chip", "Ryzen", "list")
        self.assertEqual(rc, 0)
        self.assertIn("no records match", out)

    def test_compare_tflops_aggregates_across_sizes(self):
        rc, out, _ = self.run_cli("--source", "local",
                                  "--snapshot", str(self.snapshot),
                                  "--json", "compare", "--metric", "tflops")
        self.assertEqual(rc, 0)
        data = json.loads(out)
        self.assertEqual(data["metric"], "tflops")
        self.assertEqual(data["groups"], [{
            "device": "Apple M1", "count": 3,
            "min": 65.536, "median": 89.4784, "max": 97.6129}])

    def test_compare_by_size(self):
        rc, out, _ = self.run_cli("--source", "local",
                                  "--snapshot", str(self.snapshot),
                                  "--json", "compare",
                                  "--metric", "tflops", "--size", "512")
        self.assertEqual(rc, 0)
        data = json.loads(out)
        self.assertEqual(data["size"], 512)
        self.assertEqual(data["groups"][0]["count"], 1)
        self.assertEqual(data["groups"][0]["median"], 89.4784)

    def test_compare_median_ms(self):
        rc, out, _ = self.run_cli("--source", "local",
                                  "--snapshot", str(self.snapshot),
                                  "--json", "compare",
                                  "--metric", "median_ms")
        self.assertEqual(rc, 0)
        group = json.loads(out)["groups"][0]
        self.assertEqual(group["min"], 0.512)
        self.assertEqual(group["median"], 3.0)
        self.assertEqual(group["max"], 22.0)

    def test_compare_without_data(self):
        rc, out, _ = self.run_cli("--source", "local",
                                  "--snapshot", str(self.snapshot),
                                  "--chip", "M2",
                                  "compare", "--metric", "tflops")
        self.assertEqual(rc, 0)
        self.assertIn("no tflops benchmark data", out)

    def test_show_by_hash_prefix(self):
        rc, out, _ = self.run_cli("--source", "local",
                                  "--snapshot", str(self.snapshot),
                                  "show", SHA_DEEP_M1[:8])
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(out)["content_sha256"], SHA_DEEP_M1)

    def test_show_unknown_prefix_errors_cleanly(self):
        rc, out, err = self.run_cli("--source", "local",
                                    "--snapshot", str(self.snapshot),
                                    "show", "ffffffff")
        self.assertEqual(rc, 1)
        self.assertIn("no record with hash prefix", err)
        self.assertNotIn("Traceback", err)

    def test_missing_dataset_message(self):
        missing = self.snapshot / "does-not-exist"
        rc, out, err = self.run_cli("--source", "local",
                                    "--snapshot", str(missing), "list")
        self.assertEqual(rc, 1)
        self.assertIn("no local dataset", err)
        self.assertIn("--snapshot", err)
        self.assertNotIn("Traceback", err)

    def test_auto_uses_local_snapshot_without_network(self):
        rc, out, _ = self.run_cli("--snapshot", str(self.snapshot),
                                  "--json", "list")
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(out)["count"], 4)

    def test_unknown_schema_version_tolerated(self):
        data = self.json_list()
        future = [r for r in data["results"]
                  if r["schema_version"] == 99]
        self.assertEqual(len(future), 1)
        rc, out, err = self.run_cli("--source", "local",
                                    "--snapshot", str(self.snapshot),
                                    "--json", "compare")
        self.assertEqual(rc, 0)  # future record simply has no benchmarks

    def test_partial_record_extracts_flat_fields(self):
        rc, out, _ = self.run_cli("--source", "local",
                                  "--snapshot", str(self.snapshot),
                                  "--json", "list")
        records = {r["content_sha256"]: r
                   for r in json.loads(out)["results"]}
        self.assertEqual(query.record_chip(records[SHA_PARTIAL]),
                         "Apple M1 Pro")
        self.assertEqual(query.record_kernel(records[SHA_PARTIAL]),
                         "6.9.1-1-ARCH")
        self.assertEqual(query.record_mlx_version(records[SHA_PARTIAL]),
                         "0.2.0")


try:
    import yaml
except ImportError:
    yaml = None


@unittest.skipUnless(yaml is not None, "PyYAML not installed")
class WorkflowYamlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = WORKFLOW.read_text(encoding="utf-8")
        cls.data = yaml.safe_load(cls.text)

    def test_workflow_parses_with_expected_triggers(self):
        triggers = self.data.get("on", self.data.get(True))
        self.assertIsInstance(triggers, dict)
        self.assertIn("17 3 * * *", triggers["schedule"][0]["cron"])
        self.assertIn("workflow_dispatch", triggers)

    def test_minimum_permissions(self):
        self.assertEqual(self.data["permissions"], {"contents": "read"})
        job = self.data["jobs"]["mirror"]
        self.assertEqual(job["permissions"], {"contents": "write"})
        self.assertEqual(job["runs-on"], "ubuntu-latest")

    def test_commits_as_bot_to_data_branch_without_secrets(self):
        self.assertIn("github-actions[bot]", self.text)
        self.assertIn("community-data", self.text)
        self.assertNotIn("${{ secrets.", self.text)

    def test_endpoint_default_matches_cli_default(self):
        self.assertIn(query.DEFAULT_BASE_URL, self.text)
        self.assertIn("vars.COMMUNITY_DATA_BASE_URL", self.text)

    def test_mirror_script_is_referenced(self):
        self.assertIn("scripts/mirror_community_data.py", self.text)


if __name__ == "__main__":
    unittest.main()
