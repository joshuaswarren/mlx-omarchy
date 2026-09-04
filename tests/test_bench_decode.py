"""Tests for bench_decode.py: exact generated-ID recording and error exits.

Stdlib unittest only, no network, no mlx import: the runner's identity
and validation paths run without a GPU.
"""

import json
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import bench_decode  # noqa: E402

SCRIPT = REPO_ROOT / "scripts" / "bench_decode.py"


class IdsDigestTests(unittest.TestCase):
    def test_digest_covers_exact_ids_in_order(self):
        ids = [5, 9214, 0, 151643, 7]
        self.assertEqual(
            bench_decode.ids_digest(ids),
            bench_decode.ids_digest(list(ids)))

    def test_digest_changes_when_any_id_changes(self):
        ids = [10, 20, 30]
        self.assertNotEqual(bench_decode.ids_digest(ids),
                            bench_decode.ids_digest([10, 21, 30]))
        self.assertNotEqual(bench_decode.ids_digest(ids),
                            bench_decode.ids_digest(ids[::-1]))

    def test_digest_is_short_and_ascii_stable(self):
        d = bench_decode.ids_digest([1])
        self.assertEqual(len(d), 16)
        int(d, 16)  # hex


class RunGenerationTests(unittest.TestCase):
    def test_collects_response_token_attribute(self):
        class Resp:
            def __init__(self, token):
                self.token = token

        ids = []
        times, n = bench_decode.run_generation(
            (Resp(i) for i in (3, 1, 4)), ids)
        self.assertEqual((n, len(times)), (3, 3))
        self.assertEqual(ids, [3, 1, 4])

    def test_no_ids_list_still_works(self):
        times, n = bench_decode.run_generation(iter((1, 2)))
        self.assertEqual(n, 2)


class ReportTests(unittest.TestCase):
    def test_short_burst_exits_nonzero_without_rate(self):
        with self.assertRaises(SystemExit) as cm:
            bench_decode.report(prefill_ns=0, token_times=[1, 2, 3],
                                requested=32)
        self.assertEqual(cm.exception.code, 1)

    def test_pinned_run_prints_counted_rate_and_id_lines(self):
        import io
        from unittest.mock import patch

        ids = [7, 8, 9, 10]
        times = [1000, 2000, 3000, 4000]
        with patch("sys.stdout", new=io.StringIO()) as out:
            tps = bench_decode.report(500, times, 4, ids)
        text = out.getvalue()
        self.assertIn("over 3 tokens", text)
        self.assertIn(f"generated_ids sha256:{bench_decode.ids_digest(ids)}",
                      text)
        result = json.loads(text.strip().splitlines()[-1])
        self.assertEqual(result["engine"], "bench_decode")
        self.assertEqual(result["generated"], 4)
        self.assertEqual(result["ids_sha256_16"], bench_decode.ids_digest(ids))
        self.assertEqual(result["ids_first"], [7, 8, 9])
        self.assertEqual(result["ids_last"], [8, 9, 10])
        self.assertAlmostEqual(tps, 1_000_000.0)


class CliExitTests(unittest.TestCase):
    def test_self_test_passes_without_mlx(self):
        r = subprocess.run([sys.executable, str(SCRIPT), "--self-test"],
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("self-test: OK", r.stdout)

    def test_missing_model_exits_2(self):
        r = subprocess.run([sys.executable, str(SCRIPT), "--prompt", "p",
                            "--tokens", "8"],
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 2)

    def test_zero_tokens_exits_2(self):
        r = subprocess.run([sys.executable, str(SCRIPT), "--model", "m",
                            "--prompt", "p", "--tokens", "1"],
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 2)

    def test_missing_wheel_file_exits_2(self):
        r = subprocess.run([sys.executable, str(SCRIPT), "--model", "m",
                            "--prompt", "p", "--tokens", "8",
                            "--wheel", "/nonexistent/run.whl"],
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 2)
        self.assertIn("does not exist", r.stderr)


if __name__ == "__main__":
    unittest.main()
