import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class UpstreamSuiteTests(unittest.TestCase):
    def test_mutually_exclusive_phases_cannot_report_success(self):
        runner = Path(__file__).resolve().parents[1] / "tools/run-upstream-suite.sh"
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                ["bash", str(runner), "--cpp-only", "--py-only"],
                env={**os.environ, "OUT_DIR": directory},
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
