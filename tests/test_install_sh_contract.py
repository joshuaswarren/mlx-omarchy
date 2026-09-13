"""Installer refusal and uninstall boundaries in a disposable HOME."""

import os
import platform
import subprocess
import tempfile
import unittest
from pathlib import Path

INSTALLER = Path(__file__).resolve().parents[1] / "install.sh"
LAUNCHERS = ("mlx-omarchy", "mlx-omarchy-demo", "mlx-omarchy-info")


class InstallerContractTests(unittest.TestCase):
    @unittest.skipIf(Path("/dev/accel/accel0").exists(), "requires a host without ANE")
    def test_ane_install_refuses_before_writing_without_device(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                ["bash", str(INSTALLER), "--ane"],
                env={**os.environ, "HOME": tmp},
                capture_output=True, text=True, timeout=30, check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("ANE installation requires /dev/accel/accel0", result.stderr)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_uninstall_removes_owned_files_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            prefix = home / "custom-prefix"
            binary = home / ".local/bin"
            apps = home / ".local/share/applications"
            for directory in (prefix, binary, apps):
                directory.mkdir(parents=True)
            artifacts = [prefix / "venv", apps / "mlx-omarchy-demo.desktop"]
            artifacts.extend(binary / name for name in LAUNCHERS)
            for artifact in artifacts:
                artifact.touch()
            sentinel = binary / "unrelated"
            sentinel.write_text("preserve")
            result = subprocess.run(
                ["bash", str(INSTALLER), "--uninstall"],
                env={**os.environ, "HOME": str(home), "MLX_OMARCHY_HOME": str(prefix)},
                capture_output=True, text=True, timeout=30, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(prefix.exists())
            for artifact in artifacts:
                self.assertFalse(artifact.exists(), artifact)
            self.assertEqual(sentinel.read_text(), "preserve")

    @unittest.skipIf(platform.machine() == "aarch64", "requires an off-target host")
    def test_refused_install_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            result = subprocess.run(
                ["bash", str(INSTALLER)],
                env={**os.environ, "HOME": tmp, "MLX_OMARCHY_HOME": str(home / "prefix")},
                capture_output=True, text=True, timeout=30, check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(list(home.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
