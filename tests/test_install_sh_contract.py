"""Installer refusal, uninstall, and ANE-smoke gate in a disposable HOME."""

import os
import platform
import re
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

INSTALLER = Path(__file__).resolve().parents[1] / "install.sh"
LAUNCHERS = ("mlx-omarchy", "mlx-omarchy-demo", "mlx-omarchy-info")


def installer_text():
    return INSTALLER.read_text()


def extract_ane_smoke():
    match = re.search(r"<<'ANE_SMOKE'[^\n]*\n(.*)\nANE_SMOKE", installer_text(), re.S)
    if match is None:
        raise AssertionError("ANE_SMOKE heredoc missing from install.sh")
    return match.group(1)


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


class AneSmokeGateTests(unittest.TestCase):
    def test_gpu_smoke_always_present(self):
        text = installer_text()
        self.assertIn("import mlx.core as mx", text)
        self.assertIn("say \"Smoke test\"", text)

    def test_ane_smoke_is_character_device_gated(self):
        text = installer_text()
        self.assertIn('[[ -c "${MLX_OMARCHY_ACCEL_DEV:-/dev/accel/accel0}" ]]', text)
        self.assertIn("ANE: unavailable (no /dev/accel/accel0); GPU-only install", text)
        self.assertIn('die "ANE smoke failed"', text)
        self.assertNotIn("omarchy-pkg-add kmod-ane", text)
        self.assertNotIn("pacman -S kmod-ane", text)
        self.assertNotIn("omarchy-pkg-add ane", text)

    def test_uninstall_still_covers_info_launcher(self):
        text = installer_text()
        self.assertIn('"$BIN/mlx-omarchy-info"', text)
        self.assertIn('"$BIN/mlx-omarchy-demo"', text)
        self.assertIn('"$APPS/mlx-omarchy-demo.desktop"', text)

    def test_extracted_ane_smoke_refuses_char_device_without_fdt(self):
        script = extract_ane_smoke()
        with tempfile.TemporaryDirectory() as tmp:
            env = {**os.environ, "MLX_OMARCHY_ACCEL_DEV": "/dev/null",
                   "MLX_OMARCHY_SYSROOT": tmp}
            result = subprocess.run(
                ["python3", "-c", script],
                env=env, capture_output=True, text=True, timeout=30, check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing FDT node or ane module", result.stderr)

    def test_extracted_ane_smoke_passes_composed_fake_tree(self):
        self.assertTrue(stat.S_ISCHR(os.stat("/dev/null", follow_symlinks=False).st_mode))
        script = extract_ane_smoke()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            node = root / "sys/firmware/devicetree/base/ane@26a000000"
            node.mkdir(parents=True)
            (node / "compatible").write_bytes(b"apple,t8103-ane\x00")
            module = root / "sys/module/ane"
            module.mkdir(parents=True)
            (module / "version").write_text("f2a3e5e+lifecycle6\n")
            result = subprocess.run(
                ["python3", "-c", script],
                env={**os.environ, "MLX_OMARCHY_ACCEL_DEV": "/dev/null",
                     "MLX_OMARCHY_SYSROOT": str(root)},
                capture_output=True, text=True, timeout=30, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("ANE smoke OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
