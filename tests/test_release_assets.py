import hashlib
import runpy
import tempfile
import unittest
import zipfile
from pathlib import Path

VERIFY = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/verify-release-assets.py")
)["verify"]


class ReleaseCommitTests(unittest.TestCase):
    def test_long_abbreviation_distinguishes_shared_short_prefix(self):
        version = "0.32.2+12345678"
        dist_info = f"mlx_omarchy-{version}.dist-info"
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / (
                f"mlx_omarchy-{version}-cp311-cp311-linux_x86_64.whl"
            )
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr(f"{dist_info}/METADATA", f"Version: {version}\n")
                archive.writestr(
                    f"{dist_info}/WHEEL", "Tag: cp311-cp311-linux_x86_64\n"
                )
                archive.writestr("mlx/lib/libmlx.so", b"MLX_DISABLE_COMPILE")
            digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
            matching = "12345678" + "0" * 32
            failures, _, _ = VERIFY(
                wheel, digest, version, "test", matching, matching[:7], False
            )
            self.assertEqual(failures, [])
            different = "12345679" + "0" * 32
            failures, _, _ = VERIFY(
                wheel, digest, version, "test", different, different[:7], False
            )
            self.assertTrue(failures)


if __name__ == "__main__":
    unittest.main()
