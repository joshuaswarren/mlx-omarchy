"""End-to-end bootstrap: install.sh sections -> prefix layout -> launcher -> CLI.

This is not an argparse test. It executes the real serve-installation text
extracted from install.sh (the same heredocs and copy logic an Omarchy
machine runs), materializes the prefix layout under a fake HOME, creates a
real venv the way the installer does, and then runs the installed
mlx-omarchy-serve / omarchy-mlx-serve launchers as child processes.

Only divergence from a real install: the release-tag curl fetches are
replaced by local copies of the same files (tests never touch the network,
and the release tag for this commit does not exist while the branch is
unmerged). Everything after the fetch — package placement, launcher
generation, omarchy command-center registration, venv exec — runs verbatim.
"""

import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALLER = REPO_ROOT / "install.sh"

PACKAGE_FILES = ("__init__.py", "catalog.py", "budget.py", "__main__.py",
                 "catalog.json")
LAYA_FILES = ("__init__.py", "model.py", "sequence.py", "api.py",
              "server.py", "convert.py", "qualify.py")
BONSAI2_FILES = ("__init__.py", "packed.py", "loader.py", "server.py")


def installer_text():
    return INSTALLER.read_text()


def serve_install_section():
    """The real '5b/5c' installation text, with the network fetch swapped
    for a local copy of the identical files."""
    text = installer_text()
    start = text.index('# 5b. Serve CLI')
    end = text.index('if command -v omarchy-launch-floating-terminal-with-presentation')
    section = text[start:end]

    lines = section.splitlines(keepends=True)
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        if "_file in " in stripped and stripped.startswith("for "):
            # The release-tag loop fetches one file per iteration; the local
            # copy expands to one cp per file. Skip through the closing done.
            if stripped.startswith("for serve_file in"):
                pkg, var, names = "mlx_omarchy_serve", "SERVE_PKG", PACKAGE_FILES
            elif stripped.startswith("for laya_file in"):
                pkg, var, names = "mlx_omarchy_laya", "LAYA_PKG", LAYA_FILES
            else:
                pkg, var, names = "mlx_omarchy_bonsai2", "BONSAI2_PKG", BONSAI2_FILES
            for name in names:
                src = REPO_ROOT / "serve" / pkg / name
                out.append(f'cp "{src}" "${var}/{name}"\n')
            while lines[i].strip() != "done":
                i += 1
            i += 1
            continue
        out.append(line)
        i += 1
    return "".join(out)


class BootstrapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        home = Path(cls.tmp.name)
        cls.home = home
        prefix = home / ".local/share/mlx-omarchy"
        bin_dir = home / ".local/bin"
        bin_dir.mkdir(parents=True)
        omarchy_bin = bin_dir / "fake-omarchy-bin"
        omarchy_bin.mkdir()
        (omarchy_bin / "omarchy").write_text("#!/bin/sh\nexit 0\n")
        (omarchy_bin / "omarchy").chmod(0o755)
        env = {**os.environ,
               "HOME": str(home),
               "PATH": f"{omarchy_bin}:{os.environ['PATH']}",
               "MLX_OMARCHY_HOME": str(prefix),
               "REPO": "local.test/does-not-matter",
               "VERSION": "test"}
        script = (
            'say() { printf "== %s\\n" "$*"; }\n'
            'die() { echo "error: $*" >&2; exit 1; }\n'
            f'PREFIX="{prefix}"\n'
            f'VENV="{prefix}/venv"\n'
            f'BIN="{bin_dir}"\n'
            + serve_install_section()
        )
        result = subprocess.run(["bash", "-c", script], env=env,
                                capture_output=True, text=True, timeout=120)
        cls.install_output = result.stdout + result.stderr
        if result.returncode != 0:
            raise AssertionError(f"install section failed:\n{cls.install_output}")
        subprocess.run([sys.executable, "-m", "venv", "--without-pip",
                        str(prefix / "venv")], check=True, timeout=120)
        cls.prefix = prefix
        cls.bin_dir = bin_dir
        cls.omarchy_bin = omarchy_bin

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def launcher_env(self):
        return {**os.environ,
                "HOME": str(self.home),
                "MLX_OMARCHY_OFFLINE": "1"}

    def run_launcher(self, launcher, *args, cwd=None):
        return subprocess.run([str(launcher), *args], env=self.launcher_env(),
                              capture_output=True, text=True, timeout=120,
                              cwd=cwd)

    def test_package_lands_in_prefix(self):
        pkg = self.prefix / "mlx_omarchy_serve"
        for name in PACKAGE_FILES:
            self.assertTrue((pkg / name).is_file(), name)
        # Sibling packages ship through the same installer section but may
        # not have merged into this branch yet; when the source exists in
        # the repo, the install must have copied it.
        for pkg_name, files in (("mlx_omarchy_laya", LAYA_FILES),
                                ("mlx_omarchy_bonsai2", BONSAI2_FILES)):
            src_dir = REPO_ROOT / "serve" / pkg_name
            if not src_dir.is_dir():
                continue
            for name in files:
                self.assertTrue((self.prefix / pkg_name / name).is_file(),
                                f"{pkg_name}/{name}")

    def test_launchers_are_executable(self):
        # mlx-omarchy-serve always lands in ~/.local/bin; the command-center
        # binary lands next to the omarchy binary (see the registration test).
        path = self.bin_dir / "mlx-omarchy-serve"
        self.assertTrue(path.is_file())
        self.assertTrue(os.access(path, os.X_OK))
        registered = self.omarchy_bin / "omarchy-mlx-serve"
        self.assertTrue(registered.is_file())
        self.assertTrue(stat.S_IMODE(registered.stat().st_mode) & stat.S_IXUSR)

    def test_registered_binary_lives_beside_omarchy_with_metadata(self):
        registered = self.omarchy_bin / "omarchy-mlx-serve"
        body = registered.read_text()
        self.assertIn("# omarchy:group=mlx", body)
        self.assertIn("# omarchy:name=serve", body)
        self.assertIn("mlx_omarchy_serve", body)

    def test_installed_entrypoint_serves_the_bundled_catalog(self):
        result = self.run_launcher(self.bin_dir / "mlx-omarchy-serve",
                                   "catalog", "list")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("qwen3.8-27b-4bit", result.stdout)

    def test_installed_entrypoint_recommends_and_plans(self):
        result = self.run_launcher(self.bin_dir / "mlx-omarchy-serve", "recommend")
        self.assertEqual(result.returncode, 0, result.stderr)
        # The bundled seed is honest: generation-qualified, HTTP-untested —
        # so recommend lists it but the auto-serve gate holds it back.
        self.assertIn("qwen3.8-27b-4bit", result.stdout)
        self.assertIn("pick: none", result.stdout)
        self.assertIn("http serving unqualified", result.stdout)
        result = self.run_launcher(self.bin_dir / "mlx-omarchy-serve",
                                   "plan", "qwen3.8-27b-4bit")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("FITS", result.stdout)
        self.assertIn("http serving unqualified", result.stderr)  # explicit warning

    def test_installed_entrypoint_enforces_the_kv_context_gate(self):
        result = self.run_launcher(self.bin_dir / "mlx-omarchy-serve",
                                   "plan", "qwen3.8-27b-4bit",
                                   "--context", "65536")
        self.assertEqual(result.returncode, 2)
        self.assertIn("KV-per-token unknown", result.stderr)

    def test_omarchy_route_launcher_reaches_the_same_cli(self):
        result = self.run_launcher(self.omarchy_bin / "omarchy-mlx-serve",
                                   "catalog", "status")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("bundled fallback", result.stdout)

    def test_refusal_path_fails_closed_offline(self):
        # Offline (env set by the launcher test env) + nothing complete on
        # disk: the gate must refuse rather than download.
        result = self.run_launcher(self.bin_dir / "mlx-omarchy-serve",
                                   "serve", "qwen3.8-27b-4bit", "--yes")
        self.assertEqual(result.returncode, 1)
        self.assertIn("offline", result.stderr)
        self.assertIn("refusing to download", result.stderr)


if __name__ == "__main__":
    unittest.main()
