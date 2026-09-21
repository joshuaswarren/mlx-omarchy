"""Regression test: managed-mode bonsai-2 server releases its reservation
on SIGTERM; SIGKILL-cleanable via the existing owner-token API.

Bug history (Main review):
  _release_reservation(name) called budget.clear_reservation(name) WITHOUT
  the owner token, so the budget API raised "reservation owned by another
  holder" and the exception was swallowed. Result: SIGTERM/kill left a
  dead owner entry on disk; future MLX admission read the orphan and
  under-budgeted MemAvailable.

This test exercises the REAL serve_main subprocess against the CHECKED-IN
REPO/serve source. All imports resolve through REPO/serve and tests/; no
/tmp shadow, no hardcoded staged paths. When the MLX dependency
(mlx_omarchy_serve.budget) is not importable in the test process, the
test skips cleanly with a clear reason -- the test cannot exercise the
real budget API without it.

The pack is the tiny fixture from bonsai2_fixture; the runtime budget
API is the real mlx_omarchy_serve.budget; the lifecycle is the real
serve_main. A failure here means the fix is incomplete.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SERVE = REPO / "serve"
TESTS = Path(__file__).resolve().parent

# Single source of truth: REPO/serve for the checked-in source and
# REPO/tests for the shared fixture. We do NOT honour /tmp/bonsai2-window
# or any other staged shadow here -- CI runs from a clean checkout and
# shadowing would silently test stale code.
for p in (str(SERVE), str(TESTS)):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


# The integrated checkout CONTAINS mlx_omarchy_serve; the import is
# unconditional by Main's no-skip rule -- a missing package is a hard
# collection error, never a silent skip.
import mlx_omarchy_serve  # noqa: F401
from mlx_omarchy_serve import budget  # noqa: F401


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _reservation_path(home: Path) -> Path:
    return home / ".local" / "share" / "mlx-omarchy" / "reservations.json"


def _read_registry(home: Path) -> dict:
    path = _reservation_path(home)
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _wait_for_resident(home: Path, deadline_s: float = 60.0) -> dict | None:
    """Wait for a resident managed entry. The post-load relabel takes
    time after admission -- polling for state == "resident" avoids the
    stale-pending race that masks the relabel."""
    path = _reservation_path(home)
    end = time.time() + deadline_s
    while time.time() < end:
        if path.exists():
            try:
                data = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                data = {}
            for _name, payload in data.items():
                if payload.get("state") == "resident" and payload.get("owner"):
                    return data
        time.sleep(0.2)
    return None


def _wait_for_clear(home: Path, reservation_name: str, deadline_s: float = 30.0) -> bool:
    end = time.time() + deadline_s
    while time.time() < end:
        if reservation_name not in _read_registry(home):
            return True
        time.sleep(0.2)
    return False


def _spawn(python: str, pack_dir: Path, home: Path, port: int, *, allow_cpu: bool = True):
    """Spawn serve_main as a subprocess. Imports resolve through REPO/serve
    only (no /tmp/bonsai2-window or any other staged shadow) so CI runs
    against the checked-in source."""
    argv = [
        python,
        "-c",
        (
            "import sys; sys.path.insert(0, %r); "
            "from mlx_omarchy_bonsai2 import serve_main; "
            "serve_main(sys.argv[1:])" % str(SERVE)
        ),
        "--model", str(pack_dir),
        "--host", "127.0.0.1",
        "--port", str(port),
        "--model-id", "bonsai2-release-test",
        "--max-context", "256",
        "--managed",
    ]
    if allow_cpu:
        argv.append("--allow-cpu")
    env = os.environ.copy()
    # Pin HOME so the test never touches the user's real reservations.json.
    env["HOME"] = str(home)
    env["MLX_OMARCHY_HOME"] = str(home / ".local" / "share" / "mlx-omarchy")
    # Force the subprocess to use REPO/serve for its mlx_omarchy_bonsai2
    # import. Prepend REPO/serve to PYTHONPATH (and remove any inherited
    # path that points at a staged shadow) so the subprocess never
    # resolves through a non-checked-in copy.
    env["PYTHONPATH"] = str(SERVE)
    return subprocess.Popen(
        argv,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )


class ReservationReleaseTests(unittest.TestCase):
    """Real serve_main subprocess + real budget API + real reservations.json."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.home = self.tmp / "home"
        self.home.mkdir()
        # Import bonsai2_fixture lazily -- it imports mlx, which may
        # also be missing on hosts without MLX installed.
        import bonsai2_fixture  # noqa: F401

        self.pack_dir, _ = bonsai2_fixture.build_tiny_pack(self.tmp / "pack")
        self.port = _free_port()
        self._diag = []
        self._saved_env = {
            "HOME": os.environ.get("HOME"),
            "MLX_OMARCHY_HOME": os.environ.get("MLX_OMARCHY_HOME"),
        }
        os.environ["HOME"] = str(self.home)
        os.environ["MLX_OMARCHY_HOME"] = str(self.home / ".local" / "share" / "mlx-omarchy")

    def tearDown(self):
        for child in getattr(self, "_children", []):
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)
        for line in self._diag:
            sys.stderr.write(line)
        shutil.rmtree(self.tmp, ignore_errors=True)
        for k, v in self._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _capture_diagnostics(self, child: subprocess.Popen):
        try:
            out, err = child.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            out, err = b"", b""
        self._diag.append("\n--- child stdout ---\n%s\n" % out.decode("utf-8", "replace"))
        self._diag.append("\n--- child stderr ---\n%s\n" % err.decode("utf-8", "replace"))

    def test_sigterm_releases_owner_entry(self):
        """SIGTERM after healthy serve must clear the owner entry.

        Before the fix, the owner-scoped clear raised "owned by another
        holder" and was silently swallowed, so SIGTERM left a dead entry.
        After the fix, _release_reservation passes the owner token and
        the entry is removed from disk by the in-process finally block.
        """
        child = _spawn(sys.executable, self.pack_dir, self.home, self.port)
        self._children = [child]
        entry = _wait_for_resident(self.home, deadline_s=60)
        if entry is None:
            self._capture_diagnostics(child)
            self.fail("managed server never registered a resident reservation")
        self.assertEqual(
            len(entry), 1, "expected exactly one owner entry, got %r" % entry
        )
        [(reservation_name, payload)] = list(entry.items())
        self.assertTrue(payload.get("owner"), "reservation missing owner token")
        self.assertEqual(payload["state"], "resident")
        self.assertGreater(payload["resident_floor_bytes"], 0)

        # The server is live: hit /health before terminating.
        import urllib.request
        with urllib.request.urlopen(
            "http://127.0.0.1:%d/health" % self.port, timeout=10
        ) as resp:
            self.assertEqual(resp.status, 200)
            body = json.loads(resp.read())
            self.assertEqual(body["model"], "bonsai2-release-test")
            self.assertEqual(body["reservation_name"], reservation_name)
            self.assertGreater(body["reservation_bytes"], 0)

        # The real lifecycle: SIGTERM, then verify the owner entry is gone.
        child.send_signal(signal.SIGTERM)
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=5)
            self._capture_diagnostics(child)
            self.fail("server did not exit after SIGTERM within 30s")

        self.assertTrue(
            _wait_for_clear(self.home, reservation_name, deadline_s=10),
            "reservation %r still in registry after SIGTERM: %r"
            % (reservation_name, _read_registry(self.home)),
        )

    def test_sigkill_owner_scoped_clear_succeeds(self):
        """SIGKILL is not handled by the server; the operator cleanup
        path uses the existing budget.clear_reservation API with the
        captured owner token after dead-PID is verified. NEVER force-
        clear an unknown or live entry."""
        child = _spawn(sys.executable, self.pack_dir, self.home, self.port)
        self._children = [child]
        entry = _wait_for_resident(self.home, deadline_s=60)
        if entry is None:
            self._capture_diagnostics(child)
            self.fail("managed server never registered a resident reservation")
        [(reservation_name, payload)] = list(entry.items())
        owner_token = payload["owner"]
        child.kill()
        child.wait(timeout=10)

        # Dead-pid verify: owner-token prefix encodes pid<NNN>.
        try:
            os.kill(int(owner_token.split("-")[0][3:]), 0)
            self.fail("server pid still alive after kill")
        except ProcessLookupError:
            pass

        # Operator cleanup: EXISTING budget.clear_reservation API with the
        # EXACT preserved owner token. Same call shape the fixed serve_main
        # makes on shutdown; SIGKILL just bypasses the in-process handler.
        cleared = budget.clear_reservation(reservation_name, owner=owner_token)
        self.assertTrue(
            cleared,
            "budget.clear_reservation returned False for known owner",
        )
        self.assertNotIn(
            reservation_name,
            _read_registry(self.home),
            "owner-scoped clear did not remove the entry",
        )


if __name__ == "__main__":
    unittest.main()
