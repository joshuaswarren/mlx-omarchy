"""Serve CLI: approval gate, admission refusal, launch argv, catalog commands."""

import contextlib
import unittest.mock
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "serve"))

from mlx_omarchy_serve import budget, catalog  # noqa: E402
from mlx_omarchy_serve import __main__ as serve_cli  # noqa: E402

GiB = 1024**3


def fixture_entry(**over):
    base = {
        "id": "test-chat-4b",
        "repo": "mlx-community/Test-Chat-4b",
        "revision": "b" * 40,
        "kind": "chat",
        "license": "Apache-2.0",
        "family": "test-family",
        "priority": 1,
        "quant": {"bits": 4, "group_size": 64, "mode": "affine"},
        "memory": {"weights_bytes": int(4.0 * GiB), "kv_bytes_per_token": 64 * 1024,
                   "peak_estimate_bytes": None},
        "context": {"max_tokens": 8192},
        "capability": {"arch": None, "min_mem_gib": None},
        "qualification": {
            "generation": {"status": "qualified", "receipt": "r.md", "date": "2026-09-20"},
            "http": {"status": "untested", "receipt": None, "date": None},
        },
        "recommended": True,
        "serve": {"backend": "mlx-lm", "module": None},
        "availability": {"size_bytes": int(4.0 * GiB), "refreshed_at": None},
    }
    base.update(over)
    return base


FIXTURE = {
    "version": 1,
    "generated_at": "2026-09-20T00:00:00Z",
    "source": "https://example.invalid/catalog.json",
    "models": [fixture_entry()],
}


class CliTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name)
        self._patchers = [
            unittest.mock.patch.object(catalog, "refresh",
                                       lambda *a, **k: {"status": "skipped", "path": "", "detail": ""}),
            unittest.mock.patch.object(catalog, "load_catalog", lambda *a, **k: FIXTURE),
            unittest.mock.patch.object(budget, "mem_available", lambda: int(16 * GiB)),
            unittest.mock.patch.object(budget, "default_home", lambda: self.home),
        ]
        for patcher in self._patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()

    def run_cli(self, argv):
        out, err = contextlib.redirect_stdout(self.stdout), contextlib.redirect_stderr(self.stderr)
        with out, err:
            code = serve_cli.main(argv)
        return code, self.stdout.getvalue(), self.stderr.getvalue()


class RecommendTests(CliTestBase):
    def test_recommends_fitting_curated_entry(self):
        code, out, _ = self.run_cli(["recommend"])
        self.assertEqual(code, 0)
        self.assertIn("pick: test-chat-4b", out)

    def test_no_fit_when_memory_small(self):
        with unittest.mock.patch.object(budget, "mem_available", lambda: int(5 * GiB)):
            code, out, _ = self.run_cli(["recommend"])
        self.assertEqual(code, 0)
        self.assertIn("none", out)


class PlanTests(CliTestBase):
    def test_plan_by_catalog_id_downloads_nothing(self):
        code, out, _ = self.run_cli(["plan", "test-chat-4b"])
        self.assertEqual(code, 0)
        self.assertIn("FITS", out)
        self.assertIn("kv cache:     0.50 GiB at 8192 tokens", out)

    def test_plan_over_context_limit_refuses(self):
        code, _, err = self.run_cli(["plan", "test-chat-4b", "--context", "90000"])
        self.assertEqual(code, 2)
        self.assertIn("exceeds this model's limit", err)

    def test_kv_unknown_model_refuses_long_context(self):
        FIXTURE["models"].append(fixture_entry(
            id="no-kv-facts", repo="mlx-community/NoKv", recommended=False,
            priority=2,
            memory={"weights_bytes": int(2 * GiB), "kv_bytes_per_token": None,
                    "peak_estimate_bytes": None},
            context={"max_tokens": None}))
        code, _, err = self.run_cli(["plan", "no-kv-facts", "--context", "8192"])
        self.assertEqual(code, 2)
        self.assertIn("KV-per-token unknown", err)
        code, out, _ = self.run_cli(["plan", "no-kv-facts"])
        self.assertEqual(code, 0)
        self.assertIn("UNKNOWN", out)
        FIXTURE["models"].pop()

    def test_plan_off_catalog_repo_requires_weights_gib(self):
        code, _, err = self.run_cli(["plan", "someone/Some-Model"])
        self.assertEqual(code, 2)
        self.assertIn("--weights-gib", err)
        code, out, _ = self.run_cli(["plan", "someone/Some-Model", "--weights-gib", "9"])
        self.assertEqual(code, 0)
        self.assertIn("9.00 GiB", out)
        self.assertIn("UNKNOWN", out)

    def test_plan_disk_shortfall_marked(self):
        with unittest.mock.patch.object(budget, "disk_free", lambda _p: 0):
            code, out, _ = self.run_cli(["plan", "test-chat-4b"])
        self.assertEqual(code, 0)
        self.assertIn("INSUFFICIENT", out)

    def test_no_fit_plan_refuses_on_serve_only(self):
        with unittest.mock.patch.object(budget, "mem_available", lambda: int(5 * GiB)):
            code, _, err = self.run_cli(["serve", "test-chat-4b", "--yes"])
        self.assertEqual(code, 1)
        self.assertIn("DOES NOT FIT", err)


class ServeApprovalTests(CliTestBase):
    def test_noninteractive_without_yes_refuses(self):
        with unittest.mock.patch.object(serve_cli, "needs_download", lambda _r: True):
            code, _, err = self.run_cli(["serve", "test-chat-4b"])
        self.assertEqual(code, 1)
        self.assertIn("noninteractive", err)

    def test_yes_without_explicit_target_refuses(self):
        with unittest.mock.patch.object(serve_cli, "needs_download", lambda _r: True):
            code, _, err = self.run_cli(["serve", "--yes"])
        self.assertEqual(code, 1)
        self.assertIn("explicitly named target", err)

    def test_interactive_rejection_aborts(self):
        inputs = iter(["no\n"])
        with unittest.mock.patch.object(serve_cli, "needs_download", lambda _r: True), \
             unittest.mock.patch.object(serve_cli.sys.stdin, "isatty", lambda: True), \
             unittest.mock.patch("builtins.input", lambda *_: next(inputs)):
            code, _, err = self.run_cli(["serve", "test-chat-4b"])
        self.assertEqual(code, 1)

    def test_interactive_yes_downloads_and_launches(self):
        seen = {}

        def fake_download(resolved):
            seen["downloaded"] = resolved.repo
            model_dir = Path(self.tmp.name) / "model"
            model_dir.mkdir(exist_ok=True)
            return model_dir

        def fake_run(argv, **_kw):
            seen["argv"] = argv
            return unittest.mock.Mock(returncode=0)

        with unittest.mock.patch.object(serve_cli, "needs_download", lambda _r: True), \
             unittest.mock.patch.object(serve_cli, "download_snapshot", fake_download), \
             unittest.mock.patch.object(serve_cli.subprocess, "run", fake_run), \
             unittest.mock.patch.object(serve_cli.sys.stdin, "isatty", lambda: True), \
             unittest.mock.patch("builtins.input", lambda *_: "yes\n"):
            code, _, _ = self.run_cli(["serve", "test-chat-4b"])
        self.assertEqual(code, 0)
        self.assertEqual(seen["downloaded"], "mlx-community/Test-Chat-4b")
        self.assertIn("-m", seen["argv"])
        self.assertIn("mlx_lm.server", seen["argv"])
        self.assertIn("127.0.0.1", seen["argv"])
        self.assertNotIn("trust_remote_code", " ".join(seen["argv"]))


class ServeLaunchTests(CliTestBase):
    def serve_local(self, extra=None):
        model_dir = Path(self.tmp.name) / "local-model"
        model_dir.mkdir(exist_ok=True)
        seen = {}

        def fake_run(argv, **_kw):
            seen["argv"] = argv
            return unittest.mock.Mock(returncode=0)

        patchers = [
            unittest.mock.patch.object(serve_cli, "needs_download", lambda _r: False),
            unittest.mock.patch.object(serve_cli, "download_snapshot", lambda _r: model_dir),
            unittest.mock.patch.object(serve_cli.subprocess, "run", fake_run),
        ]
        for patcher in patchers:
            patcher.start()
        try:
            code, out, err = self.run_cli(["serve", str(model_dir), "--weights-gib", "3", *(extra or [])])
        finally:
            for patcher in patchers:
                patcher.stop()
        return code, seen, out, err

    def test_local_model_launches_mlx_lm(self):
        code, seen, out, err = self.serve_local()
        self.assertEqual(code, 0)
        self.assertIn("mlx_lm.server", seen["argv"])

    def test_nonloopback_warns(self):
        code, seen, out, err = self.serve_local(["--host", "0.0.0.0"])
        self.assertEqual(code, 0)
        self.assertIn("no authentication", err)

    def test_omlx_missing_is_honest_error(self):
        with unittest.mock.patch.object(serve_cli.importlib.util, "find_spec",
                                        lambda _n: None):
            code, _, _, err = self.serve_local(["--server", "omlx"])
        self.assertEqual(code, 3)
        self.assertIn("omlx is not installed", err)

    def test_omlx_argv_uses_model_dir_flag(self):
        fake_spec = object()

        def find_spec(name):
            return fake_spec if name == "omlx" else None

        model_dir = Path(self.tmp.name) / "local-model"
        model_dir.mkdir(exist_ok=True)
        seen = {}

        def fake_run(argv, **_kw):
            seen["argv"] = argv
            return unittest.mock.Mock(returncode=0)

        with unittest.mock.patch.object(serve_cli, "needs_download", lambda _r: False), \
             unittest.mock.patch.object(serve_cli, "download_snapshot", lambda _r: model_dir), \
             unittest.mock.patch.object(serve_cli.subprocess, "run", fake_run), \
             unittest.mock.patch.object(serve_cli.importlib.util, "find_spec", find_spec):
            code, _, _ = self.run_cli(["serve", str(model_dir), "--weights-gib", "3",
                                       "--server", "omlx"])
        self.assertEqual(code, 0)
        self.assertIn("omlx.server", seen["argv"])
        self.assertIn("--model-dir", seen["argv"])

    def test_module_backend_routes_to_serve_main(self):
        decisions = FIXTURE["models"].append(fixture_entry(
            id="laya-typed-decisions",
            repo="mlx-community/Laya",
            kind="decisions",
            recommended=False,
            serve={"backend": "module", "module": "mlx_omarchy_laya.server"},
            context={"max_tokens": 1024},
        ))
        seen = {}

        def fake_run(argv, **_kw):
            seen["argv"] = argv
            return unittest.mock.Mock(returncode=0)

        model_dir = Path(self.tmp.name) / "laya-ckpt"
        model_dir.mkdir(exist_ok=True)
        with unittest.mock.patch.object(serve_cli, "needs_download", lambda _r: False), \
             unittest.mock.patch.object(serve_cli, "download_snapshot", lambda _r: model_dir), \
             unittest.mock.patch.object(serve_cli.subprocess, "run", fake_run):
            code, _, _ = self.run_cli(["serve", "laya-typed-decisions"])
        self.assertEqual(code, 0)
        argv = seen["argv"]
        self.assertIn("-c", argv)
        self.assertIn("mlx_omarchy_laya.server", argv[argv.index("-c") + 1])
        self.assertIn("serve_main", argv[argv.index("-c") + 1])
        FIXTURE["models"].pop()

    def test_decisions_kind_without_module_backend_is_honest(self):
        FIXTURE["models"].append(fixture_entry(
            id="dec-no-route", repo="mlx-community/Dec", kind="decisions",
            recommended=False, serve={"backend": "mlx-lm", "module": None}))
        code, _, _ = self.run_cli(["plan", "dec-no-route"])
        self.assertEqual(code, 0)  # plan reports; mlx-lm backend accepted as declared
        FIXTURE["models"].pop()

    def test_custom_code_checkpoint_warns_never_trusts(self):
        model_dir = Path(self.tmp.name) / "coded"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps({"auto_map": {"AutoModel": "x"}}))
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            serve_cli.check_custom_code(model_dir)
        self.assertIn("never", err.getvalue())
        self.assertIn("trust_remote_code", err.getvalue())


class CatalogCommandTests(CliTestBase):
    def test_catalog_list_prints_table(self):
        code, out, _ = self.run_cli(["catalog", "list"])
        self.assertEqual(code, 0)
        self.assertIn("test-chat-4b", out)
        self.assertIn("qualified", out)

    def test_catalog_status_offline_names_bundled_fallback(self):
        with unittest.mock.patch.object(catalog, "cache_path",
                                        lambda _h=None: Path(self.tmp.name) / "none.json"):
            code, out, _ = self.run_cli(["catalog", "status"])
        self.assertEqual(code, 0)
        self.assertIn("bundled fallback", out)

    def test_reserve_round_trip(self):
        code, out, _ = self.run_cli(["reserve", "laya", "1.1", "--note", "decisions"])
        self.assertEqual(code, 0)
        self.assertEqual(budget.load_reservations(self.home)["laya"]["bytes"],
                         int(1.1 * GiB))
        code, _, _ = self.run_cli(["unreserve", "laya"])
        self.assertEqual(code, 0)
        self.assertEqual(budget.load_reservations(self.home), {})

    def test_offline_flag_blocks_download(self):
        with unittest.mock.patch.object(serve_cli, "needs_download", lambda _r: True):
            code, _, err = self.run_cli(["serve", "test-chat-4b", "--yes", "--offline"])
        self.assertEqual(code, 1)
        self.assertIn("offline", err)




if __name__ == "__main__":
    unittest.main()
