"""Serve CLI: approval gate, admission refusal, launch argv, catalog commands."""

import contextlib
import os
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
            "http": {"status": "qualified", "receipt": "http.md", "date": "2026-09-20"},
        },
        "recommended": True,
        "serve": {"backend": "mlx-lm", "module": None},
        "availability": {"size_bytes": int(4.0 * GiB), "refreshed_at": None},
    }
    base.update(over)
    return base


def http_unqualified_entry():
    return fixture_entry(id="unqualified-chat", repo="mlx-community/Unqualified",
                         priority=2,
                         qualification={
                             "generation": {"status": "qualified", "receipt": "r.md",
                                            "date": "2026-09-20"},
                             "http": {"status": "untested", "receipt": None, "date": None},
                         })


FIXTURE = {
    "version": 1,
    "generated_at": "2026-09-20T00:00:00Z",
    "source": "https://example.invalid/catalog.json",
    "models": [fixture_entry(), http_unqualified_entry()],
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

    def test_auto_pick_skips_http_unqualified_entries(self):
        FIXTURE["models"] = [http_unqualified_entry()]
        code, out, _ = self.run_cli(["recommend"])
        self.assertEqual(code, 0)
        self.assertIn("pick: none", out)
        self.assertIn("http serving unqualified", out)
        FIXTURE["models"] = [fixture_entry(), http_unqualified_entry()]

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

    def test_manual_unqualified_target_warns_but_plans(self):
        code, out, err = self.run_cli(["plan", "unqualified-chat"])
        self.assertEqual(code, 0)
        self.assertIn("http serving unqualified", err)
        self.assertIn("FITS", out)

    def test_module_not_in_allowlist_refuses(self):
        FIXTURE["models"].append(fixture_entry(
            id="rogue-module", repo="mlx-community/Rogue", priority=3,
            kind="decisions", recommended=False, serve={"backend": "module",
                                                        "module": "evil_pkg.server"}))
        code, _, err = self.run_cli(["plan", "rogue-module"])
        self.assertEqual(code, 2)
        self.assertIn("allowlist", err)
        FIXTURE["models"].pop()

    def test_kv_unknown_model_refuses_long_context(self):
        FIXTURE["models"].append(fixture_entry(
            id="no-kv-facts", repo="mlx-community/NoKv", recommended=False,
            priority=4,
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
        with unittest.mock.patch.object(serve_cli, "probe_snapshot", lambda _r: None):
            code, _, err = self.run_cli(["serve", "test-chat-4b"])
        self.assertEqual(code, 1)
        self.assertIn("noninteractive", err)

    def test_yes_without_explicit_target_refuses(self):
        with unittest.mock.patch.object(serve_cli, "probe_snapshot", lambda _r: None):
            code, _, err = self.run_cli(["serve", "--yes"])
        self.assertEqual(code, 1)
        self.assertIn("explicitly named target", err)

    def test_interactive_rejection_aborts(self):
        inputs = iter(["no\n"])
        with unittest.mock.patch.object(serve_cli, "probe_snapshot", lambda _r: None), \
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

        with unittest.mock.patch.object(serve_cli, "probe_snapshot", lambda _r: None), \
             unittest.mock.patch.object(serve_cli, "download_snapshot", fake_download), \
             unittest.mock.patch.object(serve_cli.subprocess, "run", fake_run), \
             unittest.mock.patch.object(serve_cli.sys.stdin, "isatty", lambda: True), \
             unittest.mock.patch("builtins.input", lambda *_: "yes\n"):
            code, _, _ = self.run_cli(["serve", "test-chat-4b"])
        self.assertEqual(code, 0)
        self.assertEqual(seen["downloaded"], "mlx-community/Test-Chat-4b")
        self.assertIn("-m", seen["argv"])
        self.assertIn("mlx_omarchy_serve._mlxlm_server", seen["argv"])
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

        def forbidden_download(_resolved):
            raise AssertionError("download must not run for a complete local snapshot")

        patchers = [
            unittest.mock.patch.object(serve_cli, "probe_snapshot", lambda _r: model_dir),
            unittest.mock.patch.object(serve_cli, "download_snapshot", forbidden_download),
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
        self.assertIn("mlx_omarchy_serve._mlxlm_server", seen["argv"])
        # server-side context cap matches the admitted budget (default 4096)
        self.assertIn("--max-tokens", seen["argv"])
        self.assertEqual(seen["argv"][seen["argv"].index("--max-tokens") + 1], "4096")
        self.assertNotIn("trust_remote_code", " ".join(seen["argv"]))

    def test_explicit_context_flows_to_server_cap(self):
        code, seen, out, err = self.serve_local(["--context", "2048"])
        self.assertEqual(code, 0)
        self.assertEqual(seen["argv"][seen["argv"].index("--max-tokens") + 1], "2048")

    def test_disk_insufficient_blocks_serve(self):
        def fake_run(argv, **_kw):
            return unittest.mock.Mock(returncode=0)
        with unittest.mock.patch.object(serve_cli.subprocess, "run", fake_run), \
             unittest.mock.patch.object(budget, "disk_free", lambda _p: 0):
            code, out, err = self.run_cli(["serve", "test-chat-4b", "--yes"])
        self.assertEqual(code, 1)
        self.assertIn("memory or disk", err)

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

        with unittest.mock.patch.object(serve_cli, "probe_snapshot", lambda _r: model_dir), \
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
        with unittest.mock.patch.object(serve_cli, "probe_snapshot", lambda _r: model_dir), \
             unittest.mock.patch.object(serve_cli.subprocess, "run", fake_run):
            code, _, _ = self.run_cli(["serve", "laya-typed-decisions"])
        self.assertEqual(code, 0)
        argv = seen["argv"]
        self.assertIn("-c", argv)
        self.assertIn("mlx_omarchy_laya.server", argv[argv.index("-c") + 1])
        self.assertIn("serve_main", argv[argv.index("-c") + 1])
        self.assertIn("--managed", argv)  # admission-controlled launch
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
        with unittest.mock.patch.object(serve_cli, "probe_snapshot", lambda _r: None):
            code, _, err = self.run_cli(["serve", "test-chat-4b", "--yes", "--offline"])
        self.assertEqual(code, 1)
        self.assertIn("offline", err)

    def test_offline_env_var_blocks_download(self):
        with unittest.mock.patch.dict(os.environ, {"MLX_OMARCHY_OFFLINE": "1"}):
            with unittest.mock.patch.object(serve_cli, "probe_snapshot", lambda _r: None):
                code, _, err = self.run_cli(["serve", "test-chat-4b", "--yes"])
        self.assertEqual(code, 1)
        self.assertIn("offline", err)

    def test_offline_complete_snapshot_launches_without_network(self):
        model_dir = Path(self.tmp.name) / "complete"
        model_dir.mkdir()
        (model_dir / "model.safetensors").write_bytes(b"x" * 64)
        seen = {}

        def fake_run(argv, **_kw):
            seen["argv"] = argv
            return unittest.mock.Mock(returncode=0)

        class FakeHub:
            def snapshot_download(self, **kw):
                assert kw.get("local_files_only") is True, "offline must stay local"
                return model_dir

        def forbidden(_resolved):
            raise AssertionError("online download ran under --offline")

        with unittest.mock.patch.object(serve_cli, "_import_huggingface_hub",
                                        lambda: FakeHub()), \
             unittest.mock.patch.object(serve_cli, "download_snapshot", forbidden), \
             unittest.mock.patch.object(serve_cli.subprocess, "run", fake_run):
            code, _, _ = self.run_cli(["serve", "test-chat-4b", "--yes", "--offline"])
        self.assertEqual(code, 0)
        self.assertIn("mlx_omarchy_serve._mlxlm_server", seen["argv"])
        self.assertIn("--max-tokens", seen["argv"])

    def test_partial_snapshot_counts_as_download(self):
        model_dir = Path(self.tmp.name) / "partial"
        model_dir.mkdir()
        (model_dir / "model.safetensors.index.json").write_text(json.dumps(
            {"weight_map": {"w0": "model-00000-of-00002.safetensors",
                            "w1": "model-00001-of-00002.safetensors"}}))
        (model_dir / "model-00000-of-00002.safetensors").write_bytes(b"x" * 32)

        class FakeHub:
            def snapshot_download(self, **kw):
                assert kw.get("local_files_only") is True
                return model_dir

        # shard 2 missing -> probe says incomplete -> download gate applies;
        # noninteractive without --yes must refuse before any fetch.
        with unittest.mock.patch.object(serve_cli, "_import_huggingface_hub",
                                        lambda: FakeHub()):
            code, _, err = self.run_cli(["serve", "test-chat-4b"])
        self.assertEqual(code, 1)
        self.assertIn("noninteractive", err)

    def test_complete_snapshot_passes_the_probe(self):
        model_dir = Path(self.tmp.name) / "complete"
        model_dir.mkdir()
        (model_dir / "model.safetensors.index.json").write_text(json.dumps(
            {"weight_map": {"w0": "model-00000-of-00001.safetensors"}}))
        (model_dir / "model-00000-of-00001.safetensors").write_bytes(b"x" * 32)
        self.assertTrue(serve_cli.snapshot_complete(model_dir))
        empty = Path(self.tmp.name) / "empty"
        empty.mkdir()
        self.assertFalse(serve_cli.snapshot_complete(empty))

    def test_weights_gib_rejects_nonpositive_and_nonfinite(self):
        for bad in ("-3", "0", "nan", "inf", "abc"):
            with self.subTest(bad):
                with self.assertRaises(SystemExit) as ctx:
                    self.run_cli(["plan", "someone/Some-Model", "--weights-gib", bad])
                self.assertEqual(ctx.exception.code, 2)




if __name__ == "__main__":
    unittest.main()
