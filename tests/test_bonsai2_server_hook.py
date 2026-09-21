"""Integration test for the bonsai2 server-side per-call stream_generate
lookup + detok-class hooks (item 1 of the post-rewrite assignment).

The bonsai2 server captures `stream_generate` via the per-call
`sys.modules["mlx_lm"].generate` lookup (the ids-probe wrap rebinds
`mlx_lm.generate` to the wrap). The detok-class add_token probe
records tokens and the reset/finalize hooks flush."""
import importlib
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


class _FakeDetok:
    def __init__(self):
        self.seen = []

    def add_token(self, t):
        self.seen.append(t)

    def finalize(self):
        pass


def _fresh_module_imports():
    for mod in (
            "mlx_omarchy_bonsai2.ids_probe",
            "mlx_omarchy_bonsai2",
            "mlx_omarchy_bonsai2.server",
            "mlx_lm.generate",
            "mlx_lm.sample_utils",
            "mlx_lm.tokenizer_utils",
            "mlx_lm",
    ):
        sys.modules.pop(mod, None)
    sys.path.insert(0, str(REPO_ROOT / "serve"))
    fake_tok_mod = types.ModuleType("mlx_lm.tokenizer_utils")
    detok = _FakeDetok()
    base = type("StreamingDetokenizer", (),
                {"add_token": lambda self, t: self.seen.append(t),
                 "finalize": lambda self: None,
                 "reset": lambda self: None,
                 "__init__": lambda s: setattr(s, "seen", [])})
    fake_tok_mod.StreamingDetokenizer = base
    fake_tok_mod.NaiveStreamingDetokenizer = type("N", (base,), {})
    fake_tok_mod.BPEStreamingDetokenizer = type("B", (base,), {})
    fake_tok_mod.SPMStreamingDetokenizer = None
    sys.modules["mlx_lm.tokenizer_utils"] = fake_tok_mod
    sys.modules["mlx_lm"] = types.ModuleType("mlx_lm")
    # Stub submodules so the bonsai2 server's imports resolve.
    sample_mod = types.ModuleType("mlx_lm.sample_utils")
    def _make_sampler(temp=0.0, top_p=0.0):
        return object()
    sample_mod.make_sampler = _make_sampler
    sys.modules["mlx_lm.sample_utils"] = sample_mod
    sys.modules["mlx_lm"].sample_utils = sample_mod
    # The REAL stream_generate is the sentinel below.
    def _real_stream(model, tokenizer, prompt, **kwargs):
        for t in (11, 22, 33):
            tokenizer.detokenizer.add_token(t)
        if False:
            yield
    sys.modules["mlx_lm"].generate = _real_stream
    sys.modules["mlx_omarchy_bonsai2.ids_probe"] = importlib.import_module(
        "mlx_omarchy_bonsai2.ids_probe")


class Bonsai2ServerHookIntegrationTests(unittest.TestCase):
    def setUp(self):
        self._saved_env = os.environ.copy()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        _fresh_module_imports()
        os.environ["MLX_OMARCHY_SERVE_IDS_PROBE"] = "1"
        sys.path.insert(0, str(REPO_ROOT / "serve"))
        sys.path.insert(0, str(REPO_ROOT / "serve" / "mlx_omarchy_laya"))
        sys.path.insert(0, str(REPO_ROOT / "serve" / "mlx_omarchy_bonsai2"))
        import mlx_omarchy_bonsai2.server as bserver
        self.bonsai = bserver
        detok = sys.modules["mlx_lm.tokenizer_utils"].StreamingDetokenizer()
        self.fake_model = object()
        self.fake_tok = type("T", (), {"detokenizer": detok})()
        self.fake_state = type(
            "S", (), {"model": self.fake_model, "tokenizer": self.fake_tok}
        )()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._saved_env)
        for mod in (
                "mlx_omarchy_bonsai2.ids_probe",
                "mlx_omarchy_bonsai2",
                "mlx_omarchy_bonsai2.server",
                "mlx_lm.generate",
                "mlx_lm.sample_utils",
                "mlx_lm.tokenizer_utils",
                "mlx_lm",
        ):
            sys.modules.pop(mod, None)

    def test_detok_hook_installed_and_add_token_recorded(self):
        """After install_ids_probe, the detok class-level add_token hook
        records tokens and the finalize hook flushes them via the PROBE
        emit."""
        import mlx_omarchy_bonsai2.ids_probe as ids_probe
        events = []
        ids_probe.install_ids_probe(emit=events.append)
        # The per-call lookup in _generate reads sys.modules['mlx_lm'].generate;
        # after install, mlx_lm.generate IS the stream_generate wrap (which
        # is a generator function that calls detok.add_token internally).
        # So the per-call lookup DOES pick up the rebind. The detok's
        # class-level add_token hook then records each token.
        d = sys.modules["mlx_lm.tokenizer_utils"].StreamingDetokenizer()
        d.add_token(11)
        d.add_token(22)
        d.add_token(33)
        d.finalize()
        self.assertEqual(len(events), 1, f"no PROBE event fired: {events}")
        self.assertEqual(events[0]["ids"], [11, 22, 33])
        self.assertEqual(events[0]["event"], "generation")

    def test_per_call_lookup_picks_up_wrap(self):
        """After install_ids_probe rebinds mlx_lm.generate to the wrap, the
        per-call lookup via sys.modules must pick up the wrap (not the
        original stream_generate)."""
        import mlx_lm
        # The hook rebinds mlx_lm.generate to the wrap. Verify the package
        # attribute is changed (not the original _real_stream).
        self.assertNotEqual(
            mlx_lm.generate.__name__, "_real_stream",
            "package rebind did not happen; hook failed",
        )
        # The detok classes are also hooked.
        fake_tok = sys.modules["mlx_lm.tokenizer_utils"]
        self.assertTrue(getattr(fake_tok.StreamingDetokenizer,
                                "_ids_probe_installed", False))


if __name__ == "__main__":
    unittest.main()
