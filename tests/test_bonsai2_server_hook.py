"""Failing-first test for the bonsai2 server-side per-call stream_generate
lookup (item 1 of the post-rewrite assignment).

The test mocks `mlx_lm` as a package with `mlx_lm.generate` (the
stream_generate function) and `mlx_lm.sample_utils` (a stub), and
imports the bonsai2 server module. The hook's `install_ids_probe`
rebinds `mlx_lm.generate` to the wrap; the bonsai2 server's per-call
lookup via `sys.modules["mlx_lm"].generate` picks up the wrap on
each request and the wrap records tokens + force-finalizes the detok."""
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

    def test_per_call_lookup_picks_up_wrap(self):
        """After install_ids_probe rebinds mlx_lm.generate to the wrap, the
        per-call lookup via sys.modules in _generate must pick up the
        wrap, the wrap records tokens, and the forced finalize emits
        the PROBE event with the correct ids array."""
        import mlx_omarchy_bonsai2.ids_probe as ids_probe
        events = []
        ids_probe.install_ids_probe(emit=events.append)
        # The rebind: mlx_lm.generate is now the wrap, NOT the real
        # stream_generate.
        import mlx_lm
        self.assertIsNot(
            mlx_lm.generate.__name__, "_real_stream",
            "rebind did not happen; hook failed",
        )
        # Call _generate on the fake state.
        text, finish, _ = self.bonsai._generate(
            self.fake_state, [1, 2, 3], max_tokens=4, temperature=0.0, top_p=0.0,
        )
        self.assertEqual(len(events), 1, f"no PROBE event fired: {events}")
        self.assertEqual(events[0]["ids"], [11, 22, 33])
        self.assertEqual(events[0]["event"], "generation")


if __name__ == "__main__":
    unittest.main()
