"""Failing-first tests for the bonsai2 ids-probe hook (default OFF, env-gated).

The hook contract: per-uid/add_token accumulates, reset/finalize flushes.
The bonsai shim wraps stream_generate to force-finalize the detokenizer
on generator close. In mlx_lm 0.31.3, `mlx_lm.generate` is the
stream_generate function itself (re-exported at the package level),
not a submodule; the hook handles this layout."""
import importlib
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _fake_tokenizer_utils():
    def add_token(self, token):
        self.seen.append(token)
    def reset(self):
        self.seen = []
    def finalize(self):
        self.seen = []
    base = type("StreamingDetokenizer", (),
                {"add_token": add_token, "reset": reset, "finalize": finalize,
                 "__init__": lambda self: setattr(self, "seen", [])})
    naive = type("NaiveStreamingDetokenizer", (base,), {})
    bpe = type("BPEStreamingDetokenizer", (base,), {})
    mod = types.ModuleType("mlx_lm.tokenizer_utils")
    mod.StreamingDetokenizer = base
    mod.NaiveStreamingDetokenizer = naive
    mod.BPEStreamingDetokenizer = bpe
    mod.SPMStreamingDetokenizer = None
    return mod


def _fake_stream_fn(*a, **k):
    return iter(())


def _fresh_module_under_test():
    for mod in (
            "mlx_omarchy_bonsai2.ids_probe",
            "mlx_omarchy_bonsai2",
            "mlx_lm.generate",
            "mlx_lm.tokenizer_utils",
            "mlx_lm",
    ):
        sys.modules.pop(mod, None)
    sys.path.insert(0, str(REPO_ROOT / "serve"))
    fake_tok = _fake_tokenizer_utils()
    # mlx_lm 0.31.3 flattens: mlx_lm.generate IS the stream_generate
    # function itself, not a submodule. The package has only the function.
    sys.modules["mlx_lm"] = types.ModuleType("mlx_lm")
    sys.modules["mlx_lm"].generate = _fake_stream_fn
    sys.modules["mlx_lm.tokenizer_utils"] = fake_tok
    mod = importlib.import_module("mlx_omarchy_bonsai2.ids_probe")
    return mod, fake_tok


class Bonsai2IdsProbeTests(unittest.TestCase):
    def setUp(self):
        self._saved_env = os.environ.copy()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.mod, self.fake = _fresh_module_under_test()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._saved_env)
        for mod in (
                "mlx_omarchy_bonsai2.ids_probe",
                "mlx_omarchy_bonsai2",
                "mlx_lm.generate",
                "mlx_lm.tokenizer_utils",
                "mlx_lm",
        ):
            sys.modules.pop(mod, None)

    def test_noop_without_env(self):
        os.environ[self.mod.PROBE_ENV] = "0"
        state = self.mod.install_ids_probe(emit=lambda e: self.fail(e))
        self.assertIsNone(state)
        self.assertFalse(getattr(self.fake.StreamingDetokenizer,
                                "_ids_probe_installed", False))

    def test_records_ids_and_emits_on_reset_when_env_on(self):
        os.environ[self.mod.PROBE_ENV] = "1"
        events = []
        self.mod.install_ids_probe(emit=events.append)
        d = self.fake.StreamingDetokenizer()
        d.add_token(11)
        d.add_token(22)
        d.reset()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["ids"], [11, 22])
        self.assertEqual(events[0]["event"], "generation")
        self.assertEqual(len(events[0]["ids_sha16"]), 16)

    def test_records_ids_and_emits_on_finalize_when_env_on(self):
        os.environ[self.mod.PROBE_ENV] = "1"
        events = []
        self.mod.install_ids_probe(emit=events.append)
        d = self.fake.StreamingDetokenizer()
        d.add_token(100)
        d.add_token(200)
        d.add_token(300)
        d.finalize()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["ids"], [100, 200, 300])
        self.assertEqual(events[0]["event"], "generation")

    def test_stream_generate_wrap_end_to_end(self):
        """End-to-end: the wrap calls the original stream_generate (which
        adds 3 tokens via the patched add_token probe) and force-finalizes
        the detok on generator close. Uses positional args (model,
        tokenizer, prompt) like the real bonsai2 server."""
        os.environ[self.mod.PROBE_ENV] = "1"
        import sys as _sys
        fake_outer = self.fake
        def _fake_stream(model, tokenizer, prompt, **kwargs):
            for t in (11, 22, 33):
                tokenizer.detokenizer.add_token(t)
            return
            yield  # makes this a generator function
        # The wrap saves the original stream_generate BEFORE rebinding, so
        # this fake MUST be installed as mlx_lm.generate BEFORE install_ids_probe.
        # In mlx_lm 0.31.3, mlx_lm.generate IS the function (not a
        # submodule); the hook reads it via the package attr.
        _sys.modules["mlx_lm"].generate = _fake_stream
        bonsai = _sys.modules["mlx_omarchy_bonsai2.server"] = types.ModuleType(
            "mlx_omarchy_bonsai2.server")
        bonsai.stream_generate = _fake_stream  # will be rebound to wrap
        try:
            class _Tok:
                def __init__(self_inner, _fake=fake_outer):
                    self_inner.detokenizer = _fake.StreamingDetokenizer()
            tok = _Tok()
            events = []
            self.mod.install_ids_probe(emit=events.append)
            for _ in bonsai.stream_generate(None, tok, [1, 2, 3],
                                          max_tokens=4):
                pass
            self.assertEqual(len(events), 1, f"events={events}")
            self.assertEqual(events[0]["ids"], [11, 22, 33])
            self.assertEqual(events[0]["event"], "generation")
        finally:
            _sys.modules.pop("mlx_omarchy_bonsai2.server", None)


if __name__ == "__main__":
    unittest.main()
