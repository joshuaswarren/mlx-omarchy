"""Failing-first tests for the bonsai2 ids-probe hook (default OFF, env-gated).

The hook contract: per-uid/add_token accumulates, reset/finalize flushes.
The bonsai shim also wraps stream_generate to force-finalize the
detokenizer on generator close (the bonsai2 for-loop never calls
detok.finalize itself)."""
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
        # Real mlx_lm 0.31.3 streaming classes call reset on finalize;
        # mirror that for the hook contract.
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


def _fresh_module_under_test():
    """Purge mlx_lm / mlx_lm.tokenizer_utils / mlx_lm.generate /
    mlx_omarchy_bonsai2(.ids_probe) so the next import sees the fakes
    below. Real mlx_lm is not importable in this env, but if the
    test runner's import machinery pre-loaded any of them via
    site-packages, our fakes must win the next import to test the
    hook contract."""
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
    fake_generate = types.ModuleType("mlx_lm.generate")
    # stream_generate is patched in install_ids_probe; we leave a sentinel
    # here so the wrap has something to bind to.
    fake_generate.stream_generate = lambda *a, **k: (_ for _ in ()).throw(
        StopIteration)
    sys.modules["mlx_lm"] = types.ModuleType("mlx_lm")
    sys.modules["mlx_lm.tokenizer_utils"] = fake_tok
    sys.modules["mlx_lm.generate"] = fake_generate
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
        """The bonsai2 server's for-loop never calls detok.finalize
        explicitly; the stream_generate wrap triggers finalize on
        GeneratorExit. This test exercises the detok.finalize path
        directly (which stream_probe does on close)."""
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


if __name__ == "__main__":
    unittest.main()
