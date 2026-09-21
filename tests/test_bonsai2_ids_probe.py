"""Failing-first tests for the bonsai2 ids-probe hook (default OFF, env-gated).

The hook contract is identical to the mlxlm shim hook: per-uid/add_token
accumulates, reset/finalize flushes. The bonsai shim uses the SAME
mlx_lm.tokenizer_utils streaming classes, so the test mirror is the
single-class add_token + reset path."""
import contextlib
import importlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
import unittest.mock
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "serve"))


def _fake_tokenizer_utils():
    def add_token(self, token):
        self.seen.append(token)
    def reset(self):
        self.seen = []
    base = type("StreamingDetokenizer", (),
                {"add_token": add_token, "reset": reset,
                 "__init__": lambda self: setattr(self, "seen", [])})
    naive = type("NaiveStreamingDetokenizer", (base,), {})
    bpe = type("BPEStreamingDetokenizer", (base,), {})
    mod = types.ModuleType("mlx_lm.tokenizer_utils")
    mod.StreamingDetokenizer = base
    mod.NaiveStreamingDetokenizer = naive
    mod.BPEStreamingDetokenizer = bpe
    mod.SPMStreamingDetokenizer = None
    return mod


class Bonsai2IdsProbeTests(unittest.TestCase):
    def setUp(self):
        self._saved_env = os.environ.copy()
        sys.modules.pop("mlx_omarchy_bonsai2.ids_probe", None)
        self.mod = importlib.import_module("mlx_omarchy_bonsai2.ids_probe")
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.fake = _fake_tokenizer_utils()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._saved_env)
        sys.modules.pop("mlx_lm.tokenizer_utils", None)
        sys.modules.pop("mlx_omarchy_bonsai2.ids_probe", None)

    def test_noop_without_env(self):
        os.environ[self.mod.PROBE_ENV] = "0"
        with unittest.mock.patch.dict(
                sys.modules, {"mlx_lm.tokenizer_utils": self.fake}):
            state = self.mod.install_ids_probe(emit=lambda e: self.fail(e))
        self.assertIsNone(state)
        # Class NOT marked installed.
        self.assertFalse(getattr(self.fake.StreamingDetokenizer,
                                "_ids_probe_installed", False))

    def test_records_ids_and_emits_on_reset_when_env_on(self):
        os.environ[self.mod.PROBE_ENV] = "1"
        events = []
        with unittest.mock.patch.dict(
                sys.modules, {"mlx_lm.tokenizer_utils": self.fake}):
            self.mod.install_ids_probe(emit=events.append)
        d = self.fake.StreamingDetokenizer()
        d.add_token(11)
        d.add_token(22)
        d.reset()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["ids"], [11, 22])
        self.assertEqual(events[0]["event"], "generation")
        self.assertEqual(len(events[0]["ids_sha16"]), 16)


if __name__ == "__main__":
    unittest.main()
