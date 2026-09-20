"""Total-context cap shim for mlx_lm.server: enforcement proofs.

Upstream 0.31.3 fact (verified in the published wheel, server.py):
  self.max_tokens = body.get("max_completion_tokens") or
                    body.get("max_tokens", cli_args.max_tokens)
  self._validate("max_tokens", int, min_val=0)
i.e. --max-tokens is a per-request DEFAULT and a client may send any
prompt plus any non-negative max_tokens: nothing upstream bounds
prompt + output. These tests prove the shim does.
"""

import os
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "serve"))

from mlx_omarchy_serve import _mlxlm_server as cap  # noqa: E402


def fresh_server():
    """A pristine fake ResponseGenerator per test: installing the cap
    mutates the class, so tests must never share one."""

    class ResponseGenerator:
        def _tokenize(self, tokenizer, request, args):
            return ([0] * request.prompt_len, [], [], None)

    # class bodies cannot read enclosing function locals; go through type()
    return type("Server", (), {"ResponseGenerator": ResponseGenerator})


class FakeRequest:
    def __init__(self, prompt_len):
        self.prompt_len = prompt_len


class FakeArgs:
    def __init__(self, max_tokens):
        self.max_tokens = max_tokens


class CapTests(unittest.TestCase):
    def setUp(self):
        self.server = fresh_server()
        cap.install(self.server, limit=1024)
        self.gen = self.server.ResponseGenerator()

    def test_under_limit_passes_through(self):
        result = self.gen._tokenize(None, FakeRequest(1000), FakeArgs(24))
        self.assertEqual(len(result[0]), 1000)

    def test_prompt_plus_output_over_limit_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            self.gen._tokenize(None, FakeRequest(1000), FakeArgs(25))
        self.assertIn("admitted context budget", str(ctx.exception))

    def test_prompt_alone_over_limit_rejected(self):
        with self.assertRaises(ValueError):
            self.gen._tokenize(None, FakeRequest(2000), FakeArgs(0))

    def test_missing_max_tokens_treated_as_zero(self):
        result = self.gen._tokenize(None, FakeRequest(1024), FakeArgs(None))
        self.assertEqual(len(result[0]), 1024)

    def test_rejections_carry_no_partial_generation(self):
        with self.assertRaises(ValueError):
            self.gen._tokenize(None, FakeRequest(1025), FakeArgs(64))

    def test_refuses_unknown_internals_loudly(self):
        class NewerServer:
            class ResponseGenerator:
                pass

        with self.assertRaises(RuntimeError) as ctx:
            cap.install(NewerServer, limit=1024)
        self.assertIn("refuses to launch", str(ctx.exception))

    def test_main_rejects_missing_or_bad_limit(self):
        saved = os.environ.pop(cap.LIMIT_ENV, None)
        try:
            for bad in (None, "not-a-number", "0", "-5"):
                if bad is not None:
                    os.environ[cap.LIMIT_ENV] = bad
                with self.subTest(bad=bad):
                    with self.assertRaises(SystemExit) as ctx:
                        cap.main()
                    self.assertEqual(ctx.exception.code, 3)
        finally:
            os.environ.pop(cap.LIMIT_ENV, None)
            if saved is not None:
                os.environ[cap.LIMIT_ENV] = saved

    def test_main_installs_cap_and_calls_server_main(self):
        server = fresh_server()
        calls = {"main": False}

        class RecordingServer:
            ResponseGenerator = server.ResponseGenerator

            @staticmethod
            def main():
                calls["main"] = True

        sys.modules["mlx_lm"] = type(sys)("mlx_lm")
        sys.modules["mlx_lm.server"] = RecordingServer
        os.environ[cap.LIMIT_ENV] = "512"
        try:
            cap.main()
        finally:
            os.environ.pop(cap.LIMIT_ENV, None)
            del sys.modules["mlx_lm.server"], sys.modules["mlx_lm"]
        self.assertTrue(calls["main"])
        self.assertEqual(RecordingServer.ResponseGenerator._tokenize.__name__, "capped")
        gen = RecordingServer.ResponseGenerator()
        with self.assertRaises(ValueError):
            gen._tokenize(None, FakeRequest(600), FakeArgs(0))  # 600 > 512


if __name__ == "__main__":
    unittest.main()
