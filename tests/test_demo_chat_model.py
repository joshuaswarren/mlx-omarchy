"""CLI parsing checks for demo/chat.py — no mlx import, no model download.

demo/chat.py must resolve/prompt for the model id BEFORE importing mlx or
mlx_lm, so these checks run the parser alone and assert the contract:
explicit value passes, an interactive terminal is prompted, a non-TTY
run without --model is rejected with usage text, and an empty answer is
rejected. No Apple hardware and no network.
"""

import io
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]

import importlib.util

_spec = importlib.util.spec_from_file_location("demo_chat", REPO / "demo" / "chat.py")
demo_chat = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(demo_chat)


class ResolveModelTests(unittest.TestCase):
    def test_explicit_model_flag_is_accepted_without_prompting(self):
        with mock.patch("builtins.input", side_effect=AssertionError("must not prompt")):
            args = demo_chat.resolve_model(["--model", "some-org/some-model", "--once"])
        self.assertEqual(args.model, "some-org/some-model")
        self.assertTrue(args.once)

    def test_interactive_tty_without_model_prompts_once(self):
        argv = ["--max-tokens", "8"]
        fake_stdin = io.StringIO("user-picked/model\n")
        fake_stdin.isatty = lambda: True
        fake_stdout = io.StringIO()
        fake_stdout.isatty = lambda: True
        with mock.patch("sys.stdin", new=fake_stdin), \
             mock.patch("sys.stdout", new=fake_stdout):
            args = demo_chat.resolve_model(argv)
        self.assertEqual(args.model, "user-picked/model")
        self.assertEqual(args.max_tokens, 8)

    def test_non_tty_without_model_is_a_usage_error(self):
        fake_stdin = io.StringIO("")  # isatty() -> False
        fake_stderr = io.StringIO()
        with mock.patch("sys.stdin", new=fake_stdin), \
             mock.patch("sys.stdout", new=io.StringIO()), \
             mock.patch("sys.stderr", new=fake_stderr):
            with self.assertRaises(SystemExit) as ctx:
                demo_chat.resolve_model([])
        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("--model is required when stdin/stdout is not an interactive terminal", fake_stderr.getvalue())

    def test_empty_interactive_answer_is_a_usage_error(self):
        fake_stdin = io.StringIO("   \n")  # whitespace-only answer
        fake_stdin.isatty = lambda: True
        fake_stdout = io.StringIO()
        fake_stdout.isatty = lambda: True
        fake_stderr = io.StringIO()
        with mock.patch("sys.stdin", new=fake_stdin), \
             mock.patch("sys.stdout", new=fake_stdout), \
             mock.patch("sys.stderr", new=fake_stderr):
            with self.assertRaises(SystemExit) as ctx:
                demo_chat.resolve_model([])
        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("--model is required (no model id entered)", fake_stderr.getvalue())

    def test_whitespace_model_flag_is_rejected_before_imports(self):
        fake_stderr = io.StringIO()
        with mock.patch("sys.stderr", new=fake_stderr):
            with self.assertRaises(SystemExit) as ctx:
                demo_chat.resolve_model(["--model", "   "])
        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("--model must be a non-empty model id", fake_stderr.getvalue())

    def test_mlx_never_imported_during_parsing(self):
        import builtins
        real_import = builtins.__import__

        def guard(name, *a, **k):
            if name == "mlx" or name.startswith("mlx.") or name.startswith("mlx_lm"):
                raise AssertionError(f"mlx import during parsing: {name}")
            return real_import(name, *a, **k)

        with mock.patch("builtins.__import__", side_effect=guard):
            args = demo_chat.resolve_model(["--model", "some-org/some-model"])
        self.assertEqual(args.model, "some-org/some-model")


if __name__ == "__main__":
    unittest.main()
