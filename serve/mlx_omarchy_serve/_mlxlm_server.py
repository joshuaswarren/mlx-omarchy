"""Total-context-capped launcher for mlx_lm.server (0.31.3).

The upstream server treats --max-tokens as a per-request DEFAULT: a client
may send any prompt plus any max_tokens of its own, and nothing bounds
prompt + output. This shim adds the missing cap at the tokenization trust
boundary — the one place where the full prompt token list exists — and
rejects any request whose prompt + max_tokens exceeds the admitted context
budget before a single token is generated.

The limit arrives via MLX_OMARCHY_SERVE_CONTEXT_LIMIT (set by the serve
CLI to exactly the admitted context). If the mlx-lm pin bump moves the
internals this shim hooks, the launch fails loudly instead of silently
dropping the cap.
"""

from __future__ import annotations

import os
import sys

LIMIT_ENV = "MLX_OMARCHY_SERVE_CONTEXT_LIMIT"


def install(server_module, limit: int) -> None:
    generator = server_module.ResponseGenerator
    original = getattr(generator, "_tokenize", None)
    if not callable(original):
        raise RuntimeError(
            "mlx_lm.server.ResponseGenerator._tokenize not found; this shim "
            "does not know this mlx-lm version and refuses to launch without "
            "the total-context cap"
        )

    def capped(self, tokenizer, request, args):
        result = original(self, tokenizer, request, args)
        prompt = result[0]
        max_tokens = int(getattr(args, "max_tokens", 0) or 0)
        total = len(prompt) + max_tokens
        if total > limit:
            raise ValueError(
                f"request rejected: prompt ({len(prompt)} tokens) + max_tokens "
                f"({max_tokens}) exceeds the admitted context budget of {limit}; "
                "lower the prompt or max_tokens"
            )
        return result

    generator._tokenize = capped


def main() -> None:
    raw = os.environ.get(LIMIT_ENV, "")
    try:
        limit = int(raw)
    except ValueError:
        print(f"error: {LIMIT_ENV} must be an integer, got {raw!r}", file=sys.stderr)
        raise SystemExit(3)
    if limit <= 0:
        print(f"error: {LIMIT_ENV} must be positive", file=sys.stderr)
        raise SystemExit(3)
    try:
        import mlx_lm.server as server_module
    except ImportError as exc:
        print(f"error: mlx_lm is not installed ({exc})", file=sys.stderr)
        raise SystemExit(3)
    install(server_module, limit)
    server_module.main()


if __name__ == "__main__":
    main()
