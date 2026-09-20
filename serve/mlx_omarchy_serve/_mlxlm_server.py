"""Total-context-capped launcher for mlx_lm.server (pinned 0.31.3).

The upstream server treats --max-tokens as a per-request DEFAULT: a client
may send any prompt plus any max_tokens of its own, and nothing bounds
prompt + output. This shim adds the missing cap at the tokenization trust
boundary — the one place where the full prompt token list exists — and
rejects any request whose prompt + max_tokens exceeds the admitted context
budget before a single token is generated.

Aggregate bound: the serve CLI launches this server with decode and prompt
concurrency 1 and the prompt cache disabled, so exactly ONE request's
tokens are ever resident and the aggregate equals the admitted context.
A larger --prompt-cache-size re-enables prefix caching at a documented
cost: worst-case resident KV becomes (cache + 1) x the admitted context.

Fail-closed pinning: the shim supports exactly the pinned mlx-lm version.
A different mlx_lm.__version__ refuses to launch unless
MLX_OMARCHY_ALLOW_UNPINNED_MLXLM=1 is set explicitly (loud warning); if
the hooked internals moved anyway, the launch still fails loudly instead
of silently dropping the cap.
"""

from __future__ import annotations

import os
import sys

LIMIT_ENV = "MLX_OMARCHY_SERVE_CONTEXT_LIMIT"
UNPINNED_ENV = "MLX_OMARCHY_ALLOW_UNPINNED_MLXLM"
SUPPORTED_MLXLM_VERSIONS = ("0.31.3",)


def _mlxlm_version(server_module) -> str:
    for holder in (server_module, getattr(server_module, "_version", None)):
        version = getattr(holder, "__version__", None)
        if version:
            return str(version)
    return "unknown"


def check_pinned(server_module) -> str:
    version = _mlxlm_version(server_module)
    if version in SUPPORTED_MLXLM_VERSIONS:
        return version
    if os.environ.get(UNPINNED_ENV) == "1":
        print(
            f"warning: {UNPINNED_ENV}=1 — mlx-lm {version} is not the pinned "
            f"{SUPPORTED_MLXLM_VERSIONS}; the cap hooks may not hold",
            file=sys.stderr,
        )
        return version
    raise RuntimeError(
        f"mlx-lm {version} is not the pinned version "
        f"{SUPPORTED_MLXLM_VERSIONS}; refusing to launch without the "
        f"verified total-context cap (export {UNPINNED_ENV}=1 to override "
        "explicitly)"
    )


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
        max_tokens = getattr(args, "max_tokens", 0)
        if max_tokens is None:
            max_tokens = 0
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
            raise ValueError(
                f"request rejected: max_tokens must be a non-negative int, "
                f"got {type(max_tokens).__name__}"
            )
        if max_tokens < 0:
            raise ValueError("request rejected: max_tokens must be >= 0")
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
    check_pinned(server_module)
    install(server_module, limit)
    server_module.main()


if __name__ == "__main__":
    main()
