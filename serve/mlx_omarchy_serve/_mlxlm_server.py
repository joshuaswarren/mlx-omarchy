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

import inspect
import json
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


EXPECTED_TOKENIZE_SIGNATURE = ("self", "tokenizer", "request", "args")


def install(server_module, limit: int) -> None:
    generator = server_module.ResponseGenerator
    original = getattr(generator, "_tokenize", None)
    if not callable(original):
        raise RuntimeError(
            "mlx_lm.server.ResponseGenerator._tokenize not found; this shim "
            "does not know this mlx-lm version and refuses to launch without "
            "the total-context cap"
        )
    # Enforced for the pinned version AND any pin bypass: if the internals
    # moved, the cap cannot be proven to hold and the launch refuses.
    try:
        params = tuple(inspect.signature(original).parameters)
    except (TypeError, ValueError):
        params = ()
    if params != EXPECTED_TOKENIZE_SIGNATURE:
        raise RuntimeError(
            "unexpected _tokenize signature "
            f"{params} != {EXPECTED_TOKENIZE_SIGNATURE}; this shim does not "
            "know this mlx-lm build and refuses to launch without the "
            "verified total-context cap"
        )

    def capped(self, tokenizer, request, args):
        result = original(self, tokenizer, request, args)
        prompt = result[0]
        # Upstream semantics (verified in the pinned source): the HTTP
        # handler resolves max_tokens from the body or the CLI default
        # (an int, default 512) and validates int >= 0 BEFORE queuing, so
        # None or negatives never legitimately reach this boundary.
        # 0 generates nothing (bounded); -1 (infinite) is handler-blocked.
        max_tokens = getattr(args, "max_tokens", None)
        if max_tokens is None or isinstance(max_tokens, bool) \
                or not isinstance(max_tokens, int) or max_tokens < 0:
            raise ValueError(
                "request rejected: max_tokens must be a non-negative int at "
                f"the cap boundary, got {max_tokens!r}"
            )
        total = len(prompt) + max_tokens
        if total > limit:
            raise ValueError(
                f"request rejected: prompt ({len(prompt)} tokens) + max_tokens "
                f"({max_tokens}) exceeds the admitted context budget of {limit}; "
                "lower the prompt or max_tokens"
            )
        return result

    generator._tokenize = capped


PROBE_ENV = "MLX_OMARCHY_SERVE_IDS_PROBE"


def install_ids_probe(server_module, emit=None) -> None:
    """Bench-only ids capture for the managed route (default OFF; enabled
    only when MLX_OMARCHY_SERVE_IDS_PROBE=1). Wraps the streaming
    detokenizer classes so every raw sampled token id is recorded at the
    add_token boundary and the completed per-request id list is emitted as
    one `PROBE: {...}` JSON line on stderr at reset(). Nothing on disk is
    touched; with the env unset this function is never called."""
    if emit is None:
        def emit(event):
            print("PROBE:", json.dumps(event), file=sys.stderr, flush=True)
    state = {"ids": []}
    from mlx_lm.tokenizer_utils import (
        NaiveStreamingDetokenizer,
        BPEStreamingDetokenizer,
        SPMStreamingDetokenizer,
        StreamingDetokenizer,
    )

    def wrap(cls):
        if cls is None:
            return
        # Patch only methods defined in the class's OWN dict: a probe on a
        # base class is inherited by subclasses, and wrapping both would
        # record every token twice (subclass probe -> inherited base probe).
        orig_add = cls.__dict__.get("add_token")
        if orig_add is None:
            return

        def add_token_probe(self, token, _orig=orig_add):
            state["ids"].append(int(token))
            _orig(self, token)

        cls.add_token = add_token_probe

        orig_reset = cls.__dict__.get("reset")
        if orig_reset is None:
            return

        def reset_probe(self, _orig=orig_reset):
            if state["ids"]:
                import hashlib
                emit({"event": "generation", "n": len(state["ids"]),
                      "ids_sha16": hashlib.sha256(
                          json.dumps(state["ids"]).encode()).hexdigest()[:16],
                      "ids": state["ids"]})
                state["ids"] = []
            return _orig(self)

        cls.reset = reset_probe

    for cls in (StreamingDetokenizer, NaiveStreamingDetokenizer,
                BPEStreamingDetokenizer, SPMStreamingDetokenizer):
        wrap(cls)


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
    if os.environ.get(PROBE_ENV) == "1":
        install_ids_probe(server_module)
    server_module.main()


if __name__ == "__main__":
    main()
