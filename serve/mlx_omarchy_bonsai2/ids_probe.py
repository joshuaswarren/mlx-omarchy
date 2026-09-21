"""Env-gated ids-probe hook for the bonsai2 managed shim (bench-only).

Default OFF: no behavior change. With MLX_OMARCHY_SERVE_IDS_PROBE=1, the
hook captures the raw sampled token ids at the per-token boundary and
emits one PROBE: JSON line per request. The bonsai2 server's iteration
loop (for response in stream_generate(...): break on finish_reason) never
calls detok.finalize explicitly, so the hook wraps stream_generate and
force-finalizes the detokenizer on generator close.

The bonsai2 server captures stream_generate via `from mlx_lm.generate
import stream_generate` (function-body, executed on every call) — and
the install function MUST save the original BEFORE rebinding, otherwise
the wrap's `orig_stream` captures the wrap itself (infinite recursion)."""
from __future__ import annotations
import hashlib
import json
import os
import sys

PROBE_ENV = "MLX_OMARCHY_SERVE_IDS_PROBE"

# Module-level caches: save the original stream_generate on each module
# the FIRST time install_ids_probe is called, BEFORE the rebind. This is
# a one-time save per import; subsequent install calls find the cached
# value (which the wrap rebinds, so we always re-save at the top of
# install_ids_probe by reading via a dedicated helper).
_ORIGINAL_STREAM_GENERATE = None


def install_ids_probe(emit=None):
    global _ORIGINAL_STREAM_GENERATE
    if os.environ.get(PROBE_ENV) != "1":
        return None
    if emit is None:
        def emit(event):
            print("PROBE:", json.dumps(event), file=sys.stderr, flush=True)
    try:
        from mlx_lm.tokenizer_utils import (
            BPEStreamingDetokenizer,
            NaiveStreamingDetokenizer,
            SPMStreamingDetokenizer,
            StreamingDetokenizer,
        )
        import mlx_lm.generate as _mlx_lm_generate
    except Exception as exc:
        print(f"shim: ids probe install failed: {exc}", file=sys.stderr,
              flush=True)
        return None
    state: dict = {"ids": [], "detok": None}

    def _flush():
        if state["ids"]:
            emit({"event": "generation",
                  "n": len(state["ids"]),
                  "ids_sha16": hashlib.sha256(
                      json.dumps(state["ids"]).encode()).hexdigest()[:16],
                  "ids": state["ids"]})
            state["ids"] = []

    def wrap(cls):
        if cls is None or getattr(cls, "_ids_probe_installed", False):
            return
        orig_add = getattr(cls, "add_token", None)
        if orig_add is None:
            return

        def add_token_probe(self, token, _orig=orig_add):
            state["ids"].append(int(token))
            _orig(self, token)

        cls.add_token = add_token_probe
        cls._ids_probe_installed = True

        orig_reset = getattr(cls, "reset", None)
        if orig_reset is not None:
            def reset_probe(self, _orig=orig_reset):
                _flush()
                return _orig(self)
            cls.reset = reset_probe

        orig_finalize = getattr(cls, "finalize", None)
        if orig_finalize is not None:
            def finalize_probe(self, _orig=orig_finalize):
                _flush()
                return _orig(self)
            cls.finalize = finalize_probe

    for cls in (StreamingDetokenizer, NaiveStreamingDetokenizer,
                BPEStreamingDetokenizer, SPMStreamingDetokenizer):
        wrap(cls)

    # CRITICAL: save the ORIGINAL stream_generate BEFORE rebinding, so
    # the wrap's orig_stream points to the real generator function (and
    # not to the wrap itself, which would cause infinite recursion).
    _original = _mlx_lm_generate.stream_generate

    def _resolve_detok(*args, **kwargs):
        for tok in args[1:3]:
            d = getattr(tok, "detokenizer", None) or getattr(
                tok, "_detok", None)
            if d is not None:
                return d
        return None

    def stream_probe(*args, **kwargs):
        state["detok"] = _resolve_detok(*args, **kwargs)
        gen = _original(*args, **kwargs)
        try:
            yielded = next(gen)
        except StopIteration:
            if state["detok"] is not None:
                state["detok"].finalize()
            return
        try:
            while True:
                try:
                    yield yielded
                except GeneratorExit:
                    if state["detok"] is not None:
                        state["detok"].finalize()
                    raise
                yielded = next(gen)
        except StopIteration:
            if state["detok"] is not None:
                state["detok"].finalize()

    _mlx_lm_generate.stream_generate = stream_probe
    try:
        import mlx_omarchy_bonsai2.server as _bonsai_server
        _bonsai_server.stream_generate = stream_probe
    except Exception as exc:
        print(f"shim: bonsai2 server rebind failed: {exc}",
              file=sys.stderr, flush=True)
    print("shim: ids probe installed (bonsai2)", file=sys.stderr, flush=True)
    return state
