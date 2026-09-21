"""Env-gated ids-probe hook for the bonsai2 managed shim (bench-only).

Default OFF: no behavior change. With MLX_OMARCHY_SERVE_IDS_PROBE=1, the
hook captures the raw sampled token ids at the per-token boundary and
emits one PROBE: JSON line per request. The bonsai2 server's iteration
loop (for response in stream_generate(...): break on finish_reason) never
calls detok.finalize explicitly, so the hook wraps stream_generate and
force-finalizes the detokenizer on generator close.

The bonsai2 server captures stream_generate via
`from mlx_lm.generate import stream_generate` (a function-body from-
import that is re-executed on every call). The from-import does
`import mlx_lm.generate` (submodule) + `getattr(submodule, 'stream_generate')`.
So the wrap must rebind BOTH the submodule's `.stream_generate` attr AND
the package-level `mlx_lm.generate` attribute. The original is saved
BEFORE rebinding to avoid infinite recursion."""
from __future__ import annotations
import hashlib
import json
import os
import sys

PROBE_ENV = "MLX_OMARCHY_SERVE_IDS_PROBE"


def install_ids_probe(emit=None):
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
        import mlx_lm as _mlx_lm_pkg
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

    # CRITICAL: resolve stream_generate. In mlx_lm 0.31.3, mlx_lm.generate
    # may be the function (re-exported at the package level, in which case
    # _mlx_lm_generate has no .stream_generate attr) or a submodule. Try
    # both forms; whichever has a callable stream_generate attr wins.
    _original = None
    if hasattr(_mlx_lm_generate, "stream_generate"):
        _original = _mlx_lm_generate.stream_generate
    elif hasattr(_mlx_lm_pkg, "generate") and hasattr(_mlx_lm_pkg.generate, "stream_generate"):
        _original = _mlx_lm_pkg.generate.stream_generate
    if _original is None:
        print("shim: could not resolve stream_generate from mlx_lm.generate "
              "or mlx_lm_pkg.generate", file=sys.stderr, flush=True)
        return None

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

    # Rebind all three sites: the generate submodule, the package
    # attribute, and the bonsai2 server module. The bonsai2 server's
    # function-body `from mlx_lm.generate import stream_generate` is
    # re-executed on every call — it reads the submodule's
    # `.stream_generate` attr, so the submodule rebind is the one that
    # counts for the per-call capture.
    _mlx_lm_generate.stream_generate = stream_probe
    _mlx_lm_pkg.generate = stream_probe
    try:
        import mlx_omarchy_bonsai2.server as _bonsai_server
        _bonsai_server.stream_generate = stream_probe
    except Exception as exc:
        print(f"shim: bonsai2 server rebind failed: {exc}",
              file=sys.stderr, flush=True)
    print("shim: ids probe installed (bonsai2)", file=sys.stderr, flush=True)
    return state
