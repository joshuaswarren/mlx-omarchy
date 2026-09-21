"""Env-gated ids-probe hook for the bonsai2 managed shim (bench-only).

Default OFF: no behavior change. With MLX_OMARCHY_SERVE_IDS_PROBE=1, the
hook captures the raw sampled token ids at the per-token boundary and
emits one PROBE: JSON line per request. The bonsai2 server's iteration
loop (for response in stream_generate(...): break on finish_reason) never
calls detok.finalize() explicitly, so the class-level finalize hook would
never fire — wrap stream_generate itself and force-finalize the
detokenizer on generator close."""
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
    except Exception as exc:
        print(f"shim: ids probe install failed: {exc}", file=sys.stderr,
              flush=True)
        return None
    state: dict = {"ids": []}

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

    # Wrap stream_generate to force-finalize the detok on generator close.
    # The bonsai2 server iterates stream_generate and breaks on
    # finish_reason; the per-detok finalize is never called from the
    # call site, so the class-level finalize hook alone is insufficient.
    # A GeneratorExit fires when the for-loop exits (break / completion);
    # we use that to flush the most recently-seen detok.
    orig_stream = getattr(_mlx_lm_generate, "stream_generate", None)
    last_detok = {"obj": None}

    def stream_probe(*args, **kwargs):
        gen = orig_stream(*args, **kwargs)
        try:
            yielded = next(gen)
        except StopIteration:
            return
        last_detok["obj"] = getattr(yielded, "token", None) and None  # not the detok
        # The detok is owned by the tokenizer; capture from the call args.
        # stream_generate(model, tokenizer, prompt, ...) — tokenizer is args[1].
        for tok in args[1:3]:
            d = getattr(tok, "detokenizer", None) or getattr(tok, "_detok", None)
            if d is not None:
                last_detok["obj"] = d
                break
        try:
            while True:
                try:
                    yield yielded
                except GeneratorExit:
                    # flush via finalize; the class hook emits via finalize
                    d = last_detok["obj"]
                    if d is not None and hasattr(d, "finalize"):
                        d.finalize()
                    raise
                yielded = next(gen)
        except StopIteration:
            # End of generation: ensure a final flush
            d = last_detok["obj"]
            if d is not None and hasattr(d, "finalize"):
                d.finalize()

    if orig_stream is not None:
        _mlx_lm_generate.stream_generate = stream_probe
    print("shim: ids probe installed (bonsai2)", file=sys.stderr, flush=True)
    return state
