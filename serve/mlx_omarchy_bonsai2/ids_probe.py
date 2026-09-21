"""Env-gated ids-probe hook for the bonsai2 managed shim (bench-only).

Default OFF: no behavior change. With MLX_OMARCHY_SERVE_IDS_PROBE=1, the
hook captures the raw sampled token ids at the per-token boundary and
emits one PROBE: JSON line per request at reset/finalize. Mirrors the
_mlxlm_server install_ids_probe contract so a single harness can
exercise both managed routes."""
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
    print("shim: ids probe installed (bonsai2)", file=sys.stderr, flush=True)
    return state
