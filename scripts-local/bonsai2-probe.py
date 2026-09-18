#!/usr/bin/env python3
"""Ternary-Bonsai-2-27B pack probe: load via bundled runtime/ artifact loader,
generate text-only on the candidate wheel. One JSON line on stdout."""
import json
import sys
import time
import traceback

import mlx.core as mx

snap = sys.argv[1]
max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 16

sys.path.insert(0, snap.rstrip("/") + "/runtime")

out = {"id": "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit (runtime/ artifact loader)"}
try:
    t0 = time.time()
    from artifact import load_model  # noqa: E402  (pack-local loader)

    model, config = load_model(snap)
    out["load_s"] = round(time.time() - t0, 2)
    out["arch"] = config.get("model_type")

    from mlx_lm import generate
    from mlx_lm.tokenizer_utils import load as load_tokenizer

    tokenizer = load_tokenizer(snap)
    t1 = time.time()
    text = generate(
        model, tokenizer, "The capital of France is", max_tokens=max_tokens
    )
    out["status"] = "GENERATED"
    out["gen_s"] = round(time.time() - t1, 2)
    out["sample"] = text[:120].replace("\n", " ")
except Exception as e:
    out["status"] = "FAILED"
    out["error"] = f"{type(e).__name__}: {e}"
    out["trace"] = traceback.format_exc(limit=4)
print(json.dumps(out), flush=True)
