#!/usr/bin/env python3
"""Per-stage timeline of the Kokoro infer path (D1, v2 — shim based).

Each pipeline-internal module is replaced by a shim that times the call
and forces mx.eval on its output, so GPU work attributes to the stage
that built it. Pass 1 warms (cold numbers kept separately); passes 2-3
are the warm table. The decoder is timed whole (it is the
streaming-relevant unit) and once more warm to expose per-call fixed
cost."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, sys.argv[3])
from mlx_omarchy_assistant.synthesis import (  # noqa: E402
    KOKORO_PACK, _kokoro_install_trig_reduction, _kokoro_runtime)


def say(o):
    print(json.dumps(o), flush=True)


assets = str(Path(sys.argv[1]) / "kokoro-82m-bf16")

_kokoro_install_trig_reduction()
import espeakng_loader  # noqa: E402
from phonemizer.backend.espeak.wrapper import EspeakWrapper  # noqa: E402
import mlx.core as mx  # noqa: E402

EspeakWrapper.set_library(espeakng_loader.get_library_path())
EspeakWrapper.set_data_path(espeakng_loader.get_data_path())

pipe = _kokoro_runtime(assets)
model = pipe.model
TEXT = "The meeting starts at nine and the review follows at eleven."

real_eval = mx.eval
stats = {"n": 0, "s": 0.0}


def counting_eval(*a, **k):
    t = time.perf_counter()
    real_eval(*a, **k)
    stats["n"] += 1
    stats["s"] += time.perf_counter() - t


mx.eval = counting_eval


class Shim:
    def __init__(self, target, name, records, eval_out=True):
        self.target = target
        self.name = name
        self.records = records
        self.eval_out = eval_out

    def __call__(self, *a, **k):
        t = time.perf_counter()
        out = self.target(*a, **k)
        if self.eval_out:
            real_eval(out)
        dt = time.perf_counter() - t
        self.records.append({"stage": self.name,
                             "ms": round(dt * 1000, 1)})
        return out


STAGES = [
    (model, "bert", "bert"),
    (model, "bert_encoder", "bert_encoder"),
    (model.predictor, "text_encoder", "pred_text_enc"),
    (model.predictor, "lstm", "pred_lstm"),
    (model.predictor, "duration_proj", "duration_proj"),
    (model.predictor, "F0Ntrain", "f0n"),
    (model, "text_encoder", "text_encoder"),
    (model, "decoder", "decoder_whole"),
]

records = []
undo = []
for parent, attr, name in STAGES:
    target = getattr(parent, attr)
    shim = Shim(target, name, records)
    setattr(parent, attr, shim)
    undo.append((parent, attr, target))

# warm the full path once (cold numbers recorded but not part of the table)
for _ in pipe(TEXT, voice="af_heart", speed=1.0):
    pass

for rep in range(2):
    records.clear()
    stats["n"] = 0
    stats["s"] = 0.0
    n_chunks = 0
    for _ in pipe(TEXT, voice="af_heart", speed=1.0):
        n_chunks += 1
    say({"stage": f"warm_{rep}", "chunks": n_chunks,
         "mx_evals": stats["n"], "mx_eval_s": round(stats["s"], 3),
         "rows": records})

for parent, attr, target in undo:
    setattr(parent, attr, target)
mx.eval = real_eval
say({"stage": "done"})
