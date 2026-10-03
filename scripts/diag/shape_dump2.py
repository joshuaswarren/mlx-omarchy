#!/usr/bin/env python3
"""Shape dump v2: robust against mlx Module attribute quirks.
Patches only plain attributes / list items; wraps whole callables."""
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

shapes = {}
undo = []


def mk_shim(name, target):
    def shim(*a, **k):
        outs = target(*a, **k)
        rec = {"in": [list(getattr(t, "shape", ())) for t in a
                      if hasattr(t, "shape")]}
        first = outs[0] if isinstance(outs, tuple) else outs
        rec["out"] = list(getattr(first, "shape", ())) if hasattr(
            first, "shape") else None
        shapes.setdefault(name, rec)
        return outs
    return shim


def patch_callable(parent, attr, name):
    target = getattr(parent, attr)
    if not callable(target):
        shapes[name] = {"not_callable": True}
        return
    setattr(parent, attr, mk_shim(name, target))
    undo.append((parent, attr, target))


def patch_list_items(parent, attr, prefix):
    lst = getattr(parent, attr)
    if not isinstance(lst, list):
        shapes[prefix] = {"not_a_list": type(lst).__name__}
        return
    for i, item in enumerate(lst):
        if callable(item):
            lst[i] = mk_shim(f"{prefix}_{i}", item)
    undo.append((parent, attr, lst))


D = model.decoder
G = D.generator

patch_callable(D, "F0_conv", "F0_conv")
patch_callable(D, "N_conv", "N_conv")
patch_callable(D, "encode", "encode")
patch_callable(D, "asr_res", "asr_res")
patch_list_items(D, "decode", "decode_block")
patch_callable(G, "f0_upsamp", "f0_upsamp")
patch_callable(G, "m_source", "m_source")
patch_list_items(G, "ups", "ups")
patch_list_items(G, "noise_convs", "noise_conv")
patch_list_items(G, "noise_res", "noise_res")
patch_callable(G, "conv_post", "conv_post")
if hasattr(G, "stft"):
    t = getattr(G.stft, "transform", None)
    if callable(t):
        G.stft.transform = mk_shim("stft_transform", t)
    inv = getattr(G.stft, "inverse", None)
    if callable(inv):
        G.stft.inverse = mk_shim("stft_inverse", inv)

n_chunks = 0
for _ in pipe(TEXT, voice="af_heart", speed=1.0):
    n_chunks += 1
say({"stage": "shapes", "chunks": n_chunks, "data": shapes})

for parent, attr, target in undo:
    if isinstance(target, list):
        setattr(parent, attr, target)
    else:
        setattr(parent, attr, target)
say({"stage": "done"})
