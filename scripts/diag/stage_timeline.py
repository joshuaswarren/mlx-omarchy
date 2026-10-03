#!/usr/bin/env python3
"""Per-stage timeline of ONE KokoroPipeline.infer call (pre-registered
infer-floor entry, D1). Forces an mx.eval after each front-half stage so
GPU time attributes to the stage that built it; the decoder is timed
whole (it is the streaming-relevant unit), then re-timed warm. Also
counts mx.eval calls and their blocking time via a counting wrapper."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, sys.argv[3])
from mlx_omarchy_assistant.synthesis import (  # noqa: E402
    KOKORO_PACK, _kokoro_install_trig_reduction, _kokoro_runtime)


def say(o):
    print(json.dumps(o), flush=True)


assets = str(Path(sys.argv[1]) / "voice" / KOKORO_PACK["id"])
TEXT = "The meeting starts at nine and the review follows at eleven."

_kokoro_install_trig_reduction()
import espeakng_loader  # noqa: E402
from phonemizer.backend.espeak.wrapper import EspeakWrapper  # noqa: E402
import mlx.core as mx  # noqa: E402
from mlx_audio.tts.models.kokoro.pipeline import KokoroPipeline  # noqa: E402

EspeakWrapper.set_library(espeakng_loader.get_library_path())
EspeakWrapper.set_data_path(espeakng_loader.get_data_path())

t0 = time.perf_counter()
pipe = _kokoro_runtime(assets)
say({"stage": "runtime_load_s", "v": round(time.perf_counter() - t0, 3)})

model = pipe.model
ps = " ".join(next(iter(["The meeting starts at nine and the review "
                         "follows at eleven."])))
# tokenize exactly like the pipeline does
graphemes = TEXT
_, tokens = pipe.g2p(graphemes)
phonemes = next(ps for gs, ps, tks in pipe.en_tokenize(tokens) if ps)
say({"stage": "phonemes", "n": len(phonemes), "text": phonemes[:60]})

pack = pipe.voices["af_heart"]
ref_s = pack

# mx.eval counter
eval_stats = {"n": 0, "s": 0.0}
real_eval = mx.eval


def counting_eval(*args):
    t = time.perf_counter()
    real_eval(*args)
    eval_stats["n"] += 1
    eval_stats["s"] += time.perf_counter() - t


mx.eval = counting_eval

# Warm one full call first (primer-equivalent), then time a warm call
# stage by stage with forced syncs.
t0 = time.perf_counter()
_ = model(phonemes, ref_s, speed=1.0)
say({"stage": "first_call_total_s", "v": round(time.perf_counter() - t0, 3)})

REPS = 3
for rep in range(REPS):
    t = {}
    t0 = time.perf_counter()
    input_ids = mx.array([[0, *filter(None, map(lambda p: model.vocab.get(p),
                                                phonemes)), 0]])
    t["ids"] = time.perf_counter() - t0
    input_lengths = mx.array([input_ids.shape[-1]])
    text_mask = mx.arange(int(input_lengths.max()))[None, ...]
    text_mask = mx.repeat(text_mask, input_lengths.shape[0], axis=0)
    text_mask = (text_mask + 1 > input_lengths[:, None]).astype(model.dtype
                                                                if hasattr(model, "dtype") else text_mask.dtype)
    t0 = time.perf_counter()
    bert_dur, _ = model.bert(input_ids,
                             attention_mask=(~text_mask).astype(mx.int32))
    mx.eval(bert_dur)
    t["bert"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    d_en = model.bert_encoder(bert_dur).transpose(0, 2, 1)
    s = ref_s[:, 128:]
    d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
    x, _ = model.predictor.lstm(d)
    duration = model.predictor.duration_proj(x)
    duration = mx.sigmoid(duration).sum(axis=-1) / 1.0
    duration = mx.clip(mx.round(mx.nan_to_num(
        duration, nan=1.0)), a_min=1, a_max=100).astype(mx.int32)[0]
    mx.eval(duration)
    t["predictor_dur"] = time.perf_counter() - t0
    indices_list = []
    for i, n in enumerate(duration):
        c = min(max(int(n), 0), 100)
        if c > 0:
            indices_list.append(mx.repeat(mx.array(i), c))
    indices = mx.concatenate(indices_list)
    T = int(indices.shape[0])
    pred_aln_trg = mx.zeros((input_ids.shape[1], T))
    t0 = time.perf_counter()
    pred_aln_trg = pred_aln_trg[None, :]
    pred_aln_trg = mx.array(pred_aln_trg)
    pred_aln_trg[indices, mx.arange(T)] = 1
    pred_aln_trg = pred_aln_trg[None, :]
    en = d.transpose(0, 2, 1) @ pred_aln_trg
    F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
    t_en = model.text_encoder(input_ids, input_lengths, text_mask)
    asr = t_en @ pred_aln_trg
    mx.eval(asr, F0_pred, N_pred)
    t["align_f0_textenc"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    audio = model.decoder(asr, F0_pred, N_pred, ref_s[:, :128])[0]
    mx.eval(audio)
    t["decoder_incl_istft"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    audio2 = model.decoder(asr, F0_pred, N_pred, ref_s[:, :128])[0]
    mx.eval(audio2)
    t["decoder_again"] = time.perf_counter() - t0
    frames = int(audio2.shape[-1]) if audio2 is not None else 0
    say({"stage": f"timed_warm_{rep}", "frames": frames,
         "T_aligned": T,
         "ms": {k: round(v * 1000, 1) for k, v in t.items()},
         "mx_evals": eval_stats["n"], "mx_eval_s": round(eval_stats["s"], 3)})

mx.eval = real_eval
say({"stage": "done"})
