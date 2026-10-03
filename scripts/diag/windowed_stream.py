#!/usr/bin/env python3
"""Windowed-vocoder streaming wrapper (pre-registered infer-floor lever 4).

Design (mlx_audio 0.5.6, pinned ==0.5.6):
- FRONT HALF once per sentence: bert -> predictor -> alignment -> F0/N ->
  text encoder -> asr (replicates KokoroModel.__call__ steps 1-6), one
  mx.eval to materialize asr/F0/N.
- SOURCE once per sentence: f0_upsamp(F0_pred) -> m_source(f0_up) gives
  the harmonic source at audio rate. SineGen's phase is a cumsum over
  time, so it MUST be computed on the whole curve and sliced per window
  (a per-window sine gen would reset phase and click at the seams).
  The STFT-domain harmonic (har) is then computed once and sliced too.
- DECODER per time window: the conv stack + resblocks + conv_post +
  iSTFT are strictly local in time (conv kernels <= 7, stft window 2048),
  so a window with receptive-field overlap produces output identical to
  the whole-call result inside the trimmed region — except:
    (a) InstanceNorm (AdaIN) statistics are per-WINDOW instead of
        per-utterance (time-global mean/var) — measured, not assumed:
        corr(windowed, whole) must clear the same-wheel noise floor;
    (b) mx.random noise draws differ per call — the same class of
        perturbation as two whole-call runs (0.9895 same-wheel floor).
- Each window's audio is mx.eval'd as soon as it is decoded and yielded.

This wrapper is measurement/lever code: it couples to pinned mlx_audio
0.5.6 internals by calling the model's own submodules (no fork).
"""
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
TEXT = sys.argv[4] if len(sys.argv) > 4 else (
    "The meeting starts at nine and the review follows at eleven.")

_kokoro_install_trig_reduction()
import espeakng_loader  # noqa: E402
from phonemizer.backend.espeak.wrapper import EspeakWrapper  # noqa: E402
import mlx.core as mx  # noqa: E402

EspeakWrapper.set_library(espeakng_loader.get_library_path())
EspeakWrapper.set_data_path(espeakng_loader.get_data_path())

pipe = _kokoro_runtime(assets)
model = pipe.model

# --- front half (replicates KokoroModel.__call__ steps 1-6) ---
_, tokens = pipe.g2p(TEXT)
phonemes = next(ps for gs, ps, tks in pipe.en_tokenize(tokens) if ps)
input_ids = mx.array([[0, *filter(None, map(model.vocab.get, phonemes)), 0]])
input_lengths = mx.array([input_ids.shape[-1]])
text_mask = mx.arange(int(input_lengths.max()))[None, ...]
text_mask = mx.repeat(text_mask, input_lengths.shape[0], axis=0)
text_mask = text_mask + 1 > input_lengths[:, None]
bert_dur, _ = model.bert(input_ids,
                         attention_mask=(~text_mask).astype(mx.int32))
d_en = model.bert_encoder(bert_dur).transpose(0, 2, 1)
ref_s = pipe.voices["af_heart"]
s = ref_s[:, 128:]
d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
x, _ = model.predictor.lstm(d)
duration = model.predictor.duration_proj(x)
duration = mx.clip(mx.round(mx.sigmoid(duration).sum(axis=-1)), 1, 100)
duration = mx.round(duration).astype(mx.int32)[0]
indices_list = []
for i, n in enumerate(duration):
    c = min(max(int(n), 0), 100)
    if c > 0:
        indices_list.append(mx.repeat(mx.array(i), c))
indices = mx.concatenate(indices_list)
T = int(indices.shape[0])
pred_aln = mx.zeros((1, input_ids.shape[1], T))
flat = pred_aln.reshape(T, -1)
flat[indices, mx.arange(T)] = 1
pred_aln = flat.reshape(1, input_ids.shape[1], T)
en = d.transpose(0, 2, 1) @ pred_aln
F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
t_en = model.text_encoder(input_ids, input_lengths, text_mask)
asr = t_en @ pred_aln
mx.eval(asr, F0_pred, N_pred)
say({"stage": "front_half_done", "T_frames": T})

gen = model.decoder.generator
hop = gen.stft.hop_length

# --- source + STFT-domain harmonic once for the whole sentence ---
t0 = time.perf_counter()
f0_up = gen.f0_upsamp(F0_pred[:, None].transpose(0, 2, 1))
har_source, _noi, _uv = gen.m_source(f0_up)
har_source = mx.squeeze(har_source.transpose(0, 2, 1), axis=1)
har_spec, har_phase = gen.stft.transform(har_source)
har = mx.concatenate([har_spec, har_phase], axis=1).swapaxes(2, 1)
mx.eval(har)
say({"stage": "source_once_s", "v": round(time.perf_counter() - t0, 3),
     "har_frames": int(har.shape[1])})

ups_product = 1
for u in gen.upsample_rates:
    ups_product *= u
stft_frames = int(har.shape[1])
x_frames = stft_frames // ups_product
say({"stage": "frame_math", "ups_product": ups_product,
     "x_frames": x_frames})

RF_FRAMES = 64          # receptive-field overlap on the x axis
WINDOW_X = 96           # ~1 s of audio per window at 256 hop x ups

# whole-call reference
t0 = time.perf_counter()
whole = model.decoder(asr, F0_pred, N_pred, ref_s[:, :128])[0]
mx.eval(whole)
t_whole = time.perf_counter() - t0
whole_np = __import__("numpy").asarray(whole).reshape(-1)
say({"stage": "whole_call_s", "v": round(t_whole, 3),
     "samples": int(whole_np.shape[0])})

# --- windowed decode ---
import numpy as np  # noqa: E402

pieces = []
pos = 0
t_stream_start = time.perf_counter()
first_piece_s = None
while pos < x_frames:
    w0 = max(0, pos - RF_FRAMES)
    w1 = min(x_frames, pos + WINDOW_X)
    x_win = asr[:, :, w0:w1]
    f0_win = F0_pred[:, w0 * ups_product:(w1 * ups_product) if w1 < x_frames
                     else None]
    n_win = N_pred[:, w0 * ups_product:(w1 * ups_product) if w1 < x_frames
                   else None]
    a0 = (w0 * ups_product) if w1 < x_frames else 0
    a1 = (stft_frames if w1 >= x_frames
          else w1 * ups_product)
    har_win = har[:, a0:a1, :]
    t0 = time.perf_counter()
    # replicate Generator.__call__ with the sliced precomputed harmonic
    s_full = ref_s[:, :128]
    xw = gen.F0_conv(
        f0_win[:, None, :].swapaxes(2, 1), mx.conv1d).swapaxes(2, 1)
    nw = gen.N_conv(
        n_win[:, None, :].swapaxes(2, 1), mx.conv1d).swapaxes(2, 1)
    xcat = mx.concatenate([x_win, xw, nw], axis=1)
    xcat = gen.encode(xcat, s_full)
    asr_res = gen.asr_res[0](x_win.swapaxes(2, 1), mx.conv1d).swapaxes(2, 1)
    res = True
    for block in gen.decode:
        if res:
            xcat = mx.concatenate([xcat, asr_res, xw, nw], axis=1)
        xcat = block(xcat, s_full)
        if hasattr(block, "upsample_type") and block.upsample_type != "none":
            res = False
    # generator loop with the sliced harmonic
    xg = xcat
    for i in range(gen.num_upsamples):
        xg = mx.leaky_relu(xg, negative_slope=0.1)
        h_i = har[:, a0:a1, :]
        x_source = gen.noise_convs[i](h_i)
        x_source = x_source.swapaxes(2, 1)
        x_source = gen.noise_res[i](x_source, s_full)
        xg = xg.swapaxes(2, 1)
        xg = gen.ups[i](xg, mx.conv_transpose1d)
        xg = xg.swapaxes(2, 1)
        if i == gen.num_upsamples - 1:
            xg = gen.reflection_pad(xg)
        xg = xg + x_source
        xs = None
        for j in range(gen.num_kernels):
            r = gen.resblocks[i * gen.num_kernels + j](xg, s_full)
            xs = r if xs is None else xs + r
        xg = xs / gen.num_kernels
    xg = mx.leaky_relu(xg, negative_slope=0.01)
    xg = xg.swapaxes(2, 1)
    xg = gen.conv_post(xg, mx.conv1d)
    xg = xg.swapaxes(2, 1)
    spec = mx.exp(xg[:, : gen.post_n_fft // 2 + 1, :])
    phase = mx.sin(xg[:, gen.post_n_fft // 2 + 1:, :])
    win_audio = gen.stft.inverse(spec, phase)
    mx.eval(win_audio)
    dt = time.perf_counter() - t0
    if first_piece_s is None:
        first_piece_s = time.perf_counter() - t_stream_start
    a_np = np.asarray(win_audio).reshape(-1)
    # trim the receptive field on the left (interior seams only)
    left = (pos - w0) * hop * ups_product // 1
    left = min(left, a_np.shape[0])
    piece = a_np[left:]
    pieces.append(piece)
    say({"stage": "window", "pos": pos, "w0": w0, "w1": w1,
         "decode_s": round(dt, 3), "samples": int(piece.shape[0])})
    pos += WINDOW_X

streamed = np.concatenate(pieces)
n = min(streamed.shape[0], whole_np.shape[0])
a = streamed[:n]
b = whole_np[:n]
corr = float(np.corrcoef(a, b)[0, 1])
max_abs = float(np.max(np.abs(a - b)))
say({"stage": "compare", "corr": round(corr, 5),
     "noise_floor": 0.9895, "max_abs_diff": round(max_abs, 5),
     "first_piece_s": round(first_piece_s, 3),
     "stream_total_s": round(time.perf_counter() - t_stream_start, 3),
     "samples_streamed": int(streamed.shape[0]),
     "samples_whole": int(whole_np.shape[0])})
say({"stage": "done"})
