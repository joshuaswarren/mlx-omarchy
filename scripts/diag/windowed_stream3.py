#!/usr/bin/env python3
"""Windowed-vocoder streaming v3: har-frame windows (measured shapes).

Measured map (shape dump v2, M2): asr (1,512,155) aligned frames; F0/N
(1,310) at 2x; decode_block_3 upsamples x to 310; generator ups x300 to
audio (93000); har (STFT-domain) 18601 frames = 5 audio samples each =
120 har-frames per aligned frame. All generator ops are conv/STFT-local;
SineGen phase + noise are precomputed whole (source module runs once on
the full F0); InstanceNorm stats are per-window (the measured
perturbation; corr comparator = same-wheel 0.9895)."""
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
TEXT = sys.argv[4] if len(sys.argv) > 4 else (
    "The meeting starts at nine and the review follows at eleven.")

_kokoro_install_trig_reduction()
import espeakng_loader  # noqa: E402
from phonemizer.backend.espeak.wrapper import EspeakWrapper  # noqa: E402
import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

EspeakWrapper.set_library(espeakng_loader.get_library_path())
EspeakWrapper.set_data_path(espeakng_loader.get_data_path())

pipe = _kokoro_runtime(assets)
model = pipe.model
gen = model.decoder
G = gen.generator

captured = {}


class CaptureDecoder:
    def __init__(self, target):
        self.target = target

    def __call__(self, asr, F0_pred, N_pred, s):
        captured.update(asr=asr, F0=F0_pred, N=N_pred, s=s)
        return mx.zeros((1, 2))


real_decoder = model.decoder
model.decoder = CaptureDecoder(real_decoder)
for _ in pipe(TEXT, voice="af_heart", speed=1.0):
    pass
model.decoder = real_decoder
asr = captured["asr"]
F0_pred = captured["F0"]
N_pred = captured["N"]
s = captured["s"]
T_full = int(asr.shape[-1])
say({"stage": "captured", "T_full": T_full})

# whole-call reference (warm)
_ = real_decoder(asr, F0_pred, N_pred, s)[0]
t0 = time.perf_counter()
whole = real_decoder(asr, F0_pred, N_pred, s)[0]
mx.eval(whole)
t_whole = time.perf_counter() - t0
whole_np = np.asarray(whole).reshape(-1)
say({"stage": "whole_call_s", "v": round(t_whole, 3),
     "samples": int(whole_np.shape[0])})

# --- source + har once (whole F0 curve) ---
t0 = time.perf_counter()
f0_up = G.f0_upsamp(F0_pred[:, None].transpose(0, 2, 1))     # (1, 93000, 1)
har_source, _noi, _uv = G.m_source(f0_up)
har_source = mx.squeeze(har_source.transpose(0, 2, 1), axis=1)
har_spec, har_phase = G.stft.transform(har_source)
har = mx.concatenate([har_spec, har_phase], axis=1).swapaxes(2, 1)
mx.eval(har)
T_HAR = int(har.shape[1])
say({"stage": "source_once_s", "v": round(time.perf_counter() - t0, 3),
     "T_har": T_HAR})

# measured mapping: 1 aligned frame = 600 audio samples = 120 har frames
ALIGNED_TO_HAR = T_HAR // T_full
samples_per_aligned = whole_np.shape[0] // T_full
say({"stage": "map", "aligned_to_har": ALIGNED_TO_HAR,
     "samples_per_aligned": samples_per_aligned})

WINDOW_ALIGNED = 24          # ~0.5 s of audio per window
RF_ALIGNED = 24              # receptive-field overlap each side

pieces = []
timings = []
pos = 0
t_stream = time.perf_counter()
first_piece_s = None
while pos < T_full:
    w0 = max(0, pos - RF_ALIGNED)
    w1 = min(T_full, pos + WINDOW_ALIGNED)
    h0 = min(T_HAR, max(0, w0) * ALIGNED_TO_HAR)
    h1 = min(T_HAR, w1 * ALIGNED_TO_HAR)  # no +1: we drop the reflection pad
    t0 = time.perf_counter()
    f0c = gen.F0_conv(F0_pred[:, None, 2 * w0:2 * w1].transpose(
        0, 2, 1), mx.conv1d).transpose(0, 2, 1)
    nc = gen.N_conv(N_pred[:, None, 2 * w0:2 * w1].transpose(
        0, 2, 1), mx.conv1d).transpose(0, 2, 1)
    xw = mx.concatenate([asr[..., w0:w1], f0c, nc], axis=1)
    xw = gen.encode(xw, s)
    asr_res = gen.asr_res[0](asr[..., w0:w1].transpose(0, 2, 1),
                             mx.conv1d).transpose(0, 2, 1)
    res = True
    for block in gen.decode:
        if res:
            xw = mx.concatenate([xw, asr_res, f0c, nc], axis=1)
        xw = block(xw, s)
        if hasattr(block, "upsample_type") and block.upsample_type != "none":
            res = False
    xg = xw
    for i in range(G.num_upsamples):
        xg = mx.where(xg > 0, xg, xg * 0.1)
        x_source = G.noise_convs[i](har[:, h0:h1, :]).transpose(0, 2, 1)
        x_source = G.noise_res[i](x_source, s)
        xg = G.ups[i](xg.transpose(0, 2, 1), mx.conv_transpose1d).transpose(
            0, 2, 1)
        if False and i == G.num_upsamples - 1:
            xg = G.reflection_pad(xg)
        xg = xg + x_source
        xs = None
        for j in range(G.num_kernels):
            r = G.resblocks[i * G.num_kernels + j](xg, s)
            xs = r if xs is None else xs + r
        xg = xs / G.num_kernels
    xg = mx.where(xg > 0, xg, xg * 0.01)
    xg = G.conv_post(xg.transpose(0, 2, 1), mx.conv1d).transpose(0, 2, 1)
    spec = mx.exp(xg[:, : G.post_n_fft // 2 + 1, :])
    phase = mx.sin(xg[:, G.post_n_fft // 2 + 1:, :])
    win_audio = G.stft.inverse(spec, phase)  # (B, F, T) expected
    mx.eval(win_audio)
    dt = time.perf_counter() - t0
    if first_piece_s is None:
        first_piece_s = time.perf_counter() - t_stream
    timings.append(round(dt * 1000, 1))
    a = np.asarray(win_audio).reshape(-1)
    left = (pos - w0) * samples_per_aligned
    piece = a[left:]
    pieces.append(piece)
    say({"stage": "window", "pos": pos, "w0": w0, "w1": w1, "h0": h0,
         "h1": h1, "decode_ms": round(dt * 1000, 1),
         "samples": int(piece.shape[0])})
    pos += WINDOW_ALIGNED

streamed = np.concatenate(pieces)
n = min(streamed.shape[0], whole_np.shape[0])
a, b = streamed[:n], whole_np[:n]
corr = float(np.corrcoef(a, b)[0, 1])
max_abs = float(np.max(np.abs(a - b)))
seams = []
idx = 0
for p in pieces[:-1]:
    idx += p.shape[0]
    if 1 < idx < n - 1:
        seams.append(abs(float(streamed[idx]) - float(streamed[idx - 1])))
local = np.abs(np.diff(b))
say({"stage": "compare", "corr": round(corr, 5), "noise_floor": 0.9895,
     "max_abs_diff": round(max_abs, 5),
     "seam_jumps_us": [round(s * 1e6, 1) for s in seams],
     "median_local_jump_us": round(float(np.median(local)) * 1e6, 1),
     "max_window_decode_ms": max(timings),
     "first_piece_s": round(first_piece_s, 3),
     "stream_total_s": round(time.perf_counter() - t_stream, 3),
     "samples_streamed": int(streamed.shape[0]),
     "samples_whole": int(whole_np.shape[0])})
say({"stage": "done"})
