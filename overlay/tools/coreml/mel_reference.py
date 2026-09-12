# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Reference-derived host mel preprocessing for the pinned Parakeet capture.

Derived from ``mweinbach/parakeet-coreml-swift`` @
``75aec2a1c991319657ff4dec5f602c12da6c5012`` (Apache-2.0),
``Sources/ParakeetTDT/MelFeatureExtractor.swift`` and
``MelFilterBank.swift``, reproducing the pinned algorithm op-for-op with
installed NumPy only.

Status: DIAGNOSTIC, not a qualified frontend (plan §62 stop, 2026-09-12).

Per plan §43 this is host scalar/preprocessing DSP; no tensor primitive
runs here and none of this is model inference. The golden capture
(``mel.npy``) was produced by Apple Accelerate ``vDSP_DFT_zrop``
(float32 DFT) on macOS/arm64. The portable NumPy pipeline below computes
the same mathematical transform but cannot reproduce Accelerate's
internal float32 summation order bit-for-bit: across 13 tested precision
variants (float64/float32 FFT, float32-rounded spectra, float32 vs
float64 mel dot, float32 vs float64 log, FMA-emulated preemphasis) the
best reaches 56% bit-identical values with max |Δ| 4.8e-5 on the pinned
LibriSpeech capture. Golden equality is required to be exact, so this
module ships as a first-divergence diagnostic only; no unqualified
frontend lands on main. Run the diagnostic:

    python3 overlay/tools/coreml/mel_reference.py <capture-dir>

Framing semantics note (verified against the golden tail rows): the
Swift source windows ``padded[t*hop .. t*hop+win-1]`` and places the
result at FFT bins ``[56 .. 455]``. That is 56 samples earlier than
``torch.stft(center=True)`` would window the same frame; the golden
capture's silence-tail structure (identical rows from frame 1046, not
1044) confirms the Swift offset is what was captured.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from reference import MelConfig, ReferenceLock

CHUNK_SAMPLES = 3000 * 160  # 30 s chunk, as chunked by ParakeetTranscriber
ENCODER_MAX_TIME = 3000     # encoder input frames per chunk (traced shape)


def load_mel_config(lock_path: Path | None = None) -> MelConfig:
    """Mel config from the pinned lock (never hardcoded here)."""
    lock = ReferenceLock.load(lock_path or Path(__file__).parent / "parakeet-reference.lock")
    return lock.mel


def hann_window(win_length: int) -> np.ndarray:
    """``torch.hann_window(win_length, periodic=False)``, float32."""
    n = np.arange(win_length, dtype=np.float32)
    return (np.float32(0.5) - np.float32(0.5)
            * np.cos(np.float32(2.0) * np.float32(np.pi) * n
                     / np.float32(win_length - 1))).astype(np.float32)


def mel_filterbank(cfg: MelConfig) -> np.ndarray:
    """Slaney-normalised mel filterbank, float64 math cast to float32.

    Same construction as ``librosa.filters.mel(htk=False, norm="slaney")``
    and ``MelFilterBank.swift``: rows [n_mels, n_fft/2 + 1].
    """
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    def hz_to_mel(hz):
        with np.errstate(divide="ignore"):
            return np.where(hz >= min_log_hz,
                            min_log_mel + np.log(hz / min_log_hz) / logstep,
                            hz / f_sp)
    def mel_to_hz(mel):
        return np.where(mel >= min_log_mel,
                        min_log_hz * np.exp(logstep * (mel - min_log_mel)),
                        f_sp * mel)

    f_min, f_max = 0.0, cfg.sample_rate / 2.0
    mels = hz_to_mel(f_min) + (hz_to_mel(f_max) - hz_to_mel(f_min)) * (
        np.arange(cfg.n_mels + 2) / (cfg.n_mels + 1))
    hz_pts = mel_to_hz(mels)
    fft_freqs = np.arange(cfg.n_fft // 2 + 1) * cfg.sample_rate / cfg.n_fft
    lower, center, upper = hz_pts[:-2], hz_pts[1:-1], hz_pts[2:]
    enorm = 2.0 / (upper - lower)
    tri = np.maximum(
        0.0,
        np.minimum((fft_freqs[None, :] - lower[:, None]) / (center - lower)[:, None],
                   (upper[:, None] - fft_freqs[None, :]) / (upper - center)[:, None]))
    return (tri * enorm[:, None]).astype(np.float32)


def preemphasize(waveform: np.ndarray, cfg: MelConfig) -> np.ndarray:
    """``y[0]=x[0]``; ``y[n]=x[n]-p*x[n-1]``, plain float32 mul-sub.

    The pinned binary compiles without FMA contraction here (FMA-emulated
    arithmetic matches the golden capture strictly worse; see receipt).
    """
    x = np.ascontiguousarray(waveform, dtype=np.float32)
    if x.size == 0:
        raise ValueError("waveform must be non-empty")
    y = np.empty_like(x)
    y[0] = x[0]
    y[1:] = (x[1:] - np.float32(cfg.preemphasis) * x[:-1]).astype(np.float32)
    return y


def stft_frames(waveform: np.ndarray, cfg: MelConfig) -> np.ndarray:
    """Windowed frames exactly as ``MelFeatureExtractor.extract`` lays them out.

    Returns ``(num_frames, n_fft)`` float32. Zero-pads ``n_fft/2`` both
    sides, then per frame t places ``padded[t*hop .. t*hop+win-1] * hann``
    at bins ``[(n_fft-win)/2 .. ]`` (Swift offset; see module docstring).
    """
    pre = preemphasize(waveform, cfg)
    padded = np.zeros(pre.size + cfg.n_fft, np.float32)
    padded[cfg.n_fft // 2:cfg.n_fft // 2 + pre.size] = pre
    num_frames = (padded.size - cfg.n_fft) // cfg.hop_length + 1
    if num_frames <= 0:
        return np.zeros((0, cfg.n_fft), np.float32)
    win_offset = (cfg.n_fft - cfg.win_length) // 2
    idx = np.arange(cfg.win_length)[None, :] + cfg.hop_length * np.arange(num_frames)[:, None]
    frames = np.zeros((num_frames, cfg.n_fft), np.float32)
    frames[:, win_offset:win_offset + cfg.win_length] = (
        padded[idx] * hann_window(cfg.win_length)[None, :]).astype(np.float32)
    return frames


def logmel(frames: np.ndarray, cfg: MelConfig,
           filterbank: np.ndarray | None = None) -> np.ndarray:
    """Power spectrum -> mel projection -> ``log(mel + guard)``, float32.

    The sqrt-then-square two-step form matches the reference round-off
    order. FFT runs in float64 (numpy pocketfft); see module docstring
    for why float32 Accelerate output cannot be reproduced portably.
    """
    fb = mel_filterbank(cfg) if filterbank is None else filterbank
    spec = np.fft.rfft(frames.astype(np.float64), axis=1)
    re = spec.real.astype(np.float32)
    im = spec.imag.astype(np.float32)
    mag = np.sqrt(re * re + im * im).astype(np.float32)
    power = (mag * mag).astype(np.float32)
    mel = (fb.astype(np.float64) @ power.T.astype(np.float64)).T
    return np.log(mel + float(np.float32(cfg.log_guard))).astype(np.float32)


def normalize(logmel_frames: np.ndarray, cfg: MelConfig) -> np.ndarray:
    """Per-mel-bin mean/std over the frame axis, float32 sequential sums.

    Bessel-corrected variance (n-1), ``max(n-1, 1)`` denominator, output
    ``(x - mean) / (std + epsilon)``. Sequential accumulation is emulated
    exactly with float32 ``cumsum`` (the reference loops t in order).
    """
    lm = np.ascontiguousarray(logmel_frames, dtype=np.float32)
    n_t = np.float32(lm.shape[0])
    mean = np.cumsum(lm, axis=0, dtype=np.float32)[-1] / n_t
    dev = (lm - mean[None, :]).astype(np.float32)
    denom = np.float32(max(lm.shape[0] - 1, 1))
    var = np.cumsum((dev * dev).astype(np.float32), axis=0, dtype=np.float32)[-1] / denom
    std = np.sqrt(var).astype(np.float32)
    return ((lm - mean[None, :]) / (std[None, :] + np.float32(cfg.epsilon))).astype(np.float32)


def chunk_for_mel(waveform: np.ndarray) -> np.ndarray:
    """First-reference chunk contract: one 30 s chunk, zero-padded tail."""
    x = np.ascontiguousarray(waveform, dtype=np.float32)
    if x.size == 0:
        raise ValueError("waveform must be non-empty")
    if x.size > CHUNK_SAMPLES:
        raise ValueError("diagnostic covers the single-chunk reference path")
    if x.size < CHUNK_SAMPLES:
        x = np.concatenate([x, np.zeros(CHUNK_SAMPLES - x.size, np.float32)])
    return x


def extract_chunk_features(waveform_30s: np.ndarray, cfg: MelConfig) -> tuple[np.ndarray, np.ndarray]:
    """(mel [3001,128] float32, attention mask [3001] int32) for one chunk.

    Mirrors ``MelFeatureExtractor.extract``: every mel frame is valid
    (mask all ones) because the caller pre-pads the chunk to 30 s.
    """
    frames = stft_frames(chunk_for_mel(waveform_30s), cfg)
    feats = normalize(logmel(frames, cfg), cfg)
    mask = np.ones(frames.shape[0], np.int32)
    return feats, mask


def encoder_slice(feats: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Encoder input contract: first 3000 frames of each chunk.

    ``ModelRunner.runEncoder`` copies ``min(frames, 3000)`` rows into a
    zero-filled [1,3000,128] buffer; the 3001st mel frame is computed
    but never fed. This slice reproduces that.
    """
    return feats[:ENCODER_MAX_TIME], mask[:ENCODER_MAX_TIME]


@dataclass
class DivergenceReport:
    exact: bool
    shape_expected: tuple
    shape_actual: tuple
    bit_exact_count: int
    total: int
    max_abs: float
    mean_abs: float
    first_divergence: tuple | None  # (frame, bin, golden, ours, ulps)
    structural: dict


def compare_capture(capture_dir: Path, lock_path: Path | None = None) -> DivergenceReport:
    """Recompute mel for a golden capture and locate the first divergence.

    Structural contracts are checked exactly; value equality is reported
    honestly (expected: not bit-exact, see module docstring / §62).
    """
    cfg = load_mel_config(lock_path)
    wave = np.load(capture_dir / "waveform.npy")
    golden = np.load(capture_dir / "mel.npy")
    ours, mask = extract_chunk_features(wave, cfg)

    diff = np.abs(ours.astype(np.float64) - golden.astype(np.float64))
    bad = np.argwhere(diff > 0)
    first = None
    if bad.size:
        f, b = bad[0]
        golden_v = np.float32(golden[f, b])
        ulps = abs(np.float32(ours[f, b]) - golden_v) / np.spacing(golden_v)
        first = (int(f), int(b), float(golden_v), float(ours[f, b]), float(ulps))

    # Structural contracts that must hold exactly regardless of FFT rounding.
    g_mask = np.load(capture_dir / "mel_mask.npy")
    g_eif = np.load(capture_dir / "encoder_input_features.npy")
    g_eim = np.load(capture_dir / "encoder_input_mask.npy")
    tail_identical = (golden[1046:] == golden[1046]).all() if golden.shape[0] > 1046 else False
    es, em = encoder_slice(ours, mask)
    structural = {
        "shape_matches": ours.shape == golden.shape,
        "mask_all_ones": bool((mask == 1).all()) and mask.shape == g_mask.shape,
        "mask_equals_golden": bool(np.array_equal(mask, g_mask)),
        "encoder_slice_shape": es.shape == tuple(g_eif.shape[1:]),
        "encoder_slice_mask_equals_golden": bool(np.array_equal(em, g_eim[0])),
        "encoder_slice_features_exact": bool(np.array_equal(es, g_eif[0])),
        "silence_tail_identical_rows_from_1046": bool(tail_identical),
    }

    return DivergenceReport(
        exact=bool(np.array_equal(ours, golden)),
        shape_expected=golden.shape,
        shape_actual=ours.shape,
        bit_exact_count=int((diff == 0).sum()),
        total=int(diff.size),
        max_abs=float(diff.max()),
        mean_abs=float(diff.mean()),
        first_divergence=first,
        structural=structural,
    )


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__.strip())
        return 2
    rep = compare_capture(Path(argv[1]))
    print(f"shape: expected {rep.shape_expected}, computed {rep.shape_actual}")
    print(f"exact golden equality: {rep.exact}")
    print(f"bit-identical values: {rep.bit_exact_count}/{rep.total}"
          f" ({100.0 * rep.bit_exact_count / rep.total:.2f}%)")
    print(f"max |diff|: {rep.max_abs:.6e}   mean |diff|: {rep.mean_abs:.6e}")
    if rep.first_divergence:
        f, b, g, o, u = rep.first_divergence
        print(f"first divergence: frame {f} bin {b}: golden {g!r} vs computed {o!r}"
              f" ({u:.0f} ulps)")
    for k, v in rep.structural.items():
        print(f"structural {k}: {v}")
    if not rep.exact:
        print("VERDICT: portable pipeline is not bit-exact with the Accelerate"
              " vDSP golden (§62 stop). Diagnostic only; no frontend claim.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
