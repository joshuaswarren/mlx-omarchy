# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Focused tests for the reference-derived host mel diagnostic.

Golden-capture tests are skipped when the pinned capture is not cached;
everything else runs from synthetic waveforms.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[2] / "overlay" / "tools" / "coreml"
sys.path.insert(0, str(TOOLS))

import mel_reference as M
import reference as R

CAPTURE = R.default_cache_root() / "captures" / "b650695c-75aec2a" / "20260912T154759Z-librispeech" / "ane"


requires_capture = pytest.mark.skipif(
    not CAPTURE.is_dir(), reason="golden librispeech capture not cached"
)


@pytest.fixture(scope="module")
def cfg() -> M.MelConfig:
    return M.load_mel_config()


# ---------- pure-DSP units (no golden needed) ----------


def test_filterbank_shape_and_slaney_norm(cfg):
    fb = M.mel_filterbank(cfg)
    assert fb.shape == (cfg.n_mels, cfg.n_fft // 2 + 1)
    assert fb.dtype == np.float32
    assert (fb >= 0).all()
    # Slaney enorm: each row peaks at 2/(upper-lower); rows must be nonzero.
    assert (fb.sum(axis=1) > 0).all()



def test_filterbank_first_bin_zero_and_top_filter_reaches_nyquist(cfg):
    fb = M.mel_filterbank(cfg)
    # Top filter: weight just below Nyquist (its upper edge is exactly
    # f_max = sr/2, where the triangle closes to zero).
    assert float(fb[cfg.n_mels - 1, cfg.n_fft // 2 - 1]) > 0.0

def test_preemphasis_first_taps(cfg):
    x = np.array([1.0, 0.5, 0.25], dtype=np.float32)
    y = M.preemphasize(x, cfg)
    assert y[0] == np.float32(1.0)
    assert y[1] == np.float32(np.float32(0.5) - np.float32(0.97) * np.float32(1.0))
    assert y.dtype == np.float32


def test_preemphasis_rejects_empty(cfg):
    with pytest.raises(ValueError):
        M.preemphasize(np.zeros(0, np.float32), cfg)


def test_frame_count_matches_reference_formula(cfg):
    # 166960-sample waveform (the pinned capture length) chunk-padded to
    # 480000 -> (480000 + 512 - 512)/160 + 1 = 3001 frames.
    wave = np.zeros(166960, np.float32)
    frames = M.stft_frames(M.chunk_for_mel(wave), cfg)
    assert frames.shape == (3001, cfg.n_fft)


def test_chunk_contract_pads_short_and_rejects_long(cfg):
    short = np.ones(160, np.float32)
    assert M.chunk_for_mel(short).size == M.CHUNK_SAMPLES
    assert float(M.chunk_for_mel(short)[-1]) == 0.0  # zero-padded tail
    with pytest.raises(ValueError):
        M.chunk_for_mel(np.zeros(M.CHUNK_SAMPLES + 1, np.float32))


def test_mask_all_valid_and_encoder_slice(cfg):
    wave = np.zeros(16000, np.float32)  # 1 s of silence
    feats, mask = M.extract_chunk_features(wave, cfg)
    assert feats.shape == (3001, cfg.n_mels) and feats.dtype == np.float32
    assert mask.shape == (3001,) and (mask == 1).all()  # reference: all valid
    es, em = M.encoder_slice(feats, mask)
    assert es.shape == (3000, cfg.n_mels) and em.shape == (3000,)
    assert np.array_equal(es, feats[:3000]) and np.array_equal(em, mask[:3000])


def test_silence_normalizes_to_zero(cfg):
    # Constant logmel rows -> std 0 -> (x-mean)/(0+eps) == 0 exactly.
    lm = np.full((5, 4), -3.5, np.float32)
    out = M.normalize(lm, cfg)
    assert np.array_equal(out, np.zeros((5, 4), np.float32))


def test_silence_chunk_logmel_is_log_guard(cfg):
    # A zero chunk has zero power everywhere, so mel dot == 0 and every
    # logmel value is exactly log(guard) -- precision-independent.
    wave = np.zeros(16000, np.float32)
    frames = M.stft_frames(M.chunk_for_mel(wave), cfg)
    lm = M.logmel(frames, cfg)
    guard32 = np.float32(cfg.log_guard)
    assert np.array_equal(lm, np.full_like(lm, np.float32(np.log(float(guard32)))))


# ---------- golden capture (skipped when not cached) ----------


@requires_capture
def test_golden_structural_contracts_hold_exactly():
    rep = M.compare_capture(CAPTURE)
    assert rep.shape_expected == rep.shape_actual == (3001, 128)
    assert rep.structural["mask_all_ones"]
    assert rep.structural["mask_equals_golden"]
    assert rep.structural["encoder_slice_shape"]
    assert rep.structural["encoder_slice_mask_equals_golden"]
    assert rep.structural["silence_tail_identical_rows_from_1046"]


@requires_capture
def test_golden_portable_divergence_is_the_documented_stop():
    """§62 receipt, kept as a tripwire.

    The portable pipeline must stay close to (but not bit-exact with) the
    Accelerate vDSP golden until someone reproduces Accelerate's float32
    DFT internals. If this test fails because a future pipeline IS exact,
    update it deliberately with new evidence -- do not delete it.
    """
    rep = M.compare_capture(CAPTURE)
    assert rep.exact is False
    assert rep.bit_exact_count / rep.total > 0.5
    assert rep.max_abs < 1e-4
    assert rep.mean_abs < 1e-6
    assert rep.first_divergence is not None
