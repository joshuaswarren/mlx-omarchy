# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Per-stage comparison of the portable mel pipeline against certified
macStudio intermediates (``mel-stage-capture`` output).

The Swift tool certifies its dumps with a byte-equality gate against the
pinned ``MelFeatureExtractor`` and reproduces the golden ``mel.npy``
hash, so each dumped stage is ground truth for what the reference
computed. This script re-derives each stage with the portable pipeline,
feeding certified upstream inputs (so exactly one stage differs at a
time), and prints the first divergent preprocessing stage.

Usage:
    python3 mel_stage_compare.py <capture-dir> <stage-dump-dir>

Exits 0 only if every stage is byte-identical.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from reference import ReferenceLock

TOOLS = Path(__file__).resolve().parent


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stage_diff(name: str, reference: np.ndarray, truth: np.ndarray) -> tuple[bool, str]:
    exact = reference.tobytes() == truth.tobytes()
    if exact:
        return True, f"{name:10s}: EXACT ({truth.size} values)"
    n_diff = int((reference != truth).sum())
    diff = np.abs(reference.astype(np.float64) - truth.astype(np.float64))
    detail = f"{name:10s}: DIVERGE n_diff={n_diff}/{truth.size} max|d|={diff.max():.3e}"
    bad = np.argwhere(reference != truth)
    first = tuple(int(v) for v in bad[0])
    detail += f" first={first} truth={truth[first]!r} portable={reference[first]!r}"
    return False, detail


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__.strip())
        return 2
    capture_dir, dumps_dir = Path(argv[1]), Path(argv[2])
    lock = ReferenceLock.load(TOOLS / "parakeet-reference.lock")
    cfg = lock.mel

    # Gate: the dumps must be certified (manifest) and reproduce golden.
    manifest = json.loads((dumps_dir / "manifest.json").read_text())
    for name, want in manifest.items():
        got = sha256_file(dumps_dir / f"{name}.npy")
        if got != want:
            print(f"dump manifest mismatch for {name}")
            return 3
    golden = np.load(capture_dir / "mel.npy")
    if np.load(dumps_dir / "mel_pinned.npy").tobytes() != golden.tobytes():
        print("mel_pinned.npy does not reproduce golden mel.npy byte-for-byte")
        return 3
    print("gate: dump manifest verified; mel_pinned == golden mel.npy")

    wave = np.load(capture_dir / "waveform.npy")
    chunk = M_chunk(wave, cfg)

    mac = {n: np.load(dumps_dir / f"{n}.npy") for n in
           ("preemph", "hann", "mel_fb", "frames", "dft_real", "dft_imag",
            "power", "melproj", "logmel", "mean", "std")}

    ok = True

    # Stage 1: preemphasis (downstream stages use certified inputs).
    ref_pre = preemphasize(chunk, cfg)
    good, line = stage_diff("preemph", ref_pre, mac["preemph"])
    print(line); ok &= good

    # Stage 2: hann window (formula-only candidate).
    ref_hann = hann_window(cfg.win_length)
    good, line = stage_diff("hann", ref_hann, mac["hann"])
    print(line); ok &= good

    # Stage 3: Slaney filterbank.
    ref_fb = mel_filterbank(cfg)
    good, line = stage_diff("mel_fb", ref_fb, mac["mel_fb"])
    print(line); ok &= good

    # Stage 4: framing (certified window fed forward).
    padded = np.zeros(ref_pre.size + cfg.n_fft, np.float32)
    padded[cfg.n_fft // 2:cfg.n_fft // 2 + ref_pre.size] = ref_pre
    idx = np.arange(cfg.win_length)[None, :] + cfg.hop_length * np.arange(mac["frames"].shape[0])[:, None]
    ref_frames = np.zeros_like(mac["frames"])
    off = (cfg.n_fft - cfg.win_length) // 2
    ref_frames[:, off:off + cfg.win_length] = (padded[idx] * mac["hann"][None, :]).astype(np.float32)
    good, line = stage_diff("frames", ref_frames, mac["frames"])
    print(line); ok &= good

    # Stage 5: DFT on certified frames.
    spec = np.fft.rfft(mac["frames"].astype(np.float64), axis=1)
    good, line = stage_diff("dft_real", spec.real.astype(np.float32), mac["dft_real"])
    print(line); ok &= good
    good, line = stage_diff("dft_imag", spec.imag.astype(np.float32), mac["dft_imag"])
    print(line); ok &= good

    # Stage 6: power on certified spectrum.
    re32, im32 = mac["dft_real"], mac["dft_imag"]
    mag = np.sqrt(re32 * re32 + im32 * im32).astype(np.float32)
    good, line = stage_diff("power", (mag * mag).astype(np.float32), mac["power"])
    print(line); ok &= good

    # Stage 7: mel projection on certified power (float64 matmul, cast).
    ref_mp = (mac["mel_fb"].astype(np.float64) @ mac["power"].T.astype(np.float64)).T.astype(np.float32)
    good, line = stage_diff("melproj", ref_mp, mac["melproj"])
    print(line); ok &= good

    # Stage 8: log on certified projection.
    guarded = (mac["melproj"] + np.float32(cfg.log_guard)).astype(np.float32)
    ref_lg = np.log(guarded.astype(np.float64)).astype(np.float32)
    good, line = stage_diff("logmel", ref_lg, mac["logmel"])
    print(line); ok &= good

    # Stage 9: normalization constants on certified logmel.
    lm = mac["logmel"]
    n_t = np.float32(lm.shape[0])
    mean = np.cumsum(lm, axis=0, dtype=np.float32)[-1] / n_t
    dev = (lm - mean[None, :]).astype(np.float32)
    var = np.cumsum((dev * dev).astype(np.float32), axis=0, dtype=np.float32)[-1] / np.float32(max(lm.shape[0] - 1, 1))
    std = np.sqrt(var).astype(np.float32)
    good, line = stage_diff("mean", mean, mac["mean"])
    print(line); ok &= good
    good, line = stage_diff("std", std, mac["std"])
    print(line); ok &= good

    print("VERDICT:", "all stages byte-identical" if ok else
          "divergent stages listed above (first divergent preprocessing stage "
          "is the earliest DIVERGE line whose upstream inputs were certified)")
    return 0 if ok else 1


def M_chunk(wave, cfg):
    import numpy as np
    x = np.ascontiguousarray(wave, dtype=np.float32)
    target = 3000 * cfg.hop_length
    if x.size < target:
        x = np.concatenate([x, np.zeros(target - x.size, np.float32)])
    return x


def preemphasize(x, cfg):
    y = np.empty_like(x)
    y[0] = x[0]
    y[1:] = (x[1:] - np.float32(cfg.preemphasis) * x[:-1]).astype(np.float32)
    return y


def hann_window(win_length):
    n = np.arange(win_length, dtype=np.float32)
    return (np.float32(0.5) - np.float32(0.5)
            * np.cos(np.float32(2.0) * np.float32(np.pi) * n
                     / np.float32(win_length - 1))).astype(np.float32)


def mel_filterbank(cfg):
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    with np.errstate(divide="ignore"):
        def hz_to_mel(hz):
            return np.where(hz >= min_log_hz,
                            min_log_mel + np.log(hz / min_log_hz) / logstep, hz / f_sp)

    def mel_to_hz(m):
        return np.where(m >= min_log_mel,
                        min_log_hz * np.exp(logstep * (m - min_log_mel)), f_sp * m)

    with np.errstate(divide="ignore"):
        mels = hz_to_mel(0.0) + (hz_to_mel(cfg.sample_rate / 2.0) - hz_to_mel(0.0)) * (
            np.arange(cfg.n_mels + 2) / (cfg.n_mels + 1))
    hz_pts = mel_to_hz(mels)
    freqs = np.arange(cfg.n_fft // 2 + 1) * cfg.sample_rate / cfg.n_fft
    lo, c, up = hz_pts[:-2], hz_pts[1:-1], hz_pts[2:]
    tri = np.maximum(0.0, np.minimum((freqs[None, :] - lo[:, None]) / (c - lo)[:, None],
                                     (up[:, None] - freqs[None, :]) / (up - c)[:, None]))
    return (tri * (2.0 / (up - lo))[:, None]).astype(np.float32)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
