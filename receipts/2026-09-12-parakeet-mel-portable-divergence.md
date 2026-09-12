# Receipt: Portable reference mel frontend — §62 stop with first-divergence evidence, 2026-09-12

Branch `parakeet-mel-exact` (isolated worktree from `main-land2` @
`d8eb0037`). Scope: reference-derived host mel preprocessing under
`overlay/tools/coreml/`, focused tests, `docs/parakeet.md`, this receipt.
Inspector, `reference.lock`, and the root session's uncommitted
inspect/test work were not touched.

## Assignment contract

* Target: exact reference mel values; exact public-reference mel
  frontend from 16 kHz mono float32 waveform; host DSP labelled per §43.
* Fallback (assignment text): "Build full working host-only frontend
  only if it meets fixed contract; otherwise receipt-only source
  diagnostic on separate branch, main receives no unqualified feature."
* Golden equality required: exact. No tolerance loosening, no golden
  outputs stored as implementation, no hardcoded sample values.

## Pinned inputs (verified by executed commands)

* Reference source read: `mweinbach/parakeet-coreml-swift` @
  `75aec2a1c991319657ff4dec5f602c12da6c5012`, fetched as tarball,
  `Sources/ParakeetTDT/MelFeatureExtractor.swift`, `MelFilterBank.swift`,
  plus chunking/encoder-slice paths (`ParakeetTranscriber.transcribe`,
  `ModelRunner.runEncoder`, capture `main.swift`). License: Apache-2.0
  (`LICENSE` read). Python port carries attribution in its header.
* Golden capture:
  `~/.cache/mlx-omarchy/parakeet-reference/captures/b650695c-75aec2a/20260912T154759Z-librispeech/ane`.
  `sha256sum` of waveform.npy, mel.npy, mel_mask.npy,
  encoder_input_features.npy, encoder_input_mask.npy all match
  `parakeet-reference.lock` `macos_reference_paths` (no mismatches).

## What the pinned source actually does (verified against golden)

* Chunking: 30 s chunks (3000 × 160 samples), tail chunk zero-padded;
  mel runs per padded chunk → 3001 frames, mask all ones.
* Preemphasis `y[0]=x[0]`, `y[n]=x[n]-0.97·x[n-1]`, plain float32
  mul-sub (no FMA contraction — see variants below).
* Framing: `n_fft/2` zero-pad both sides; per frame t the Swift source
  windows `padded[t·160 .. t·160+399]` and places the result at FFT
  bins `[56 .. 455]`. This is **56 samples earlier than
  `torch.stft(center=True)`** would window the same frame. The golden
  tail proves it: with true center framing, frames 1045+ of this
  capture would be pure silence and bit-identical; the golden has its
  last non-silence frame at 1045 (17 real samples in-window) and
  identical rows only from 1046. The capture recorded the Swift offset.
* FFT: `vDSP_DFT_zrop` float32, output scaled ×2 relative to the
  mathematical DFT (undone by ×0.5); power via sqrt-then-square.
* Mel: Slaney filterbank (float64 math → float32 rows), per-row float32
  dot; `log(mel + 2^-24)`; per-bin mean/std over all 3001 frames,
  sequential float32 accumulation, Bessel (n−1, min 1);
  `(x−mean)/(std+1e-5)`.
* Encoder input: first 3000 frames of each chunk (the 3001st mel frame
  is computed but never fed).

## Numerical result: portable pipeline cannot match the vDSP golden exactly

Pipeline reimplemented in NumPy (`overlay/tools/coreml/mel_reference.py`),
13 precision variants tested on the pinned capture
(`bit_exact = identical float32 values / 384128`):

| variant (fft × mel-dot × log) | bit_exact | max abs diff |
|---|---|---|
| f64 × f64 × f64 | 215401 (56.08%) | 4.770e-05 |
| f64 × f64 × f32 | 215408 (56.07%) | 4.770e-05 |
| f64 × f32 × f64 | 210188 (54.72%) | 4.770e-05 |
| f64 × f32 × f32 | 212698 (55.37%) | 4.770e-05 |
| f32-rounded spectra × f64 × f64 | 215401 (56.08%) | 4.770e-05 |
| scipy float32 pocketfft × f64 × f64 | 207262 (53.96%) | 6.227e-05 |
| FMA-emulated preemphasis (f64 path) | 206076 (53.65%) | 4.910e-05 |
| … remaining combinations | ≤ 215408 | ≤ 6.227e-05 |

Diagnostic output (executed; exit 1 by design on non-exact):

```text
shape: expected (3001, 128), computed (3001, 128)
exact golden equality: False
bit-identical values: 215401/384128 (56.08%)
max |diff|: 4.769862e-05   mean |diff|: 1.532242e-07
first divergence: frame 0 bin 0: golden -0.1173354983329773 vs
  computed -0.1173354908823967 (-1 ulps)
structural shape_matches / mask_all_ones / mask_equals_golden /
  encoder_slice_shape / encoder_slice_mask_equals_golden /
  silence_tail_identical_rows_from_1046: True
structural encoder_slice_features_exact: False (inherits value rounding)
VERDICT: portable pipeline is not bit-exact with the Accelerate vDSP
golden (§62 stop). Diagnostic only; no frontend claim.
```

Attribution of the residual: the divergence persists identically across
float32/float64 dot and log variants and does not shrink when the
spectra are float32-rounded, so it originates in the float32 DFT itself
(Accelerate's internal butterfly/twiddle summation order), with
possible ≤1 ulp contributions from arm64 `cosf`/`logf` in the pinned
binary. These internals are proprietary and not portable-reproducible;
13 sane portable variants bracket the space without reaching exactness.

Structural contracts verified exact (precision-independent): chunk
padding to 480000 samples; 3001 frames; all-ones mel mask equal to
golden; encoder slice = first 3000 frames; silence-tail row identity
from frame 1046; `mel[:3000]` equals the golden encoder input
up to the value rounding above.

## Delivered

* `overlay/tools/coreml/mel_reference.py` — reference-derived host mel
  pipeline + `compare_capture` first-divergence diagnostic; labelled
  host DSP per §43; config taken from `parakeet-reference.lock`
  (`MelConfig`), nothing hardcoded.
* `tests/coreml/test_mel_reference.py` — 11 focused tests: 9
  synthetic (filterbank, Hann, preemphasis, chunk/frame contract, mask
  + encoder slice, silence → log(guard), Bessel normalization), 2
  golden-gated (structural contracts exact; documented §62 divergence
  as a deliberate tripwire).
* `docs/parakeet.md` — "Portable mel frontend status (§62 stop)"
  section; conventions bullet corrected for the 56-bin Swift offset.
* Main branch receives no frontend claim; this branch is the
  receipt-only diagnostic path per the assignment fallback.

## Commands (exact)

```bash
git worktree add <tree> d8eb0037 -b parakeet-mel-exact
curl -sL -o ref.tar.gz https://github.com/mweinbach/parakeet-coreml-swift/archive/75aec2a1c991319657ff4dec5f602c12da6c5012.tar.gz
sha256sum <capture>/{waveform,mel,mel_mask,encoder_input_features,encoder_input_mask}.npy
python3 overlay/tools/coreml/mel_reference.py \
  ~/.cache/mlx-omarchy/parakeet-reference/captures/b650695c-75aec2a/20260912T154759Z-librispeech/ane
python3 -m pytest tests/coreml/ -q   # 39 passed
```

## Verdict

§62 stop honored: numerical mismatch explained and evidenced; no
tolerance loosened; no approximate frontend claimed. Exact mel parity
on Linux requires either Apple Accelerate or a bit-exact reproduction
of `vDSP_DFT_zrop` internals — both out of scope for a portable host
frontend.
