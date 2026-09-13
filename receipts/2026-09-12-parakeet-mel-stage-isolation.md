# Receipt: mel stage isolation with certified macStudio intermediates, 2026-09-12

Branch `parakeet-mel-exact`. Continues the portable mel diagnostic
(`receipts/2026-09-12-parakeet-mel-portable-divergence.md`, which records
the open exact-parity gap without stage attribution). This work isolates
the first divergent preprocessing stage using ground-truth intermediates,
per plan §43 (host DSP kept separate from tensor inference) and §62
(diagnosis only; release/execution gates unchanged).

## Method

1. New CPU-only Swift target `mel-stage-capture`
   (`overlay/tools/coreml/capture/`, same pinned SwiftPM dependency
   `parakeet-coreml-swift @ 75aec2a1c991319657ff4dec5f602c12da6c5012`).
   No CoreML, no model loads, no serving changes.
2. It runs the pinned `MelFeatureExtractor.extract` AND a stepwise
   re-expression of the same operations, then gates: the stepwise final
   array must be byte-identical to the pinned library output. Observed:
   `gate: stepwise == pinned extract() byte-identical (384128 values)`.
3. Input is the pinned capture waveform as raw f32
   (sha256 `29ff16924c9577f5e335fcc7d64dd5d9ee261db7e9ec9c8eb7f8129b31cc1643`,
   verified by the tool against `--expect-sha256` before computing).
4. macstudio (Apple M1 Ultra, macOS 26.6.2 25G83, Swift 6.3.3 — the
   golden environment): `mel_pinned.npy` sha256 =
   `4ed24d7da64c17f58419ec45f15aa24ac9d1cfd93a4de51ee4f05e1d712b4d22`,
   byte-identical to the golden `mel.npy` pinned in
   `parakeet-reference.lock`. Every dumped intermediate is therefore
   certified ground truth for what the reference computed.

## Stage truth table (portable NumPy vs certified intermediates)

Each stage is compared with certified upstream inputs fed forward, so
exactly one stage differs at a time
(`overlay/tools/coreml/mel_stage_compare.py <capture-dir> <dump-dir>`):

```text
gate: dump manifest verified; mel_pinned == golden mel.npy
preemph   : EXACT (480000 values)
hann      : DIVERGE n_diff=243/400  max|d|=4.768e-07
mel_fb    : EXACT (32896 values)
frames    : EXACT (1536512 values)        (framing geometry exact given window)
dft_real  : DIVERGE n_diff=225133/771257  max|d|=4.768e-07
dft_imag  : DIVERGE n_diff=222670/771257  max|d|=4.768e-07
power     : EXACT (771257 values)         (given certified spectrum)
melproj   : DIVERGE n_diff=29426/384128   max|d|=5.960e-08
logmel    : DIVERGE n_diff=3198/384128    max|d|=1.907e-06
mean      : EXACT (128 values)            (given certified logmel)
std       : EXACT (128 values)
```

## Per-stage characterization

* **hann window** — Swift computes `0.5 - 0.5*cos(2.0*.pi*Float(n)/N)` in
  float32; the argument is bit-identical to the portable one, but Apple's
  `cosf` result differs from NumPy float32 `cos` on 243/400 entries and
  from float64 correctly-rounded cos (same argument, cast) on 304/400.
  Apple `cosf` is not correctly rounded, so the window is not
  formula-reproducible portably. It is a fixed 400-value algorithm
  constant; pinning it as reference-derived data would be an owner
  decision, not made here.
* **DFT** — `vDSP_DFT_zrop` float32 output vs pocketfft float64 cast:
  29% of bins differ (pocketfft float32 native: similar), max |d|
  4.768e-07, typical 1–2 ulp. This is now stage-level evidence; the
  final-output-only inference ("proprietary internals") is replaced by
  measured per-bin divergence against certified truth.
* **mel dot** — `vDSP_dotpr` vs tested accumulation orders (sequential,
  pairwise, k-lane k∈{2,4,8,16,32} × horizontal orders, FMA-emulated
  lanes, wide accumulators): best is 4-lane pairwise with rounded
  products + scalar tail at 961/384128 residual, all ±1 ulp. Not
  exactly emulated.
* **log** — Apple `logf` vs float64-log cast: 339/384128 values (0.09%),
  ≤1 ulp; NumPy float32-native log: 4903 values.
* **exact stages** — preemphasis (plain float32 mul-sub, no FMA
  contraction), framing geometry (window samples at `padded[t*hop..]`
  placed at bins `[56..455]`), Slaney filterbank (double math → float32
  cast is portable), sqrt-then-square power, sequential float32
  mean/std, mask/shape/encoder-slice contracts.

## Precise missing prerequisites for an exact portable frontend

1. Bit-exact Apple `cosf` and `logf` (window + log stages), or owner
   approval to pin the 400-value window constant as reference-derived
   data.
2. A bit-exact reimplementation of `vDSP_DFT_zrop`'s butterfly network
   for N=512 (unpublished Accelerate internals), or executing that
   single stage on macOS (which contradicts a host-only portable
   frontend).
3. The exact `vDSP_dotpr` lane/FMA layout (best tested emulation is
   99.75% exact).

No tolerance was loosened, no golden output substituted, no approximate
frontend shipped; the diagnostic and stage tooling stay on this branch.

## Commands (exact)

```bash
# input prep (Linux)
python3 - <<'PY'
import numpy as np, hashlib
w = np.load('<capture>/waveform.npy'); raw = w.astype('<f4').tobytes()
open('/tmp/wave.f32','wb').write(raw)
print(hashlib.sha256(raw).hexdigest())
PY
rsync -a overlay/tools/coreml/capture/ macstudio:~/parakeet-mel-stage/
ssh macstudio 'cd ~/parakeet-mel-stage && swift build -c release'
ssh macstudio 'cd ~/parakeet-mel-stage && .build/release/mel-stage-capture \
  --waveform wave.f32 --out stage-capture \
  --expect-sha256 29ff16924c9577f5e335fcc7d64dd5d9ee261db7e9ec9c8eb7f8129b31cc1643'
# -> verified waveform sha256 ok / pinned extract: 3001 frames /
#    gate: stepwise == pinned byte-identical (384128 values)
rsync -a macstudio:~/parakeet-mel-stage/stage-capture/ /tmp/stage-dumps/
python3 overlay/tools/coreml/mel_stage_compare.py <capture-dir> /tmp/stage-dumps
```

## Addendum: libm probes + DFT network probing (2026-09-12, same session)

Published-source check: `apple-oss-distributions/libm` (main) publishes
`Source/ARM/cosf.s` and `Source/ARM/logf.s` as **0-byte placeholders** —
the arm64 `cosf`/`logf` algorithms are not open-sourced, and no
license-usable published implementation of the shipping routines exists.

`mel-stage-capture` gained `--mode probe-cosf|probe-logf|probe-dft`
(CPU only). Probe data preserved outside git at
`~/.cache/mlx-omarchy/parakeet-reference/captures/b650695c-75aec2a/mel-stage-probes/`
(inputs, outputs, analysis scripts, certified stage dumps).

Findings (arithmetic, not per-input tables):

```text
logf  : float32(log(float64(x))) after a float32 add of the guard —
        EXACT on all 384128 capture values (the earlier 339-value log
        divergence was solely the float32 guard add). Wide-range probe
        (2.38M values incl. 2M synthetic): 15 outliers, outside the
        capture domain.
cosf  : Apple arm64 cosf deviates from correctly-rounded
        float32(cos(float64(x))) on 1.31% of 4.0M probed float32 in
        [0, 2*pi], always +/-1 ulp; on the 400 exact window arguments,
        5/400 deviate. Algorithm-specific, not derivable from published
        source.
DFT   : vDSP_DFT_zrop output == mathematical DFT exactly (the vDSP x2
        scale and the code's x0.5 cancel). Impulses at n=0/256 (trivial
        internal path) reproduce float32-rounded exact roots everywhere;
        general frames deviate 1-3 ulp — a float32 butterfly network
        with accurately-rounded twiddles. Tested candidates: radix-2
        DIT/DIF (bit-reversed, round32-exact per-stage twiddles),
        two-rounding and single-rounding untangle, FMA/f64-stage
        kernels. Best: radix-2 DIT + two-rounding untangle = 2799/6682
        probe values mismatched (ulp-level); radix-2 DIF worse (3095).
        The 256-point complex core's radix geometry/order is still
        unidentified.
```

Next concrete experiment (handoff): identify the vDSP 256-point complex
core — run single-impulse probes at every complex index c = 0..255
(frames of 512 with 1.0 at sample 2c and 2c+1), recover each stage's
effective twiddle multiply from the ulp signature, and test radix-4 /
split-radix geometries against the preserved probe set; then re-run
`mel_stage_compare.py`. Window (`cosf`) remains an owner decision;
logf and the untangle are solved arithmetic; no lock, threshold, or
tolerance was changed.

Probe commands (macstudio, `~/parakeet-mel-stage`):

```bash
.build/release/mel-stage-capture --mode probe-cosf --waveform cos_in.f32 --out probe-cosf
.build/release/mel-stage-capture --mode probe-logf --waveform log_in.f32 --out probe-logf
.build/release/mel-stage-capture --mode probe-dft  --waveform dft_in.f32 --out probe-dft
```
