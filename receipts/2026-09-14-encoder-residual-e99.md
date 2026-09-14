# 2026-09-14 Encoder residual at emission 99: no island, no single op

Status: named cause refined — the emission-99-moving encoder residual is
**not** produced by any ANE island, any single op family, or any single
layer. It is the macOS-ANE golden's own accumulated 24-layer fp16
execution delta from every correctly-rounded chain measured against it.
Token-exact E2E is still not claimed (99/105 stands).

Carries forward `receipts/2026-09-14-tdt-emission-99.md`, which fixed cause
(a): encoder residual (`b3d60c81` vs native `7e442034`, max_abs 0.14716,
rel_l2 0.024917) already inside the LSTM one emission before the argmax
flip. This slice asked **which island/layer contributes that residual**.

## Verdict

1. **Islands are exonerated, twice.** The pure-GPU arm (`02092bac`) sits at
   rel_l2 0.024596 from native and the ANE-island arm (`b3d60c81`) at
   0.024917: the islands move the output by 0.0027 mutual, +0.000320
   against native (parity receipt) — about 1.3 % of the residual. Removing
   the ANE entirely keeps 0.0246 of the 0.0249 gap. And Apple's own GPU
   control is 0.023345 from the same golden: the gap follows the macOS ANE
   golden, not our islands and not Linux.
2. **No op family reproduces or owns the gap.** A per-op-family precision
   probe (derived executor, GPU-only arms) moved rel_l2 by at most
   ±0.0019 per family — an order of magnitude under the 0.0246 gap:
   fp32 chain −0.00061, softmax −0.00026, silu/sigmoid +0.00008,
   linear chunked-fp16 −0.00113, matmul chunked-fp16 +0.00192,
   LN rsqrt-fp16 +0.00076, LN affine-fp16 +0.00035. Direction and size
   say: correctly-rounded implementations of every family cluster at
   0.023–0.027 while the golden sits 0.0246 from the middle of that
   cluster. `matmul16` moving away matches the island evidence: ANE
   matmul accumulation is fp32-like, not the source.
3. **Layer norm is excluded by fragility, not preference.** Rounding LN's
   internal staging (mean/centered/var/normalized) to fp16 — with exact
   fp32 reductions — explodes the encoder to rel_l2 0.986 (sequential
   fp16 reductions: 1.003). A network this sensitive to LN internals
   cannot be carrying a smooth 2.4 % LN-born deviation; the golden's LN
   must be as exact as ours. Small multiplicative rsqrt error (rsqrt16)
   and affine rounding (gamma16) are benign (≤0.0008), confirming the
   fragility is specific to reduction staging.
4. **The residual's structure is distributed.** Per-channel gain fit:
   native ≈ 0.9934·linux ± 0.0070; after the fit rel_l2 only drops to
   0.0220. Error is uniform across all ten 64-channel blocks (2.21–2.79 %
   each), full-rank (rank-1 energy 30 %), with no frame-shift structure.
   That is accumulated execution delta over 24 layers, not one op.

## Why no fix landed

The assignment's fix path is a compile/layout fix that does not invent
ISA. None exists here: the residual is a property of the **golden
itself** — Apple's ANE executing this MIL with its own internal dataflow.
Our correctly-rounded executor already reproduces the all-GPU arm
byte-identically (`out-cr` = `02092bac`), and the gap to the golden is
invariant across everything we control (islands on/off, per-op fp16/fp32
chains). Closing it would mean replicating Apple's ANE op fission and
accumulation order — the excluded path. Named op: **none; stop.**

## Hardware / protocol

- host `jwm1-linux`, kernel `7.1.6-1-1-ARCH`, aarch64; jw16 untouched
- all runs `flock -w 900 /tmp/m1-gpu.lock`, GPU-only (`--no-ane`);
  zero ANE submits this slice; lock never stolen, never unlinked
- mlx `0.32.2.dev202609121216+8f6de34c`, `Device(gpu, 0)`

## Pins

| item | sha256 (16) |
| --- | --- |
| probe `encoder_precision_probe.py` (final) | `bbce43c3669cae33` |
| encoder `model.mil` | `ac8e9526154ac8b8` |
| native golden `encoder_hidden.npy` | `7e442034cb42521f` |
| Linux ANE arm `encoder_hidden.npy` | `b3d60c81bcd9c63f` |
| Linux all-GPU arm / CR baseline | `02092bac3054e1dd` |
| mel equality | Linux e2e mel == capture mel, bit-exact |

Inputs: `EncoderParityAne/capture` features/mask (375 valid frames, mask
census all-false — masking inert). Work: `/var/tmp/EncoderResidualE99/`
on jwm1, arms `out-*`, derivation in
`receipts/2026-09-14-encoder-residual-e99/derivation/` (probe, arm
scripts, `arms.json`).

Arms ran under three byte-states of the probe (out-dir placement, the LN
fp16-branch rewrite, the rsqrt/gamma mode addition). Non-LN paths are
pinned across those states by the byte-identical `softmax16` /
`softmax16v2` pair (`95729d8857e29f97`) and by the CR gate above.

## Arm table (rel_l2 vs native `7e442034`)

| arm | precision change | rel_l2 | Δ vs CR |
| --- | --- | ---: | ---: |
| CR baseline (`cr`) | none — byte-identical `02092bac` | 0.024596 | — |
| fp32chain | all intermediates fp32 | 0.023987 | −0.00061 |
| softmax16 | softmax fp16-internal | 0.024341 | −0.00026 |
| silu16 | silu+sigmoid fp16 | 0.024676 | +0.00008 |
| linear16 | linear K-chunk fp16-acc | 0.023468 | −0.00113 |
| matmul16 | matmul K-chunk fp16-acc | 0.026516 | +0.00192 |
| ln16seq | LN sequential fp16 reductions | 1.003479 | explodes |
| ln16v2 | LN fp16 staging, exact reductions | 0.985703 | explodes |
| lnrqrt16 | only LN rsqrt rounded fp16 | 0.025356 | +0.00076 |
| lngamma16 | only LN affine rounded fp16 | 0.024945 | +0.00035 |
| full16v2 | LN+softmax+silu+sigmoid fp16 | 0.986175 | explodes |
| islands A/C (parity receipt) | real ANE attention matmuls | 0.024916 | +0.00032 |
| macOS GPU (parity receipt) | Apple's own GPU chain | 0.023345 | −0.00125 |

`linear16` is the only toward-move above 0.001; on one golden clip that is
weak directional evidence that ANE linear accumulation is slightly less
exact than our CR matmul, and it is not load-bearing for the verdict.

## Not claimed

Token-exact or transcript-exact E2E (the 99/105 emission-99 divergence
stands), bit-exact encoder_hidden, a closed LSTM operator
(`parakeet.tdt.decoder-lstm.fp16-recurrent-vs-native-coreml` remains
open), or an ANE-internal-behavior emulator. `conv16` was not taken
(group-aware chunk slicing unimplemented); conv is excluded through the
island measurement instead. Native tensors at emissions 98/99 still do
not exist; nothing here reopens cause (a)'s decoder-side mechanism.
