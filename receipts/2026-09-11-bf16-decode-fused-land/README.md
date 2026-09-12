# BF16 fused decode attention — landing receipt (2026-09-11)

Status: LANDED on `adjudicate/bf16-fused-land` (rebased onto main 5aab281f).
The composition-exact fusion collapses the BF16 single-query decode
attention's ten-dispatch f32 composition into one dispatch and moves no
generated token on any driver: the kernel reproduces the composition's
observable arithmetic op-for-op, so every digest is held by construction and
proven on hardware. No policy argument is required - parity rule 1 and
rule 2 are satisfied vacuously because no digest moves.

## Decision history

Bf16FusedDecode's original online-softmax arm (72ca97e3) moved the fork
BF16 short and long digests and was held back
(`receipts/2026-09-11-bf16-decode-fused`). Adjudication against the float64
oracle (`long.json`, `short.json`, `synth.json` in this directory; the
receipts' ULP convention, `abs(int16(got) - int16(want))`):

| leg | n | fused vs O-f64 (exact/mean/max) | comp vs O-f64 | fused farther / closer |
|---|---|---|---|---|
| BF16 short 30/32 | 709,632 | 0.999642 / 0.0008 / 261 | 0.999517 / 0.0013 / 453 | 152 / (see short.json) |
| BF16 long 262/128 | 2,774,016 | 0.999622 / 0.0124 / 32,749 | 0.999463 / 0.0125 / 32,481 | 592 / 1,066 |
| synthetic sweep | 16,128 | 0.999752 / 0.0002 / 1 | 0.999566 / 0.0004 / 1 | 1 / 4 |

The owner's bar for the amendment path was per-element "never farther from
float64 than the composition"; 152 (short) and 592 (long) captured elements
sit farther, so the amendment could not waive rule 1 for the third-value
long-leg digest. The fork short leg's move landed exactly on native's
`7fc0f968789b1882` (rule 1 admits; confirmed against
`receipts/native-baseline-2026-09-06` and fresh same-die captures), but the
long leg's `06b014aebf5d4b66` had no path. Decision: do not land the
online-softmax arm; rebuild the kernel to need no decision at all.

Correction to the earlier receipt's summary table: the raw bench JSONs in
`/tmp/bf16fused-run` (and the alpha-fix stock cells) show the stock driver's
BF16 short digest was ALREADY native's `7fc0f968789b1882` and stock long was
`46108ad71157cb4d` before any of this work; the README table there copied
stale values (`f2617...`/`c5be...`) into the stock rows.

## The composition-exact arm

`sdpa_decode_native.comp -DBF16_IO=1` now emits an arm that reproduces the
f32-score composition's observable arithmetic in order:

1. one f32 `q * scale` (the composition's elementwise multiply; the scores
   matmul alpha is 1.0),
2. scores: ascending-d `acc += a * b` over 16-wide tiles with exact-zero
   tail padding, drained by `1.0f`, stored f32 (matmul.comp),
3. softmax: 256-lane strided max scan + binary max tree, strided sequential
   exp-sum per lane + binary sum tree, `normalizer = 1.0/sum`, probs =
   `exp(score - max) * normalizer` recomputed from the stored f32 scores
   (softmax_suffix.comp),
4. output: ascending-key `acc += a * b` over 16-wide tiles with exact-zero
   tail padding, drained by `1.0f`, stored through the cast.comp RNE bf16
   pair.

Guards: uncovered shapes refuse exactly as before and fall through to the
composition by name; contexts past 2048 keys (the arm's shared-memory
stream) fall through too. The arm uses no subgroup operations, so its device
gate checks only the 1024-thread workgroup and the 9,472 bytes of static
shared it declares - the f16 route's gate is untouched, and software drivers
can exercise the bf16 route for real.

## By-construction proofs

- f16 arm SPIR-V byte-identical to main 5aab281f's blob:
  `a301a801f73640c3c2fe11888bb85526618f1cc4ac839f4557d4143b0e3a6e60`
  (glslc -O --target-env=vulkan1.3, both trees).
- Dev box (llvmpipe, MLX_OMARCHY_ALLOW_NON_APPLE=1, bf16 route ENGAGED -
  dispatches == 1 at every covered key count):
  `omarchy_sdpa_decode_fused_tests` bit-identity gate, 10,773 assertions,
  0 failed, over k = 1, 5, 16, 17, 64, 263, 320 plus the 2100-key
  refusal fall-through. `omarchy_fast_ops_tests` 35/35 cases.

## Hardware gates and measurement (window C, M1 jwm1-linux, Apple M1 G13G B1,
driver mesa-honeykrisp-omarchy 26.3.0.devel.hk6f6afc8-1, one flock window;
wheel 0.32.2.dev202609120006+148356d7 sha256 a327f759e45257659f8a7fc6fcecaf8
c012b0f4d352b0a4d3c44e78ac9acff62; stock runs via the stock Mesa ICD
override; warmup discarded, 3 reps per leg per driver, every rep's digest
asserted; stock warmup run separately, rc=0)

Bit-identity on real in-stream inputs: every one of the 3,888 captured
decode-attention calls (2,774,016 long-leg + 709,632 short-leg output
elements) stored the same words from the fused route and from the
composition route on the fork driver, with the replayed composition also
bit-identical to the in-stream original (controls clean).

Digests (every rep identical per cell; Q4 short/long/1K and all BF16 legs):

| leg | driver | base (pre-fusion) | this wheel | native |
|---|---|---|---|---|
| Q4 short/long/1K | fork + stock | 7fd25a869ff21678 / 4cc08910089477fd / 7da83f06ec9f001d | same | short+1K = native |
| BF16 short | fork | f26175202f3dabe9 | **f26175202f3dabe9** | 7fc0f968789b1882 |
| BF16 short | stock | 7fc0f968789b1882 | **7fc0f968789b1882** | = native, held |
| BF16 long | fork | 8690dc83246b39f8 | **8690dc83246b39f8** | 407b7624ed1b3b29 |
| BF16 long | stock | 46108ad71157cb4d | **46108ad71157cb4d** | 407b7624ed1b3b29 |
| BF16 1K | fork + stock | ff502900d2a179a5 | **ff502900d2a179a5** | = native, held |

Zero digest movement on either driver. Because both routes are bit-identical
per call, the 256-key perf gate added on top is digest-invariant; window C's
short-leg cells exercised the composition side and its long/1K cells the
fused side, exactly the post-gate routing.

ms/token (median of 3, fork, bf16): short 34.1 (base 31.2 - the gate routes
the short leg to the composition, so shipped short-leg timing is the base
31.2), long 34.4 (base 34.7, +0.9%), 1K 39.9 (base 43.3, +8.5%). Fractions
of the committed native baseline with the gate: short 0.566 (unchanged),
long 0.522, 1K 0.475.

## Follow-up (stated, not started)

1. The online-softmax variant of the arm measured 29.0/29.6/29.8 ms/token
   (fractions 0.613/0.607/0.618) but moved digests; its adoption is blocked
   on reproducing the composition's matmul accumulation order in-kernel
   faster than the serial order used here, or on an owner re-pin.
2. Window C's digest matrix was measured on main 5aab281f + this stack;
   the branch has since been rebased onto f425f46f with the dev-box
   bit-identity gate re-proven (10,773 assertions). A short confirmatory
   digest window on the f425f46f-based wheel is the only open check.

## Provenance

- Host: jwm1-linux (Apple M1 G13G B1), driver verified in-window.
- In-stream captures: 3,096 calls (long leg) + 792 calls (short leg) taken
  under the base wheel with the canonical generated-ID digests reproduced
  (`8690dc83246b39f8`, `f26175202f3dabe9`); prompts read verbatim from
  `scripts/bench_matrix.json`.
- Nothing was deleted under the policy's consequence list: the f32-score
  composition remains as the fallback for uncovered shapes, oversize
  contexts, and the stock driver - it is now also the arm's bit-exact
  reference, not a rounding-preservation relic.
