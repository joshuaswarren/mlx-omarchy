# BF16 fused decode attention — receipt (2026-09-11)

Status: COMPLETE. Receipt-only negative. The fusion is implemented, engages,
and wins the payload it was built for (dispatches 726 -> 510 per token,
ctx leg -31% ms/token on the fork), but the BF16 fork-driver generated-id
digest moves on the short and long legs. Per the assignment gate ("any
digest move falsifies the change — report it and stop rather than
re-pinning") nothing lands; the implementation is preserved unmerged on
`wave/Bf16FusedDecode` (72ca97e3, based on main 2a9add42).

## Question

Can the f32-score attention composition on the BF16 decode path (10
dispatches per layer: 3 q/k/v upcast casts, 2 k/v densify copies, the
scale multiply, two matmuls, softmax, the output downcast) collapse into
the single-query fused decode kernel without moving a digest?

## Census (step 1, established from the committed wall-anchored profiles of
receipts/2026-09-11-bf16-decode-attribution and confirmed on this
window's fresh diag wheel)

One BF16 decode token issues 726 dispatches (short == ctx1024). The
ScaledDotProductAttention f32 composition contributes 240 of them —
10 per layer x 24 layers:

| per token | kernel |
|---|---|
| 72 | CastBF16F32 (q/k/v upcasts, 3/layer) |
| 48 | CopyGeneralBF16 (k/v cache-slice densify, 2/layer) |
| 24 | ElementwiseF32 (scale broadcast multiply, 1/layer) |
| 48 | MatmulF32 (scores + probs matmuls, 2/layer) |
| 24 | SoftmaxF32 (1/layer) |
| 24 | CastF32BF16 (output downcast, 1/layer) |

The inherited "312 of 726" over-counted: it charged the 48 kv-cache-write
copies and 24 non-attention elementwise dispatches to attention. The
composition's own share is 240. The f32 attention payload ablation
(0.28/0.82/2.62 ms) already showed these dispatches cost structure, not
arithmetic.

## Change (wave/Bf16FusedDecode 72ca97e3, not landed)

`sdpa_decode_native.comp` gains a `-DBF16_IO` arm of the proven f16
single-query kernel: uint16 word views, cast.comp's exact-widening load
and RNE store pair, every intermediate (scores, exp, running max, running
sum, accumulator, two-pass partials) in float32 per upstream Metal
(`typedef float U`, sdpa_vector.h:50). The bf16 host side always selects
the one-pass 32-stream merge: the f32 partial array the 128-block
two-pass needs cannot fit the M1 workgroup store at the crossover, and
the bf16 parity contract is the f32 composition, not native's two-pass
partial format. The f16 arm compiles to byte-identical SPIR-V (glslc -O
--target-env=vulkan1.3, verified against the pre-change blob), so the f16
route is provably untouched. Guards are unchanged: uncovered shapes
(refusals verified for q_len=2 and additive-array-mask decode) fall
through to the existing composition by name. Regression:
`omarchy_sdpa_decode_fused_tests` (one-dispatch contract, agreement with
the f32-score composition on the covered decode shape, refusal
fall-throughs) plus the touched omarchy suites pass on the dev box
(fast_ops 33/33 cases, 1,116,299 assertions).

## Gates and measurement (one flock window on the M1; Apple M1 G13G B1,
driver mesa-honeykrisp-omarchy 26.3.0.devel.hk6f6afc8-1; host recorded as
placeholder jwm1-linux)

Paired wheels from the same tree: base = main 2a9add42
(mlx_omarchy-0.32.2.dev202609112136+2a9add4, sha256 e47d7b2f...dd16812),
fused = 72ca97e3 (dev202609112149+72ca97e, sha256 7c999fd0...5b88570),
diag = dev202609112139+diag.72ca97e (sha256 c5589105...8e99f542). One
discarded warmup then 3 reps per leg per driver (bench_matrix, pinned
revisions, MLX_DISABLE_COMPILE=1, greedy, EOS suppressed); every rep's
generated-id digest asserted.

Digest outcome (every rep identical within a cell):

| leg | driver | base | fused | native |
|---|---|---|---|---|
| Q4 short/long/1K | fork + stock | 7fd25a86... / 4cc08910... / 7da83f06... | same — HOLD | — |
| BF16 1K | fork + stock | ff502900d2a179a5 | ff502900d2a179a5 — HOLD (native-matching pin) | ff502900d2a179a5 |
| BF16 short | stock | f26175202f3dabe9 | f26175202f3dabe9 — HOLD | 7fc0f968789b1882 |
| BF16 short | fork | f26175202f3dabe9 | **7fc0f968789b1882 — MOVED** | 7fc0f968789b1882 (= fused) |
| BF16 long | stock | c5be9207833d2a26 | c5be9207833d2a26 — HOLD | 407b7624ed1b3b29 |
| BF16 long | fork | 8690dc83246b39f8 | **06b014aebf5d4b66 — MOVED** | 407b7624ed1b3b29 (neither old nor new) |

The base wheel itself already differs from the older committed BF16
short/long fork pins because main moved between the last pin record and
this window (the bf16 coopmat alpha fix, 17ebab3f, landed by
AlphaProbeDecide and re-pinned on the landed tree); the A/B here is
internally clean — same tree, only the fusion commit differs.

ms/token (median of 3, decode_mean_per_token_ms):

| leg | base fork | fused fork | base stock | fused stock |
|---|---|---|---|---|
| short 30/32 | 31.2 | **29.0** (-7.1%) | 31.1 | 31.0 |
| long 262/128 | 34.7 | **29.6** (-14.7%) | 34.3 | 34.3 |
| ctx 1053/32 | 43.3 | **29.8** (-31.2%) | 43.0 | 43.3 |

Stock times are identical base-vs-fused because the fused route refuses
on stock (capability gate) and falls through to the composition — the
stock stream is untouched by construction. Fractions of the committed
native baseline (17.78/17.98/18.43 ms/token) on the fork move from
0.566/0.518/0.426 to **0.613/0.607/0.618**.

Dispatch census, fused wheel (fresh diag profile, 10 token buckets):
726 -> **510** dispatches/token (-216). SdpaDecodeNativeBF16 +24 (1/layer);
removed exactly the composition classes above (CastBF16F32 -72,
CopyGeneralBF16 -48, ElementwiseF32 -24, MatmulF32 -48, SoftmaxF32 -24,
CastF32BF16 -24). Every other class unchanged.

## Verdict

The fusion wins its payload — one dispatch per attention, 30% of the
token's dispatch count, -7% to -31% ms/token on the fork, stock stream
provably untouched, and the native-matching 1K pin holds everywhere
including on the engaged fork path. But the assignment's gate is
absolute: the BF16 fork short and long digests moved, so the change does
not land and is not re-pinned. For the owner's future decision, the
policy framing is: the fork short-leg move lands exactly on native's
digest (7fc0f968789b1882), which parity rule 1 admits; the fork long-leg
move lands on a third value (06b014aebf5d4b66) that is neither the old
digest nor native's 407b7624ed1b3b29, which rule 1 rejects — a
one-pass-online-softmax vs full-matmul accumulation-order divergence at
greedy-unstable tokens in the 128-token leg. Closing that gap would mean
reproducing the composition's matmul accumulation order in-kernel (or a
re-pin with evidence), not more timing work.

## Provenance

- Host: jwm1-linux (Apple M1 G13G B1), driver verified
  mesa-honeykrisp-omarchy 26.3.0.devel.hk6f6afc8-1 before the window;
  one top-level flock /tmp/m1-gpu.lock for the whole window; wall-anchored
  timing (bench_decode monotonic clock), device timestamps unused.
- Wheels: base 2a9add42 (e47d7b2f...), fused 72ca97e3 (7c999fd0...),
  diag 72ca97e3 (c5589105...); full sha256 above; venvs per wheel,
  mlx-lm 0.31.3; stock runs via the stock Mesa ICD override.
- Raw data: /tmp/bf16fused-run on the M1 host (16 bench JSONs, census
  NDJSON + markers); analysis in this receipt.
- Never merged; the implementation lives on `wave/Bf16FusedDecode`
  (72ca97e3) with the regression test.
