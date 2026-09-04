# SDPA f16-scores rework: three fixes landed, copy-count prediction refuted

Date: 2026-09-04. Branch `sdpa-f16-scores` (off `ew-fused-elementwise`
tip `77c0b95`, which is `origin/main` + the 5 census/instrumentation
commits), pushed. Implements addendum 3 of
`2026-09-04-elementwise-chain-census.md`.

Status: PROVISIONAL until the M1 leg (BenchQueueM1) runs the
consolidated protocol. Every llvmpipe number below carries the
provenance line per the new fleet rule.

## Commits

- `641b96a` census: SDPA equivalence suite and per-token count
  instruments (`scripts/sdpa_equivalence.py`, `scripts/sdpa_counts.py`)
- `f231704` fix 1: f16 scores with f32 accumulation in the matmul
  shaders (alpha scale, f16 softmax storage, no output downcast,
  size-1 inner-axis guard in the old operand check)
- `59de953` fix 2: stride-aware matmul operands (lhs_gap/rhs_gap push
  constants, free batch strides, classifier replaces
  is_batched_matrix, SegmentedMM migrated strict, env-gated
  MLX_OMARCHY_TRACE_MATERIALIZE)
- `7770d4c` fix 3: GQA regroup through broadcast views (explicit
  stride views for the (kv, repeat) regroup and the transposed keys)

## Measured, llvmpipe, pinned model and prompt

Model: mlx-community/Qwen2.5-0.5B-Instruct-4bit, snapshot
`a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3`; mlx-lm 0.31.3;
chat-template prompt "What is the capital of France?", 32 max tokens,
temp 0, seed 0; `MLX_OMARCHY_ALLOW_NON_APPLE=1`,
`MLX_DISABLE_COMPILE=1`.

Provenance per build (all verified=match via
`scripts/mlx_provenance.py` against the installed wheel):

- baseline `77c0b95`: wheel
  `mlx_omarchy-0.32.2.dev202609040148+diag.77c0b95`
  sha256 `92b768844449924ede98f829c1f1266aa0742c2e42103f963dfb92d7862093c5`
- fix 1 build: wheel `...dev202609040201+diag.77c0b95`
  sha256 `5a2adbcd374b3fda0f983efaab064cc5f0d54e2ff9ea2e0652e28d0b21a5b5a5`
- final (fix 2 + fix 3): wheel
  `mlx_omarchy-0.32.2.dev202609040243+diag.f231704`

Median decode token:

| metric | baseline | fix 1 | fix 2 | fix 3 |
|---|---|---|---|---|
| dispatches/token | 1,953 | 1,833 | 1,833 | 1,833 |
| CopyGeneralF16/token | 288 (12/layer) | 288 | 288 | 288 |
| casts/token | 240 | 144 | 144 | 144 |
| MatmulF32 / MatmulF16 | 48 / 0 | 0 / 48 | 0 / 48 | 0 / 48 |
| SoftmaxF32 / SoftmaxF16 | 24 / 0 | 0 / 24 | 0 / 24 | 0 / 24 |
| ElementwiseF32 | 456 | 432 | 432 | 432 |

The 4-6-copies-deleted-per-layer prediction from addendum 3 is NOT
confirmed: the census-instrument-visible copy count did not move.

## Finding: the census copy attribution was wrong

Addendum 2 attributed 12 CopyGeneralF16 per layer per token to the KV
path and predicted 4-6 deletable from SDPA. Three checks refute that:

1. `MLX_OMARCHY_TRACE_MATERIALIZE=1` (new, env-gated) records ZERO
   operand materializations in `dispatch_matmul` across a full
   generation on the final build - the slice densification
   `to_f32`/`materialize_batched_matrix` used to do is gone, and no
   copy moved into the matmul.
2. A probe running the decode step with DENSE k/v inputs
   (`mx.contiguous(k)` before sdpa) produces the same recorded copy
   stream as the sliced inputs - the recorded growing copies do not
   depend on slice consumption.
3. The addendum's stage labels are misleading: stage B in
   `scripts/kvcopy_decompose.py` ("B-update+sdpa") contains NO sdpa -
   it is update + `k.sum() + v.sum()`; only stage C runs attention.
   Stage B's growing copies therefore belong to the reduction/update
   consumption path, not to SDPA.

Per-token copy size classes at the median token (identical in baseline
and final): `64 x96, 128 x48, 448 x96, offset x128 x48` - the growing
class is a per-layer pair of `[1, 2, offset, 64]` state
materializations (offset 38-44 across this run's 8 decode tokens,
i.e. 4864-5632 elements) that sits OUTSIDE SDPA's operand handling.
Attributing it precisely (cache realloc vs slice-assign vs a consumer
densifying for a non-matmul kernel) is open work for whoever owns the
cache path; the instruments to do it are now in the repo.

## What the rework actually deleted

- 3 q/k/v upcasts per layer (CastF16F32 72 -> 0)
- the output downcast per layer (part of CastF32F16 120 -> 96)
- the f32 scale broadcast + multiply per layer (ElementwiseF32 456 ->
  432)
- MatmulF32 48 -> MatmulF16 48 and SoftmaxF32 24 -> SoftmaxF16 24 with
  no accumulation-order change (matmul.comp accumulates in float under
  USE_FP16; softmax_suffix.comp runs float math over f16 storage -
  both verified in the shader source before coding)
- structurally, every matmul operand materialization on the SDPA path
  (trace-verified zero), which the llvmpipe census could not see as a
  dispatch delta but which is real work on any driver

## Equivalence

`scripts/sdpa_equivalence.py`: 17/17 cases pass at every kept state
(max primitive-vs-f32-reference output error 4.9e-4, bar 1e-3; max
emulated f16-storage scores error 5.7e-4, bar 2e-3). Coverage: cache
slices at offsets 1/7/41/256, GQA repeats 1/2/7, head_dim 48, q_len 1
and >1 (dense and against a cache slice), causal and additive f16
masks, batch 2.

Greedy: "The capital of France is Paris." token-identical at every
state.

Batteries (llvmpipe, final state): runtime 25 cases / 6,247
assertions, eq_math 7 / 116, compiled_tape 11 / 1,747 - green.

## Caveats carried forward

- f16 scores saturate at f16 max (65504): the causal additive uses
  -inf and a fully-masked row would NaN where the f32 path stayed
  tiny; causal shapes always expose the diagonal, and greedy is clean,
  but the M1 protocol should watch for score-overflow on real
  workloads.
- llvmpipe cannot see asynchronous or driver-specific defects; the M1
  leg is the gate.
