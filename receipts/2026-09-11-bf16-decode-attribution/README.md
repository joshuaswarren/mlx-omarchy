# BF16 decode token attribution — receipt (2026-09-11)

Status: COMPLETE. Receipt-only negative: the attribution table is
complete (residual closes at +0.2%/-2.7%/-1.8%), the one bit-preserving
attack lever was screened and measured dead, so nothing lands and no
digest moves.

## Question

Nobody had attributed a BF16 decode token since the dense vector kernel
landed (2026-09-10). At 0.565/0.517/0.424 of native (30/262/1053) BF16
decode was the weakest decode surface. Attribute one BF16 decode token
per kernel class by ablation, confirm hook engagement by digest
movement, and attack the dominant class.

## Method (same instrument as receipts/2026-09-10-decode-attribution)

Measurement-ablation wheel: the Q4 hook (74f170e4) cherry-picked onto
main a5b8c4ab (jwm1 wheel-tree branch wave/Bf16DecodeAttribution,
03248d32) and extended with the BF16 decode classes (d2ef0db0). Classes
from the dispatch census: gemv (MatmulVecBF16+MatmulBF16), attn
(MatmulF32+MatmulF32Coopmat+SoftmaxF32), cast (CastBF16F32+CastF32BF16),
ewise (ElementwiseF32), copy (CopyGeneralBF16), rope (FastRopeF32), rms
(FastRmsNormBF16), swiglu (SwigluBF16), sampler (LogSumExpBF16+
ArgReduceBF16), all. Same contract: dispatch shape, grids, loop walks
and output stores preserved; payload loads and math removed. All 14
production shader combos compile byte-identical SPIR-V (glslc -O
--target-env=vulkan1.3, verified per target); every abl variant compiles
and differs. The wheel is NEVER merged and never digest-gated.

Engagement gate (the Q4 lesson): baseline legs assert the canonical
per-driver digests; every ablated leg asserts the digest MOVED. Baseline
13/13/13 reps per leg; class arms 6-7 per leg; interleaved rounds;
per-leg peak loadavg recorded (quarantine at >= 3.0, none quarantined).
Wheels: pristine a5b8c4ab release
(mlx_omarchy-0.32.2.dev202609111524+a5b8c4a,
sha256 0c61e0b9d66b445c89e7be72bd2c6f5a55f1e1b8041bd3f4ea4abade7da87f25)
and ablate release (dev202609111520+d2ef0db,
sha256 52fb24287056eb7b4947787e8a52bd9f1e8d21340f11aedb41acdbc2dbaae435).

## Dispatch census (wall-anchored profile runs, /tmp census files copied
to census/)

A BF16 decode token issues 726 dispatches, identical at short and
ctx1024. Per-token composition (median token):

| kernel | per token | share |
|---|---|---|
| CastBF16F32 | 120 | 16.5% |
| MatmulVecBF16 | 97 | 13.4% |
| CopyGeneralBF16 | 96 | 13.2% |
| MatmulBF16 | 72 | 9.9% |
| CastF32BF16 | 72 | 9.9% |
| FastRmsNormBF16 | 49 | 6.7% |
| FastRopeF32 | 48 | 6.6% |
| MatmulF32 | 48 | 6.6% |
| BinaryVecBF16 | 48 | 6.6% |
| ElementwiseF32 | 24 | 3.3% |
| SoftmaxF32 | 24 | 3.3% |
| SwigluBF16 | 24 | 3.3% |
| sampler (LogSumExp+ArgReduce) | 2 | 0.3% |

The f32-composed attention IS present on decode (312/726 dispatches
including casts): SdpaDecodeNativeF16 requires q.dtype()==float16, so
bf16 falls through to the generic composition. But its PAYLOAD is not
where the time goes (see table).

## Attribution table (window A arms: baseline, gemv, attn, cast; window B
adds ewise, copy, rope, rms, swiglu, sampler, all)

See verdict.json (written at completion) and arms/legs.ndjson. Window A
result, ms/token (marginal = baseline median minus ablated median):

| arm | short 30/32 | long 262/128 | ctx 1053/32 |
|---|---|---|---|
| baseline | 31.43 | 34.69 | 43.29 |
| gemv | 18.31 (58.3%) | 17.96 (51.8%) | 18.06 (41.7%) |
| attn | 0.28 (0.9%) | 0.82 (2.4%) | 2.62 (6.1%) |
| cast | 0.10 (0.3%) | 0.02 (0.1%) | 0.15 (0.3%) |

BF16 decode attribution is the dense GEMV class at every leg. The
f32-composed attention costs structure (dispatch count), not payload.

## MLP/kv routing finding (Main's question)

No shape guard excludes the MLP or k/v projections: the isolation probe
(probe/) runs gate (m=1 k=896 n=4864), down (m=1 k=4864 n=896), kproj
(n=128) and every other decode shape through the PRISTINE wheel and each
one dispatches MatmulVecBF16. The in-model tile routing (MatmulBF16
3x/layer = down + k + v) is runtime state, not shape logic; the vec
guard's word-index alignment requirement (the kernel indexes
(base + k) / 4u over uvec2) is the suspect, tested by the
vec-exclusion probe (real safetensors-backed weights vs contiguous
copies).

## Final table (all arms, medians; verdict.json + arms/legs.ndjson)

ms/token, marginal = baseline minus ablated; residual = sum(marginals) +
skeleton minus measured token:

| class (disp/tok) | short 30/32 | long 262/128 | ctx 1053/32 |
|---|---|---|---|
| **gemv (169)** | **18.31 (58.3%)** | **17.96 (51.8%)** | **18.06 (41.7%)** |
| skeleton (all ablated) | 11.86 (37.7%) | 14.69 (42.4%) | 21.58 (49.9%) |
| attn (72) | 0.28 | 0.82 | 2.62 |
| rms (49) | 0.33 | 0.30 | 0.39 |
| ewise (24) | 0.34 | -0.07 | -0.02 |
| rope (48) | 0.23 | 0.10 | 0.07 |
| cast (192) | 0.10 | 0.02 | 0.15 |
| swiglu (24) | 0.01 | 0.11 | -0.19 |
| copy (96) | 0.05 | -0.13 | -0.19 |
| sampler (2) | -0.03 | -0.06 | 0.01 |
| **residual (explicit)** | **+0.054 (+0.2%)** | **-0.950 (-2.7%)** | **-0.791 (-1.8%)** |

Baseline fractions of native: 0.565 / 0.517 / 0.423 (matches the
committed matrix within noise). Residuals close; negative residuals are
per-class interaction/noise, marginals are upper-bound-fair. The
skeleton grows with context because the no-work bodies keep their loop
walks (attention-length loops), so context-dependent GPU time is
booked there.

sampler is digest-neutral by construction in this engine: logsumexp and
argreduce feed the reported logprobs, not the greedy token choice, so
their ablation cannot flip the stream (asserted per-leg for every other
class; sampler legs keep the canonical digest). Their marginal is the
wall-time delta: zero at every leg.

## Attack (screened in window B)

Bit-preserving column-group widening of MatmulVecBF16: the shader
already grid-strides row_groups (4 output columns per group), so any
workgroup span >= 4 keeps the per-column lane mapping and accumulation
order exactly; outputs are bit-identical by construction and the screen
asserts output digests across spans. Screen wheel
(dev202609111652+757eb5e, MLX_OMARCHY_VEC_WG_SPAN env, default 4 =
today's dispatch shape). Screen results + decision in verdict.json.

Lm_head context: 272 MB/token of the ~370 MB class; the isolation probe
puts the single lm_head matmul at ~9.3 ms wall (~36 GB/s effective
against the 58.5 GB/s same-instrument copy roof and Q4's measured
41 GB/s practical in-model rate).

## Screen result and decision: receipt-only negative

Span screen (eager eval rounds, 24 rounds per shape per span, wall
medians, ms; outputs bit-identical across spans for every shape, screen
digests asserted):

| span | lmhead | gate | down | qproj | kproj |
|---|---|---|---|---|---|
| 4 (today) | 4.687 | 0.399 | 0.380 | 0.231 | 0.214 |
| 16 | 4.703 | 0.394 | 0.489 | 0.262 | 0.217 |
| 64 | 4.707 | 0.501 | 1.100 | 0.385 | 0.397 |
| 512 | 4.772 | 1.111 | 2.529 | 0.719 | 0.419 |

Base span 4 is optimal; every wider span is neutral-to-worse and
degrades monotonically on the small shapes. The bit-preserving
column-group lever is measured dead for this kernel, the same outcome
as the Q4 workgroup-remap arm in receipts/2026-09-11-q4-memory-roof.
No repeating gain, so per the assignment NOTHING lands: this receipt is
the deliverable, pushed on branch `bf16-decode-attribution`. The
screened patch (env-tunable span, wave/Bf16Col16Screen 757eb5e) and the
drafted decode-shape regression test stay in receipts-work as
not-landed artifacts.

## Residual structural finding (for the next attempt, not actioned)

The second cost is the skeleton itself: 11.9 ms/token at short with
every payload removed, ~16 us per dispatch across 726 dispatches -
versus Q4's 2.53 ms at 273 dispatches. The dominant removable structure
is the f32-composed attention (312 of 726 dispatches are
attention-casts/matmuls/softmax): its payload measures ~0, so a
bit-exact fused decode attention (bf16 loads, exact in-register f32
upcast, f32 internals, today's accumulation order) buys the dispatch
structure, not arithmetic. That is a new-kernel change with a
digest-equality gate at BF16 1K (ff502900d2a179a5 must hold; it is
native-matched at full rule-2 strength) and is out of scope here.

## MLP/kv tile-routing note (honest bounds)

All decode shapes - including gate (896->4864), down (4864->896) and
k/v with bias - dispatch MatmulVecBF16 when driven in isolation, with
the model's real safetensors-backed weight buffers (probe/
vec_exclusion.txt: 16 of 16 matmul events on the vec kernel, both real
and contiguous operand sets). In the live engine, k/v and down still
ride the 16x16 tile (census: 264+528 events/token-run). The isolating
condition was not reproduced, so the routing split stays unexplained;
its payload bound is small: down + k + v are ~9.2 MB of the ~370 MB
class, so full vec routing is worth at most a few tenths of a
ms/token. Recorded as an open question; it is not the wall.

## Digest gates

Q4 digests untouched: only measurement wheels ran against them and the
six canonical values were never exercised on a modified kernel; the
screen wheel (757eb5e) keeps default dispatch byte-identical and was
used only for the screen. BF16 pins: no source change landed, so no pin
moved; the ablation/screen legs are not pin evidence. bf16fast A/B rows
in legs.ndjson independently confirm the prefill strand's digest
findings (short bit-neutral f26175202f3dabe9, long -> 9bb3388dbce0b4c8,
ctx -> 5fd2fe812bf2a4ce).

## Provenance

- Host: jwm1-linux (Apple M1 G13G B1), driver package verified
  mesa-honeykrisp-omarchy 26.3.0.devel.hk6f6afc8-1 before each window;
  every window one top-level flock /tmp/m1-gpu.lock, wall-anchored
  timing only (device timestamps unused).
- Wheels: base a5b8c4ab
  mlx_omarchy-0.32.2.dev202609111524+a5b8c4a (sha256 0c61e0b9...da87f25),
  ablate d2ef0db0 dev202609111520+d2ef0db (52fb2428...dbaae435) rebuilt
  as dev202609111726+d2ef0db (78feb0e7...2a073b) after the screen build
  replaced dist content (same source commit), screen 757eb5e
  dev202609111652+757eb5e. Census/probe diag wheel a5b8c4ab
  dev202609111457+diag.a5b8c4a.
- Incident log: one lock-contention overlap (a sibling's ~2x45 s GPU
  suite runs during window B arms) and its legs are identifiable by
  rep interleaving; medians over 6-13 reps bound the effect, and no leg
  exceeded the 3.0 loadavg quarantine line (max recorded 2.57).
- Never merged; measurement branches live on jwm1
  (~/src/mlx-Bf16DecodeAttribution: wave/Bf16DecodeAttribution hook
  03248d32 + classes d2ef0db0, wave/Bf16Col16Screen 757eb5e).
