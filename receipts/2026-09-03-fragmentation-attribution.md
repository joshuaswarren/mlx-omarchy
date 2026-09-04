# 2026-09-03 — decode fragmentation attribution: why one token becomes ~2,682 evaluations

Agent: FragmentationHunt. Base lineage: `18f59e3` (origin/main) + instrumentation
commits `1640b38`, `f5d5c1a`, `c5f18e3`, `105446b` on branch
`fragmentation-hunt` (pushed). All measurements on the x86 llvmpipe dev box,
deterministic decode (temp 0, 48 generated tokens, 20-token prompt,
`MLX_DISABLE_COMPILE=1 MLX_OMARCHY_ALLOW_NON_APPLE=1`).

Loaded wheel (attribution + name tables):
`mlx_omarchy-0.32.2.dev202609031912+diag` lineage, final prototype wheel sha256
prefix `d978d5fb60a042d3`, source `105446b`. Probe:
`scripts/fragmentation_probe.py` (committed). Raw artifacts: `/tmp/frag-p.jsonl`
(profiler stream), `/tmp/frag-events*.jsonl` (boundary crossings),
`/tmp/frag-prims.csv` (primitive names), `/tmp/frag-m*.jsonl` (token markers).

## 1. Per-token counts with attribution (llvmpipe, f5d5c1a-lineage wheel)

Per generated token, bit-stable across three runs:

| counter | per token |
|---|---|
| gpu::eval calls (primitives) | 2,682 |
| compute dispatches recorded | 1,953 |
| vkQueueSubmit | 97 |
| gpu::finalize calls | 97 |
| commits with work | 97 |
| noop commits | ~4 (193/48) |
| buffer copies / fills | 402 / 1 |

Python/MLX boundary crossings per token: **exactly two.**

| boundary call | site | n/token | host us | submissions caused |
|---|---|---|---|---|
| `mx.async_eval` | mlx_lm generate.py:460 | 1 | ~0.99-1.6 s (includes throttle waits + llvmpipe execution) | 97 |
| `y.item()` | mlx_lm generate.py:466 | 1 | ~5 | **0** |

The loop is clean: one `async_eval` pipelines the next token while `y.item()`
joins the previous one. `y.item()` costs 5 us and zero submissions — the event
generation join (event.cpp `Event::wait`) is the only synchronization and it is
the same contract Metal pays. mlx_lm does not force syncs beyond this.

## 2. The exact upstream call sites

- `mlx/transforms.cpp:80` `eval_impl(outputs, async)` — builds the tape
  (BFS, :180-224) and walks it (:229-309) **one primitive per iteration on the
  calling thread**. `mx.async_eval` (:350) and `mx.eval` (:364) both run this
  walk inline; async only attaches events (:236-247).
- `mlx/transforms.cpp:265-269` — `gpu::eval(arr)` per primitive → one backend
  `eval_impl` per primitive. This is the fragmentation itself.
- `mlx/transforms.cpp:271-285` — the throttle: `MAX_ACTIVE_TASKS = 10`
  (:25) → `gpu::finalize(s)` + `scheduler::wait_for_one()` per open stream.
- `mlx/transforms.cpp:337-345` — epilogue: `Event::signal` then
  `gpu::finalize(s)` per open stream.
- `mlx/scheduler.h:36-64` — task counter lives in the BACKEND's hands
  (`notify_new_task`/`notify_task_completion`), `wait_for_one` waits for one
  completion.
- `mlx/transforms.cpp:51-59` — the `Synchronizer` primitive (name
  "Synchronize" in my table): empty `eval_gpu`, one per eval, harmless.
- mlx_lm 0.31.3 `generate.py:396-470` — `_step` builds the graph with zero
  evals; `:455/:460` `mx.async_eval`; `:466` `y.item()`; `:442` prefill-chunk
  cache eval; `:468` `mx.clear_cache()` every 256 tokens.
- `sample_utils.py:46-47` — `make_sampler(temp=0)` is `mx.argmax`, no compile,
  no eval. mlx_lm 0.31.3 compiles NOTHING in the decode loop (the five
  `mx.compile` uses are sampler filters, inactive at temp 0).

## 3. What the ~2,682 evaluations are (Q1), per name per token

From `/tmp/frag-prims.csv` (÷48 tokens; includes a few prefill evals, so
figures are ±2%):

| category | primitives | per token |
|---|---|---|
| Broadcast | 25,287 | ~527 |
| Multiply | 19,191 | ~400 |
| Add | 10,795 | ~225 |
| QuantizedMatmul | 8,444 | ~176 (7/layer ✓) |
| Slice | 7,246 | ~151 |
| AsType | 7,197 | ~150 |
| Squeeze | 5,095 | ~106 |
| Synchronize (event carrier, empty) | 4,848 | ~101 total, ~2/token |
| Transpose / Reshape | 4,798 each | ~100 each |
| Max / Arange / Abs | 4,798 each | ~100 each |
| ExpandDims / Subtract / RMSNorm | ~2,448 each | ~51 each |
| SliceUpdate / Sin / Exp / Cos / Concatenate | ~2,399 each | ~50 each |
| Sigmoid | 1,199 | ~25 |
| ScaledDotProductAttention | 1,199 | ~25 (1/layer ✓) |
| Gather / Quantize / LogSumExp / ArgReduce / Full | 150/50/49/49/48 | ~1-3 each |

Categories: (a) view/metadata ops that record nothing (Broadcast, Transpose,
Reshape, Squeeze, ExpandDims, Slice-views) ≈ 890/token; (b) real compute the
model needs (QuantizedMatmul, SDPA, RMSNorm, Concatenate) ≈ 260/token;
(c) scalar/positional math re-derived every token (Arange, Sin, Cos, Exp, Abs,
Max around the rope fallback) ≈ 460/token; (d) elementwise/cast/copy plumbing
(Multiply, Add, AsType, SliceUpdate, Sigmoid, Subtract) ≈ 1,070/token.

## 4. Does Metal execute these? (Q2)

The view/metadata classes are handled by the SHARED gpu backend
(`mlx/backend/gpu/primitives.cpp:23-272`): `AsStrided/Broadcast/ExpandDims/
Squeeze/Split/Transpose::eval_gpu` all call the common `eval()` = shared-buffer
copy, **no dispatch on any backend**. Metal has no `Reshape::eval_gpu` of its
own for these either (metal/primitives.cpp contains only Arange/ArgReduce/Load/
RandomBits + NYI stubs). So Metal walks the same ~2,682 primitives with the
same per-category dispatch/no-dispatch behavior. Nothing in our backend forces
a materialization Metal avoids — the one candidate I audited
(`materialize_strided_slice`, eval.cpp:43-81) records a GPU gather, which is
what `slice_gpu` does on Metal too, and it performs no host read.

The one true asymmetry found: **fused RoPE**. `nn.RoPE` is `mx.fast.rope`
(positional_encoding.py:44-53); `mlx/fast.cpp:549-573` composes rope from
Arange/Exp/Sin/Cos/Multiply/... when the backend has no fused kernel (Metal:
`rope.metal`, one dispatch). Our backend has no fused rope, so we pay ~460
scalar-math evaluations + dispatches per token that Metal does not. Fix is a
new Vulkan rope kernel in the Omarchy backend (same shape as the Qmm fusion
work), not a loop change.

## 5. Cost per evaluation (Q3) and where the wall sits

llvmpipe per-token backend-active costs (profiler h/dur fields, 47-token
windows): dispatch record 2.20 ms (1,953 × p50 0.88 us), submit 0.66 ms,
begin+slot-wait 4.66 ms, join wait 0.02 ms → encoder-active ≈ 7.5 ms/token.
The remaining ~99% of the ~1.0 s/token llvmpipe wall is throttle waits
executing lavapipe software batches — llvmpipe-specific, not transferable.
A no-dispatch evaluation costs the tape-walk step + `eval_impl` bookkeeping
only (eval.cpp:91-151); it records no GPU work and submits nothing.

On the M1 the corresponding wall (HostPathOptimize: >99% outside the encoder;
~97 fragments/token) is consistent with ~97 submit-to-completion round trips
at ~2.6 ms each. Metal pays the same upstream throttle with ~20-op commits and
a ~20-30 us driver round trip. The transferable statement: wall ≈
fragments/token × per-fragment round-trip latency, and llvmpipe cannot measure
either M1 factor.

## 6. Intrinsic vs artifact

Intrinsic (upstream contract; absorb, do not fight): per-primitive tape walk
(transforms.cpp:229-309); the MAX_ACTIVE_TASKS=10 finalize+wait_for_one
throttle; the epilogue signal+finalize; one `.item()` per token
(generate.py:466); one-token-deep async pipeline (generate.py:459-460).
Artifact, removable, with owners: (1) missing fused RoPE — ~460 evals +
~460 dispatches/token vs Metal — backend kernel work; (2) the
cast/copy/elementwise plumbing (~1,530 dispatches/token here) — the fusion
program GemvDecode/FuseDecodeChains already own; (3) fragment count × round
trip on M1 — my per-eval task-accounting prototype aims here and is
llvmpipe-negative as built (below); (4) compiled tapes — the real collapse
mechanism, blocked by the tape defect TapeCorruptionFix owns.

## 7. Prototype result (negative, on record)

Per-eval task accounting (`105446b`, TinyWriteFix-reviewed, lazy decrement via
first-commit attachment, host-complete fallback, memory-guard-safe): llvmpipe
counts did not move — still 97 submissions / 97 finalize / 2,682 evals per
token (wheel d978d5fb). The llvmpipe fragment cadence therefore is not set by
per-batch task accounting; the true driver of ~27.6 primitives per fragment is
unidentified (it matches neither kBatchNodeBudget=100 nor task-count
arithmetic). Not shipped as an improvement; the M1 A/B in the protocol decides
whether the M1's throttle behaves differently (its completion dispatcher is
not llvmpipe's synchronous one).

## 8. Projected floor with arithmetic (M1, conditional)

If the M1 wall is fragments × round-trip (hypothesis H: ~97 × 2.6 ms ≈ 252 ms
≈ the observed ~258 ms/token): per-eval accounting + budget knob → fragments ≈
2,396 nodes / 100-node budget ≈ 24 + epilogue 1 ≈ 25 → ~25 × 2.6 ≈ 65 ms →
~15 tok/s; with budget 1,000 → ~3 fragments → ~8 ms → ~125 tok/s; with fused
rope + cast-fusion (nodes/token ~500) and budget 1,000 → ~1-2 fragments →
bounded by GPU busy ~1.5 ms + walk → ~250-650 tok/s. If H is false
(fragments already cheap on M1), the wall sits in the ~2,682 evals' host cost
(~2,682 × x us = 258 ms → x ≈ 96 us) and the same levers reduce the multiplier
instead. Either way the levers are the same; the M1 A/B distinguishes the
mechanism, not the action.

## 9. Hardware protocol for BenchQueueM1 (jwm1)

1. Build two diagnostics wheels from `fragmentation-hunt` at `105446b` on the
   M1: one stock, one with `kBatchNodeBudget` raised 100 → 1,000
   (encoder.h:52). Install each into its own fresh venv with mlx_lm 0.31.3
   from the pinned wheel; record wheel sha256 + commit per Main's provenance
   rule.
2. Run `MLX_DISABLE_COMPILE=1 MLX_OMARCHY_GPU_PROFILE=<p> python3
   scripts/fragmentation_probe.py --model
   ~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/
   snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3/ --prompt "The quick
   brown fox jumps over the lazy dog. Pack my box with five dozen liquor
   jugs." --max-tokens 48` per leg; 3 runs each, discard first (warm).
3. Record per leg: tokens/s, per-token submissions (probe table), fragments,
   join wait p50, wall per token, and HostPathOptimize's encoder-active
   fraction. Decision rule: keep the branch's accounting change only if
   tokens/s improves ≥20% over stock on the same wheel lineage; keep the
   budget-1000 leg only if the battery on jwm1 stays green (temporaries
   pinning grows 10x — watch allocator active memory in the same runs).
4. The full 25-binary battery on jwm1 must pass for any keep; the new
   `test_task_accounting` binary must be built and green (it drives the
   memory-guard mid-walk path deterministically; it is written, NOT yet run —
   its first build+run is part of step 1).

## Honesty notes

- `overlay/tests/omarchy/test_task_accounting.cpp` is written but never
  compiled or run (wheel build has tests OFF; no test binary was built this
  session). It needs its first build+run on the next test build.
- The llvmpipe projection numbers in section 8 are labeled conditional on
  hypothesis H; only the M1 legs in section 9 are measurements-to-be.
- Three of my earlier llvmpipe figures (sections 1, 3) come from the
  f5d5c1a-lineage wheel; the prototype run used d978d5fb. Counter totals are
  identical across both, which is what makes the section-7 negative valid.
