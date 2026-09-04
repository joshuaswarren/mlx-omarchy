# Host phase costs: ranked breakdown, kept and reverted

Date: 2026-09-03. Agent: HostPathOptimize. Branch: `hostcost`, base
`origin/main` at `4c44e19` (rebased; commits `c98ec1f`, `952f0b9`,
`f276c10`).

## What llvmpipe can and cannot prove

Host STRUCTURE (which phases cost what, in what order) reproduces across
drivers, so the ranking below is valid dev-box evidence for where host
work lives. Wall times on this box are NOT product numbers: llvmpipe is
a software renderer, its per-dispatch GPU cost (~200-330 us) dwarfs the
per-dispatch host recording cost (~2 us), and run-to-run wall spread is
about 10 percent. Every llvmpipe wall number below is therefore context
only. Ship decisions belong to the M1 protocol (handed to BenchQueueM1).

Method: fixed chain of 64 dispatches x 50 reps, elems=1024, f32, two
shapes (`dep` = dependent chain, `indep` = independent fragment). Both
shapes measured 1 submission per 64-dispatch rep on current main, so
the batched shape is the ranked one. Instrument: the compile-time-gated
NDJSON harness, extended by `c98ec1f` with per-phase host costs
(`b` split into ring-wait/BeginCommandBuffer; `d` gains lookup,
desc-alloc, desc-update, pre-barrier, bind+push+dispatch, post-barrier;
`s` split into EndCommandBuffer/assembly/flush/QueueSubmit).
`scripts/profile_analyze.py` prints the ranked table with p50/mean/p90/max;
the max column is what keeps a one-time cost (pipeline create ~13 ms,
first-QueueSubmit JIT ~17 ms) from being misread as steady state.

## Ranked host phases at baseline (steady state, per dispatch)

Raw data: `/tmp/hostcost/base_{dep,indep}.jsonl` (analysis in
`/tmp/hostcost/base_phases.txt`, process-local).

| rank | phase | dep mean | indep mean | note |
|---|---|---|---|---|
| 1 | bar-pre | 297 ns | 270 ns | 2 barriers recorded per dispatch |
| 1 | bar-post | 432 ns | 355 ns | always recorded |
| 2 | bind+push+dispatch | 486 ns | 376 ns | includes CmdBindPipeline rebind |
| 3 | desc-alloc | 219 ns | 189 ns | AllocateDescriptorSets per dispatch |
| 4 | desc-update | 110 ns | 125 ns | already a single driver call |
| 5 | pipeline lookup | 30-40 ns | 30-40 ns | p50; mean polluted by one-time create |

Per-submission phases amortize to noise at 64 nodes per submission
(assembly ~2 us, flush ~0.3 us, end-cmdbuf ~0.1 us, QueueSubmit p50
8.4 us). Per-begin ring-wait p50 ~112 us is GPU-bound, not host waste.
Attack order chosen: barriers, then bind skip, then descriptor chunking.

## Kept

### 952f0b9 - elide the pre-dispatch barrier after a dispatch

When the previous recorded command in the same command buffer is a
dispatch, its post-barrier (dst COMPUTE, SHADER_READ|WRITE) already
re-establishes exactly what the next pre-barrier would; buffer start,
copy, and fill restore the pre-barrier. The post-barrier is always kept.

| shape | bar-pre mean before | after | per-rep host saving |
|---|---|---|---|
| dep | 297 ns | 37 ns | ~16.6 us (64 dispatches) |
| indep | 270 ns | 40 ns | ~14.7 us |

Values bit-identical (`--check` z=4.500000 both shapes, before and
after). llvmpipe wall cannot arbitrate a 0.25 us/dispatch host change
(spread ~10 percent, GPU-bound) - M1 decides. PROVISIONAL until then.

## Reverted

### Skip CmdBindPipeline when unchanged (candidate A)

Tried: cache the bound pipeline in the command buffer, reset at begin,
rebind only on change. Measured (same fixed chain): bind+dispatch
dep 458 -> 412 ns, indep 396 -> 490 ns - the deltas are inside
run-to-run noise and go in opposite directions, because llvmpipe's
CmdBindPipeline is nearly as cheap as the pointer compare that would
replace it. No measurable host win. Reverted in the working tree
before commit; the diff never landed. Why it failed: the phase is
dominated by CmdBindDescriptorSets + CmdPushConstants + CmdDispatch,
which cannot be skipped (fresh descriptor set per dispatch, params per
dispatch). Conclusion for the next person: pipeline rebind is NOT a
host bottleneck on this path; do not retry without a driver whose
CmdBindPipeline measurably costs more than ~50 ns.

## Not attempted (recorded so it is not silently retried later)

- Descriptor-set chunking (one AllocateDescriptorSets per ~64 sets):
  designed and scoped (members + acquire_descriptor_set only), not
  measured this session. Expected ceiling: most of the 189-219 ns
  desc-alloc phase. Still open.
- Descriptor-update reduction: needs a pipeline-layout redesign (one
  binding with N descriptors instead of N bindings). Out of scope.
- Push descriptors (VK_KHR_push_descriptor): Honeykrisp extension
  support unverified; llvmpipe-only enablement would fork the path.

## Batteries after each kept change (llvmpipe, env unset)

| battery | result |
|---|---|
| omarchy_runtime_tests | 25 cases, 6247/6247 assertions, SUCCESS |
| omarchy_eq_math_tests | 7 cases, 116/116 assertions, SUCCESS |
| omarchy_compiled_tape_tests | 11 cases, 875/875 assertions, SUCCESS |

Session-end state: rebased on `26e8085` (TapeLayerIsolation's
assertion rewrite landed) plus this branch's five commits, all three
batteries green in one pass. During the session the compiled_tape
battery carried one red - the MLX_OMARCHY_TAPE_PER_NODE_SUBMIT
submission-delta assertion (`delta >= 3`, got 2), proven red on
pristine `7c3d6b4` and `4c44e19` (stash + rebuild + rerun) and
instrument-inert by construction; cause was the assertion predating
the 7c3d6b4 flush contract, fixed upstream by `26e8085`.

## Commits

- `c98ec1f` feat(omarchy): host phase-cost breakdown in the GPU profiler
  plus a fixed-chain bench (measurement harness)
- `952f0b9` perf(omarchy): elide the pre-dispatch barrier after a
  dispatch in the same command buffer (KEPT, provisional on M1)
- `f276c10` fix(omarchy): bench value check uses a scalar chain;
  analyzer gains p90/max columns (harness follow-up)

## Consolidated M1 protocol (handed to BenchQueueM1)

One protocol, two legs, same machine/model/thermal procedure, warm
steady state, interleaved A/B order to cancel drift. Commit under test:
`hostcost` tip (`f276c10`, contains `952f0b9`).

1. Eager decode A/B, 4-bit and bf16, receipt-standard model + prompts:
   baseline `origin/main` tip vs `hostcost` tip. Metric: median tok/s
   over >= 30 warm tokens x 3 runs, plus profiler host totals
   (MLX_OMARCHY_GPU_PROFILE) for submissions/token and host-ns/token.
   Keep gate: host-ns/token drops AND tok/s does not regress beyond
   noise; values byte-identical output text.
2. Correctness on hardware: runtime + eq_math + compiled_tape batteries
   (same expected results as above; compiled_tape PER_NODE_SUBMIT red
   expected on both sides - it is pre-existing) + one compiled-tape
   differential (tape vs eager values) since the change touches barrier
   recording. If the differential or any battery regresses vs baseline,
   revert `952f0b9` and record both sides' numbers.

Compiled tapes remain auto-disabled on Apple GPUs; nothing in
`6af35b7` depends on them.

## Wall-time attribution (added same day, Main directive)

Main's arithmetic on the ranked table is correct and now bounds the
lever: steady encoder host work is about 1.5 us/dispatch, about 95
dispatches per token, so roughly 140 us per token against a wall of
order 100 ms - about 0.1 percent of decode wall. The 89 percent
inter-submission gap is NOT encoder CPU. The instrument now attributes
it: `profile_analyze.py` gains a wall-time-attribution section that
splits the stream into encoder-active host spans, explicit
backend-visible blocks (join wait; ring wait inside begin), and
everything outside the encoder (python, evaluator, allocator,
scheduler waits, GPU-completion idle).

llvmpipe fixed-chain bench (no python in the loop), both shapes:

| shape | span | encoder_active | blocked_join | ring_wait | outside_encoder |
|---|---|---|---|---|---|
| dep | 1103.6 ms | 8.4 ms (0.8%) | 0.33 ms | 5.1 ms | 1095.2 ms (99.2%) |
| indep | 1014.9 ms | 5.8 ms (0.6%) | 0.45 ms | 3.6 ms | 1009.0 ms (99.4%) |

Even with python removed, the host thread spends >99 percent of the
span OUTSIDE encoder hooks: in this bench that is the evaluator
blocking on GPU completion inside eval() (llvmpipe is slow); on the
product path the same bucket additionally holds python and the
async_eval tape walk. FragmentationHunt's counts (2682 gpu::eval per
token, 97 submissions, 97 finalize, ~1.4-1.6 ms/token of in-walk
eval_impl+record inside the async_eval window) name what fills that
bucket on hardware; their token markers + this analyzer on an M1
decode stream is the decisive next measurement. Descriptor chunking,
push descriptors, and layout redesign are explicitly ruled out by Main
as inside the same 0.1 percent. `6af35b7` stays provisional pending
the M1 leg below.
