# 2026-09-03 — ring, descriptor pool cache, Event::wait generation join (batching deleted)

Implements the top two host-cost opportunities from
`receipts/2026-09-02-gpu-profile-decode.md` and fixes one latent contract
defect the work exposed. Base: main `959c7a0e14d11cb81b1888ad2215f920ce02a3f0`,
uncommitted diff (Main commits).

**Measurement condition (added 2026-09-03):** all jwm1-linux timings in this document were taken with only 1 of 8 CPU cores online (GRUB entry `Omarchy ANE test`, which supplies a static device tree and left cores 1-7 offline). The machine now boots all 8 cores. GPU-bound figures were materially unaffected (matmul TFLOP/s median 0.1556 -> 0.1576); these host-bound figures (wall shares, tok/s) may improve on re-measurement.

## What shipped

POST-REVIEW CHANGE (Main's question, answered with a build): the profiler
is now COMPILE-TIME gated. Without `-DMLX_OMARCHY_GPU_PROFILING=ON` (the
default) the harness compiles to no-op inline stubs — release builds carry
zero profiling code, verified by `strings libmlx.a | grep -c
MLX_OMARCHY_GPU_PROFILE` = 0 and a 25/25 battery on the default build.
With the flag on, the env var selects the output as before. The dispatch
path in an OFF build contains no profiling branch at all.

Measurement-scope correction (same review): the "record = 88.2% of host
wall" figure came from the llvmpipe run and was BIMODAL — median dispatch
record 1.7 us (the descriptor cache working; the old M1 code measured
160-260 us) with the mean inflated by ms-scale outliers. Those outliers
were the RING'S SLOT WAIT: my record-cost window included
ensure_recording, which contains the wait for a free slot — the
replacement of the old per-record join. The harness now splits them: the
d-line `h` field is pure record (descriptor + barrier + dispatch record)
and the b-line `dur` is slot wait + Begin. On llvmpipe the slot wait
includes waiting for software execution; on Honeykrisp it will show the
true residual stall. Main's question "did the cache not remove what was
expected or was record mostly something else" — answer: the cache removed
exactly the pool churn (median 1.7 us proves it), and the rest was the
join's replacement showing up inside the record window, plus llvmpipe
software execution. The M1 decomposition will separate all three.


- `encoder.{h,cpp}` — 4-slot in-flight command buffer ring: Begin no longer
  host-joins the previous submission (the profile's single largest decode
  cost: 372 ms/token of join wait). The host joins only when all four
  slots are executing, on host reads, or at teardown. Per-dispatch
  descriptor pool create/alloc/destroy replaced by a per-encoder pool
  cache (2048 sets; a retired pool stays alive until the submission
  recording at retirement completes — completion-timeline order, which is
  strictly after every submission that allocated from it). `commit()` is
  an eager flush at op granularity.
- `event.cpp` — GPU-stream `Event::signal` captures the completion-timeline
  generation carrying the signal; `Event::wait()` joins exactly that
  generation (`drained_value_` publishes only after that generation's
  handlers ran). Fixes a LATENT race: wait() previously joined only
  already-drained generations, so a waiter could cross the handler
  boundary before the generation carrying its signal drained — reachable
  before this diff under timings never hit; measured 1/30 with the diff's
  timing, 0/40 on clean main, 0/40 after the fix.
- `eval.cpp` — finalize() forces the flush (the evaluator calls finalize
  at task-throttle points and graph end, then waits on task-completion
  handlers).
- `device.{h,cpp}` — capability field `queue_timestamp_valid_bits` plus the
  four core-1.0 device-table loads (`CreateQueryPool`,
  `GetQueryPoolResults`, `CmdResetQueryPool`, `CmdWriteTimestamp`).
  COMMITTER NOTE: these loads had only ever existed as uncommitted edits;
  without them the env-gated profiler segfaults on a null table call.
- `gpu_profiler.h` — per-ring-slot profiler state, per-entry tick indices,
  host timestamps on begin/submit lines.
- `test_runtime.cpp` — two contract tests updated to the eager-flush +
  ring encoder; the drain-serialization test now exercises its actual
  target (the CompletionDispatcher) through plain commit().
- `scripts/build-wheel.sh` — `CMAKE_BUILD_PARALLEL_LEVEL` was hardcoded 16:
  on the single-core M1 that fanned out to ~96 compiler processes on one
  CPU and flirted with the OOM killer. Pinned to
  `${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}`.

## Batching: implemented, measured, DELETED (negative result on record)

Deep batching (16-100 ops per submission) was implemented first and made
generation 4.5x SLOWER on llvmpipe (0.637 vs 2.838 prompt tok/s) and
slower on the M1. Mechanism: MLX evaluates per primitive; the evaluator
throttles at `MAX_ACTIVE_TASKS = 10` (transforms.cpp:25) by calling
`gpu::finalize()` + `wait_for_one()` — the host blocks until one in-flight
task completes. Per-op flushes make that wait span one op (pipeline stays
full). A batched flush makes it span the WHOLE batch: host records N ops
(microseconds), submits, then sleeps for the entire batch execution.
Profiled budget-16 run on llvmpipe: 1,210 inter-submission gaps averaging
57 ms (p50 88 ms) = 68.6 s of an 84 s span; GPU busy 16.7% in a run that
should have been GPU-bound. The regression scales with batch depth and
model runtime — deeper batches and longer models are worse, never better.
The correct future approach is flushing when the scheduler would actually
block rather than on a node budget — upstream-shape work. The batching
code is deleted from the diff; git history and this receipt carry the
analysis.

## Where the wall time sits now (llvmpipe, measured; M1 pending)

Host-timeline decomposition of a profiled mlx-lm prefill (41 tokens,
5,327 dispatches, 7,409 submissions — op-granular):

| component | total | share of host wall |
|---|---|---|
| dispatch record (host) | 15.68 s | 88.2% |
| submit() | 0.06 s | 0.3% |
| join wait+invalidate | 0.03 s | 0.2% |
| begin() | 0.02 s | 0.1% |
| OTHER (evaluator throttle waits, python) | 2.00 s | 11.2% |

The three encoder fixes reduced submit+join+begin to a combined 0.6% of
host wall on this stack. The remaining wall is llvmpipe's own per-dispatch
software cost (~2.7 ms/dispatch, present identically in the pre-diff
wheel) plus ~11% evaluator/python. On Honeykrisp the same decomposition is
the first measurement to take when the box reopens: the pre-diff M1
profile had record ~160-260 us and join ~160-180 us per dispatch; the ring
and cache remove the join and most of the record, so the M1 residual will
show whether Honeykrisp's per-dispatch cost or the evaluator throttle is
the next ceiling. llvmpipe end-to-end (same box, same session, env off):
old wheel 3.072 prompt tok/s, new wheel 3.173 (+3.3% — neutral, as
expected where the driver's own software cost dominates).

## Validation (all observed this session)

- llvmpipe dev box, `959c7a0` + diff, CPU=ON configure,
  `MLX_OMARCHY_ALLOW_NON_APPLE=1`: full battery 25/25 binaries green
  (including the four suites fixed on main and the rewritten contract
  tests). `omarchy_fast_ops_tests` 50/50 (composition with the
  semaphore-lifetime fix). `omarchy_runtime_tests` stress 40/40 after the
  Event fix (1/30 before it).
- jwm1, `959c7a0` + an earlier revision of this diff: battery 23/24 — the
  one red is `omarchy_indexing_ops_tests` "scatter keeps named rejections"
  (test_indexing_ops.cpp:458), a stale refusal pin: `959c7a0`'s CAS
  fallback now serves `scatter_add` where the pin still expects the
  `VK_EXT_shader_atomic_float` refusal. Fails identically on clean
  `959c7a0` — predates this diff; 959c7a0 authors own the pin.
- Final tree (batching deleted, host timestamps added): llvmpipe battery
  25/25 re-verified after the deletion.

## Gate status

**M1 before/after tok/s for the shipped design: UNOBSERVED.** The first M1
A/B ran on a polluted wheel (pip's persistent build directory mixed stale
objects with new sources: the installed libmlx.so had zero
`vkCmdWriteTimestamp` strings while carrying the new encoder — its
0.704/0.103 t/s "regression" is void; every wheel number from a build dir
that predated the purge is void). jwm1 is under exclusive lock for
M1MergedVerify; the clean-wheel A/B (both wheels already built and staged
in venvs) is the first queued task when it reopens. llvmpipe A/B above is
same-session and clean; llvmpipe is not the receipt target.

## Reproduction

Dev box: `/home/joshuawarren/src/mlx-omarchy-batch2` (detached at 959c7a0 +
diff), venvs `.work/venv16` (new wheel) and `.work/venv-old` (baseline
wheel), model snapshots under `~/.cache/huggingface/hub`. jwm1:
`~/src/mlx-omarchy-batch2` (same tree, tests built, battery logs
`/tmp/fin-*.log`), wheel build stopped at Main's box lock. Profiles:
`/tmp/nobatch.jsonl` (host-timestamped), `/tmp/slow-final.jsonl` (budget-16
evidence). No commits. Main commits.
