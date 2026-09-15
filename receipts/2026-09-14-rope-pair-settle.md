# 2026-09-14/15 Rope-pair settle on jw16 — NO-LAND

## Verdict

**Named no-land.** The land rule (dispatches drop AND pins hold AND
ctx1053 median tok/s rises) is not met: no stable candidate build
existed to measure. The rope-pair layout gate named by
`receipts/2026-09-14-decode-standalone-epilogue.md` is fixed
(519d7336), and the pairs now pass every plan and runtime contract
check — but the pair fires at the first member while that member's own
producer chain is still **unscheduled**, so the dispatch cannot bind
its input. A pre-settle experiment (nested `eval` of the unsettled
chains) makes the prefill's 23 pairs fire cleanly, then crashes on the
tape re-ordering interactions it causes (SEGV, reproducible,
eval.cpp:65). Root cause chain fully diagnosed; the remaining fix is
evaluator-level work named below.

## What landed on the branch

- **519d7336** — `dispatch_rope_pair` resolves each side's input layout
  independently (`q_transposed`/`k_transposed`, flags bits 4/8) and
  accepts the transposed slice of a wider row-contiguous producer
  buffer: per-head stride D, per-time stride W*D, batch stride T*W*D
  (feature stride 1). The kernel already carried per-side stride slots;
  only the contract and the flags word changed. This removes the
  `input layout unsupported` refusal measured by 0a2370ba
  (q row-contiguous strides 3712,1856,64,1; k strides 25984,64,896,1 —
  which is the already-legal whole-buffer transposed form at
  n=14; the cross-side layout equality check was the blocker).
- **8f1ed713** — EXPERIMENTAL, labeled do-not-ship: pre-settle of
  unsettled input chains via nested `eval` at the pair hook; tape skip
  for pre-evaluated nodes (`patches/mlx-rope-settle-tape.patch`,
  wired into `scripts/prepare-mlx.sh`); skip/trace guard in
  `overlay/.../eval.cpp`.

`qmm_vec.comp` untouched (no diff at every point in this run);
`63c1d3cf` not merged; every GPU run under one `flock` hold on
`/tmp/m1-gpu.lock` (inode-stable, never unlinked, nested `flock -n`
refused).

## The hole, measured to the ground

With the layout gate open, the first pair hook SEGVs in
`dispatch_rope_pair` at `binding()` on the first member's own input.
Instrumented state at the hook (jw16, Qwen2.5-0.5B-Instruct-4bit,
`MLX_OMARCHY_TRIO_TRACE=1`):

- Pair fires at the first member = the keys rope (2 heads, input
  [1,2,29,64] row-contiguous at T=29). Partner = the query rope
  ([1,14,29,64], whole-buffer transposed, **settled**).
- The keys rope's own producer chain (Transpose ← Reshape ← Add ←
  QuantizedMatmul ← RMSNorm) is **entirely `Status::unscheduled`** at
  hook time: `bfs_max_width` (default 20, `MLX_BFS_MAX_WIDTH`)
  segments the tape via the deferred-dfs stack and defers those
  producers past the rope's pop. Raising the width to 100000 did not
  change the outcome, so the deferral is not width-limited alone; the
  ordering fact stands regardless of mechanism.
- The single-rope path never sees this because the pair hook is the
  only consumer that binds the partner's input early; the ordinary
  path's layout fallback and its later eval position keep it safe.

## The pre-settle experiment and why it crashes

`eval(chains)` at the pair hook schedules the unsettled chains with the
same kernels on the same stream (dispatch-count neutral by
construction). The prefill's 23 pairs then fire with **zero** contract
refusals. But the nested eval evaluates and **detaches** planned rope
members that have not popped yet (transformer residual/cache chains
make earlier ropes ancestors of later chains), so:

1. the outer loop later pops a detached member whose inputs are empty →
   the pair refuses `rope state mismatch` (`k_node.inputs().size()==0`),
2. `return false` sends that member down the ordinary path, which calls
   `arr.primitive().eval_gpu` on the detached node → SEGV at
   `overlay/.../eval.cpp:65` (gdb, RelWithDebInfo build, exact line).

Interim guards (tape skip of `Status::evaluated` nodes; `has_primitive`
guard in `gpu::eval`) did not close the gap: the crashing node has a
live primitive and unscheduled inputs — it is a *pair member popped
before its producer chain*, which the skip cannot distinguish without
pair-state in the evaluator.

## Remaining fix (evaluator-level, next task)

Either (a) make nested pre-settle evals non-detaching for nodes owned
by an outer tape (evaluator flag), or (b) let the pair defer to the
second member without un-planning, keeping the first member's output
written by its ordinary dispatch and having the pair recompute only the
second (savings halve to ~11 dispatches), or (c) plan-level
pre-settlement in the GEMV-group pass that materializes planned pair
input chains before the loop starts. (a) is the only variant that keeps
the full 23-dispatch win.

## Measurements

- Dispatch probes (dispatch_count.py, 8 tokens, MLX_DISABLE_COMPILE=1,
  HF_HUB_OFFLINE=1, snapshot a5339a4131f1…): with the settle-guard
  wheel (pairs refuse, ordinary paths) = baseline behaviour, no
  crash, no dispatch change measured (probes aborted at probe 1 in the
  09:11 run; the diag runs completed rc=0 at 249-equivalent
  behaviour). No stable firing build → no dispatch-drop measurement,
  no A/B, no ctx1053/ctx1024 tok/s, pins not re-verified on a firing
  build.
- Probes and evidence: `/tmp/presettle-probe.log`, `/tmp/diag3.log`,
  `/tmp/diag5.log`, `/tmp/trace-probe.log`, `/tmp/width-probe.log`,
  `/var/tmp/DecodeTrioRun/jw16-out-pairsettle/` (aborted run),
  `/tmp/diag-probe.log` (gdb transcripts in session log).

## Wheels (full sha256, all stamped +519d7336 from a dirty tree — the
stamp reflects HEAD, the hashes are the identity)

- 0907 clean per-side-layout (crash build):
  `b7a5df392d1982167996cec8bb8a9a0bc8e6079df0462623b624f0eaaf758864`
- 0933 settle-guard + chain dump:
  `b4aa03cf6be01006fd7d24fe84f73d5b1ee897273e7556eaf168dd8a0f69479b`
- 1030 tape-skip: `93a1d57afe9719fc4d46313a98f55e32ac271b76128cc02755cb809a9ffe164c`
- 1039 RelWithDebInfo: `1d737c3d4fe21ed357f553707b38d292edc8f49697a04d8989e82ee1f8bf02f1`
- 1056 final trace build (branch tip 8f1ed713 content):
  `5c5a27cc9aa0ba23f664f4ef2e88f938de7a655c30f3d62494cd82871e0c8850`

`/var/tmp/DecodeTrioRun/venv-trio` currently holds the 1056 trace
build (a copied-venv pip shebang quirk wrote early diag wheels there);
reinstall the committed-tree wheel before any future A/B.

## Decision

**No repo state on mlx-omarchy main changed; nothing merged.** The
branch `agent/decode-standalone-epilogue` carries 519d7336 (the layout
fix — reusable, pins untested against a firing build) and 8f1ed713
(labeled experimental). The receipt lives on the branch per the
epilogue precedent.
