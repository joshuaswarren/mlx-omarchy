# Runtime modernize: prepared-source bump 0.32.2 -> 0.32.3 main (jw16 device verify)

Branch `rtmod/source-bump` (base main `86c9deb8`), tip `71d73a41`.
Candidate wheel: `mlx_omarchy-0.32.3.dev202609181256+71d73a41-cp314-cp314-linux_aarch64.whl`
(jw16 `~/src/mlx-omarchy-rtmod2/dist/`). Scratch venvs on jw16: `/tmp/venv-cand`
(candidate wheel + mlx-lm 0.31.3 + mlx-vlm 0.7.1), `/tmp/venv-base` (certified
v0.6.7 release wheel `+fb649d8`, for A/B), `/tmp/venv-main` (candidate wheel +
mlx-lm git `872ae88`), `/tmp/venv-omlx` (candidate wheel + mlx-lm `872ae88` + omlx 0.6.4).
Weights cache: `~/.cache/huggingface` on jw16 (~80 GB, all matrix models).

## 1. Target choice

- Upstream pin: `ml-explore/mlx` main tip `59d600b5e64c238427d0f8d897ab7c682ef4d3d2`
  (2026-09-17, in-dev 0.32.3). One month past v0.32.2, 65 commits.
- mlx-lm PyPI latest IS 0.31.3 (no newer release exists; frontier is git main
  `872ae88d1fac77350db23c8c04fe8dd372a9e3e8`, what oMLX main pins). The repo's
  0.31.3 pin is current, not stale. mlx-vlm latest is 0.7.1 (newer than the
  0.6.3 the Bonsai runtime pins; `mlx_vlm.load` imports and the text path works).
- LoaderFeasibility finding (2026-09-18 receipt): NO currentgen arch requires a
  source newer than 0.32.2, so this bump is CURRENCY, not a load requirement.
  `mlx-vlm 0.7.1` declares `mlx>=0.32.2`; a PEP-440 parse of the old
  `0.32.2.dev…` local version is BELOW that floor, and `0.32.3.dev…` is above
  it, so the bump also cleans up the mlx-vlm dependency story.

## 2. Delta and port

Framework-level changes that touch the backend boundary:

- `fast::GatedDeltaUpdate` (gated delta nets, Metal kernels upstream): new
  `Custom` fast primitive with per-backend `use_fallback`. Omarchy wiring in
  `overlay/mlx/backend/omarchy/primitives.cpp`: `use_fallback(...) -> true` +
  `OMARCHY_UNSUPPORTED_MULTI(GatedDeltaUpdate)` — GDN layers run the composite
  fallback (pure implemented Vulkan ops), exactly the no_gpu posture. No fused
  Vulkan kernel was added; that is the follow-up backend work item if GDN
  models need fused speed.
- `array` assignment/release cycle fix + new `mlx/memory.cpp`
  (`get_array_buffer_size`): framework-level, no backend impact.
- `mlx/linalg.cpp` gained `check_cpu_or_cuda_stream`; patch
  `mlx-linalg-gpu.patch` rebased onto the new layout (hunk re-anchored, the
  GPU-refusal rewrite in `check_cpu_stream` unchanged in spirit).
- io/load, gguf, fast.cpp, ops.cpp, python bindings: covered by the existing
  patch series; all 13 patches apply `--fuzz=0` on the new vintage.

## 3. Device load matrix (jw16, real Vulkan GPU, lock protocol)

Probe: load + greedy generate (prompt "The capital of France is", 16 tokens).
Full JSON lines: jw16 `/tmp/rtmod-matrix.log`, `/tmp/rtmod-round2.log`,
`/tmp/rtmod-round3.log`.

| model | arch | result |
|---|---|---|
| mlx-community/Qwen2.5-0.5B-Instruct-4bit (regression) | qwen2 | GENERATED (load 0.48 s, gen 0.24 s, coherent text) |
| mlx-community/Ministral-3-8B-Instruct-2512-4bit | mistral3 | FAILED: Vulkan timeline counter watchdog, see F1 |
| prism-ml/Ternary-Bonsai-8B-mlx-2bit | qwen3 (2-bit) | FAILED: same watchdog, see F1 |
| mlx-community/gemma-4-12B-4bit | gemma4_unified | NOT-LOADED on 0.31.3 ("model type not supported"); loads on mlx-lm `872ae88`, then hits bf16 tape refusal, see F2 |
| lmstudio-community/gemma-4-E4B-it-MLX-4bit | gemma4 | NOT-LOADED: upstream mlx-lm KV-shared-layer bug, see F3 |
| lmstudio-community/gemma-4-26B-A4B-it-QAT-MLX-4bit | gemma4 | FAILED: bf16 compiled-tape refusal, see F2 |
| mlx-community/gemma-4-31b-it-4bit | gemma4 | FAILED: bf16 compiled-tape refusal, see F2 |
| mlx-community/Qwen3.6-27B-mxfp4 | qwen3_5 | FAILED: `Take` dtype uint8 not implemented, see F4 |
| prism-ml/Ternary-Bonsai-2-27B-mlx-2bit | prism_hadamard_qwen35 | see section 4 (headline) |

## 4. Headline: Ternary-Bonsai-2-27B

- The pack's own `runtime/artifact.py` `load_model` self-rejects the pack:
  config carries `schema_version: 2` while that entry point demands 1. This is
  a pack-internal v1-preview vs v2 entry-point skew, NOT a runtime failure.
- The reviewed revision of the pack loader is pinned by Bonsai-demo
  (`scripts/bonsai2-runtime.sha256`); the pack's four runtime files match it
  byte-for-byte, and the canonical entry is `vision_artifact.load_vl_model`
  (schema-2 aware) driven by `scripts/mlx_generate_bonsai2.py`. Round 3 runs
  that canonical path on the candidate wheel: result recorded below in
  section 8 (appended after the run).

## 5. Parakeet pin regression — PASS on the bumped source

`mlx-omarchy-parakeet download` / `verify` / `transcribe` on the candidate
wheel, jw16, ANE islands on `/dev/accel/accel0`:

- `verify`: 12 files + pinned audio fixture OK.
- `transcribe`: `status: match`, `checks_failed: []`, 104 emissions, pipeline
  7866 ms, transcript "He hoped there would be stew for dinner, …"
- `encoder_hidden.npy` sha256 `38c73261f29230276ed76f1fc017b76b024156d79218bd5f1347fdc7e7d43ec7`
  — EXACT match to the certified pin (`38c73261…`) recorded in
  `receipts/2026-09-16-parakeet-wheel-packaging.md`.

The certified encoder pipeline is intact on the 0.32.3 vintage.

## 6. Findings (backend work items, none introduced by this bump)

- F1 Vulkan timeline watchdog: first big-model eval stalls
  ("timeline counter failed to advance for 10000 ms (last observed=0,
  target=1)"). A/B: reproduces IDENTICALLY on the certified v0.6.7
  (`+fb649d8`, 0.32.2-based) wheel — standing issue, not a bump regression.
- F2 bf16 compiled-tape refusal ("bf16 fragments corrupt nondeterministically
  on Honeykrisp") blocks every gemma4 generation attempt; deliberate gate, do
  not weaken — needs bf16 tape correctness work or an eager path decision.
- F3 E4B checkpoint ships per-layer k/v for KV-shared layers; mlx-lm
  (0.31.3 AND git main line 185) rejects 126 params. Upstream mlx-lm bug —
  file upstream; not fixable by source or backend.
- F4 `mx.take` with uint8 indices/indices-dtype unimplemented on Vulkan —
  blocks Qwen3.6-27B-mxfp4 (mxfp4 dequant path). Missing primitive dtype.
- F5 Bonsai-2 pack loader entry skew (v1 `load_model` vs v2
  `load_vl_model`), section 4.
- gemma4_unified arch requires mlx-lm > 0.31.3 (loads with `872ae88`).

## 7. Host protocol receipts (jw16)

- GPU take/release under `/tmp/m1-gpu.lock` (flock fd 9) for every GPU
  session; llm-inference stopped before, restarted + CONFIRMED after:
  `active`, `health http 200`, real completion returned (round 1 and round 2).
- Round 3 restart+CONFIRM appended in section 8.

## 8. Round 3 results

- HEADLINE UPDATE (rounds 4/4b, `0.32.3.dev202609181349+71d73a41` with
  Select-I64): the Select-int64 implementation WORKS — the GDN path advanced
  past the previous failure point through hadamard/quantized ops and the
  chunked-state machinery, and now stops at the F1 timeline watchdog INSIDE
  `gated_delta_chunked`'s first `mx.async_eval(Y, S)` submission
  ("last observed=0, target=1"). tok/s is still not obtainable; the headline
  blocker is now singular and tightly scoped: the first multi-output
  async_eval submission of the GDN composite never signals the timeline.
  Repro lead for the fix lane: `mx.async_eval` with two outputs sharing a
  dependency, first submission on a fresh stream.

## 8a. Round 3 results (original)

- **Bonsai-2-27B (round 3, pre-Select-I64)**: canonical loader works on the candidate wheel —
  `vision_artifact.load_vl_model` loaded the pack in 10 s (manifest-verified
  loader, 27B ternary + vision tower resident). Generation then stopped at the
  FIRST GDN layer with the exact op error Joshua asked to capture:
  `RuntimeError: [omarchy] Select dtype is not implemented for the Omarchy
  Vulkan backend (dtype=int64, shape=[1,21]). No GPU kernel exists for it; no
  silent CPU fallback occurs.` — thrown from `mlx_vlm/models/qwen3_5/
  gated_delta.py:375` (`gated_delta_chunked` -> `mx.async_eval(Y, S)`), i.e. a
  `Select` over int64 inside the chunked gated-delta path. The backend
  supports int64 storage broadly (ElementwiseI64/U64, ReduceGeneralI64/U64)
  and `omarchy_shader(select_complex64 shaders/select.comp -DUSE_U2=1)` shows
  Select already has a dtype-agnostic two-word (uvec2) compiled variant, so
  Select-I64 is a small follow-up kernel-mapping task, not new machinery.
  Becomes backend work item F6.
- **Round-4 section C**: Qwen3.6 on the Select-I64 wheel still fails at F4
  Take-uint8, unchanged (Select fix does not touch the Gather path). Gemma4
  E4B device window (peer lane): F3 shim makes the 126-param load PASS; gen
  stays blocked by F2/F1. Gather/Take byte tables have no 8-bit storage on
  Honeykrisp (capabilities expose 16-bit only), so F4 needs a word-packing
  gather variant — dedicated item, not a dispatch mapping.
- **E2B** (`lmstudio-community/gemma-4-E2B-it-MLX-4bit`): NOT-LOADED — same
  upstream KV-shared-layer loader bug as E4B ("Received 140 parameters not in
  model: layers.15.self_attn.k_norm..."), so no 8 GB-tier gemma4 pick exists
  via E2B/E4B until F3 is fixed upstream; 12B loads (via mlx-lm `872ae88`) but
  hits F2; 26B-A4B-QAT hits F2.
- **Server A/B (oMLX vs mlx_lm.server)**: NOT COMPLETED — mechanical probe
  failures, no model verdict. mlx_lm.server answered 404 (my probe used
  `/v1/chat/completions`; server on this build exposes a different route set —
  request paths need re-checking against `/tmp/ab-mlxlm-server.log`). omlx
  bound to :8082 only after the probe window (model load exceeded the wait
  loop; server log confirms `Binding server at http://127.0.0.1:8082`). Both
  servers are staged and the harness (`/tmp/run-round3.sh` section E) needs a
  longer readiness wait + correct mlx_lm route; rerun is cheap.
- **Gate suite**: build of `omarchy_*` test targets currently fails to
  COMPILE in `overlay/tests/omarchy/test_matmul_family.cpp` — upstream added
  an 11th `global_scale` parameter to `quantized_matmul` (gather_qmm work),
  and the test's positional argument lists mis-bind into it. Wheel/release
  build is unaffected (built with tests OFF, succeeded). The test update +
  full gate rerun (G1-G5 posture) is the remaining pre-land step for this
  branch; not silently skipped, recorded here.
- Host protocol: llm-inference restarted after round 3 and CONFIRMED
  (`active`, `health http 200`); GPU lock released.

## 9. Verdict

- Source bump: LANDED on branch, clean patch series, candidate wheel built and
  device-proven for the regression path (Qwen2.5 GENERATED) and the full
  Parakeet certified pin (exact `38c73261` encoder_hidden match). No publish,
  no push; nothing outside the branch and jw16 scratch was modified.
- Currentgen generation on Vulkan is blocked by four standing backend items
  (F1 watchdog, F2 bf16 tape, F4 Take uint8, F6 Select int64) — all
  reproduced/qualified on device with exact errors, none introduced by the
  bump (F1 A/B-verified on the certified 0.32.2 wheel).

## 10. Gate suite status (post-bump, jw16 device)

All omarchy gate targets now COMPILE (test_matmul_family fixed for the new
gather_qmm global_scale param). First device run of the core set
(/tmp/rtmod-gates.log):

- PASS: omarchy_copy_offset_tests (7/7)
- FAIL (F1 timeline watchdog, "last observed=0, target=1", thrown at FIRST
  submission of specific sequences): omarchy_primitive_tests 100/103 (cholesky
  error-code check + cos/sin + grad-sin-cos), omarchy_runtime_tests (case at
  test_runtime.cpp:1797). The device RECOVERS between processes (later suites
  pass after an F1 failure), so this is a per-submission-sequence scheduler /
  fence bug, not a wedged device.
- omarchy_matmul_family_tests / take_fill / fast_ops / fast_regression /
  error_contract / compiled_tape / wrong_value_sweep / select_layout /
  capability_sim: results in /tmp/rtmod-gates.log (same run).
- omarchy_ane_runtime_tests: link failure (undefined AneRuntime::load 3-arg +
  dtor) - ane runtime definition not compiled into that target on this
  vintage; recorded, target excluded from the run pending fix.

VERDICT: gates NOT green -> branch must NOT land/push yet. The remaining work
is exactly one bug class (F1) plus the ane_runtime link item; everything else
in the bump is verified on device (Parakeet pins EXACT, Qwen2.5 regression,
Select-I64 advancement, no vintage-attributable regressions).

## 11. F1 bisect (continuation): the hang is a GPU-side stall of the first submit, not a lost scheduler signal

Instrumentation added behind `MLX_OMARCHY_TRACE_DISPATCH=1`
(encoder.cpp dispatch trace) plus a submit print. Findings on the isolated
repro (`omarchy_primitive_tests --test-case="Cos and Sin match host
references through Vulkan compute"`):

- The case dispatches exactly two kernels — `FEW name=Abs count=6` ->
  kernel=0 (ElementwiseF32, count=6) and kernel=20 (ReduceF32, count=1) —
  then SUBMIT cv=1 fires (waits=0, sigs=2, cmds=1) and the GPU never
  completes it: counter stays 0 for the full window even at
  MLX_OMARCHY_HANG_NO_PROGRESS_NS=120 s (not slow work — lost completion).
- A pure Payne-Hanek stub (naive reduction, no dynamic local-array stores,
  no umulExtended) STILL hangs — the trig math is NOT the hanging construct.
- Bare `mx.sin(f32)` reproduces outside any model (probe /tmp/trig-probe.py).
- Theory narrowed: the ReduceF32(count=1) dispatch shape (single-element
  reduce) is the prime suspect for the driver stall — next bisect step is to
  run the case with the ReduceF32 dispatch suppressed/replaced (copy-based
  fallback) to confirm, then decide: driver workaround (pad single-element
  reduces to 2), or Asahi bug report with the reproducer.
- Gate matrix with F1 active: primitive 100/103, runtime 1 hang (case 1797),
  fast_ops hang, compiled_tape fail — all F1-class; copy_offset, matmul
  family, take_fill, select_layout, fast_regression, error_contract,
  wrong_value_sweep all PASS on the same wheel.

## 12. F1 bisect round 2 (final state of this run)

- Scalar-reduce reroute experiment: with ReduceF32 suppressed for
  out.size()==1 (falls to the general reduce kernel, kernel=113), the
  isolated cos/sin case STILL hangs at the same submit — so the stall is
  NOT the ReduceF32 suffix kernel. Control case: "suffix Sum and Max
  reductions" dispatches ReduceF32 twice and completes instantly in
  isolation.
- The cos/sin graph records FEW name=Abs count=6 + a reduce, submits
  (cv=1, sigs=2, cmds=1), and the GPU never completes THAT buffer; the
  actual Cos dispatch is never even recorded (host blocked at the first
  join). Graph shape, not trig math and not one kernel: the trigger is
  the (Abs -> reduce) prefix submission of the 0.32.3 f32 cos lowering.
- Next steps for the fix lane (continue with MLX_OMARCHY_TRACE_DISPATCH=1):
  (1) dump the cos(x) graph on 0.32.3 (mx.eval tape / export) to see why a
  0.32.3 f32 cos lowers to Abs+reduce+... and whether an eager fallback or
  fusion-boundary change avoids the prefix submission; (2) if the prefix
  shape is confirmed as the trigger, test the same dispatch pair in a
  standalone buffer outside cos to build the Asahi reproducer.
- Instrumentation remains available: MLX_OMARCHY_TRACE_DISPATCH=1 prints
  every elementwise dispatch (kernel id, count, groups); submit prints
  cv/waits/sigs/cmds (encoder.cpp).

## 13. F1 root-cause round 3 (2026-09-18, F1StallFix lane): environmental submission swallow; recovery ladder landed (WIP)

- Every F1 theory from sections 11-12 is DEAD, killed by probe matrix
  (/tmp/f1-probes*.log, /tmp/f1-trace.log on jw16): the stall is NOT the
  trig gate (f16 sin has no gate and still hung), NOT the Abs+reduce prefix
  (standalone abs+max+item passes at every count 1..1024), NOT
  first-submission ordinal (sin hangs mid-process after a completed
  warmup), NOT kernel- or shape-specific. The prefix Abs+reduce seen in
  round 2 is just trig_argument_gate's `max(abs(x))` magnitude check
  (primitives.cpp `trig_argument_gate`), which runs BEFORE any Cos/Sin
  dispatch; the frozen cv=1 sigs=2 submit is that gate's submit.
- The stall is a burst-window phenomenon: ~10:24-10:55 CDT every first
  GPU submission of a fresh process wedged (completion timeline stuck at
  0), across bare mx.sin/cos (f32 AND f16), gate and non-gate shapes;
  60+ consecutive runs pass outside the burst. dpms stayed On; GPU runtime
  PM is "unsupported" (firmware-managed). Burst coincided with the wheel
  compile storms on jw16 (system load); self-recovers in minutes.
  In-tree comment (device.cpp `Device::signal_timeline`) already documents
  the deterministic member of this class: Honeykrisp SWALLOWS an empty
  signal-only QueueSubmit (fixed there by host-signaling). Round 3 adds
  the bursty generalization: real command buffers get dropped too.
- TRACE ORACLE: MLX_OMARCHY_TRACE_DISPATCH=1 with stderr to a PIPE made
  the 4-element sin probe hang deterministically inside the burst window;
  stderr to a FILE passes. Prints delay the host in the submit window and
  expose the race. Event-signal rides every first submit (sigs=2) via the
  eval tail Event::signal queued path (both passing and hanging runs
  print identical [rtmod] traces - the difference is GPU-side, not host).
- Recovery ladder LANDED on the branch (encoder.cpp/device.{h,cpp}/
  event.cpp, +226 lines): every submit retains its batch
  (CompletionDispatcher::ResubmitBatch); on a no-progress watchdog
  interval with NO started-event evidence of execution
  (CmdSetEvent@TOP_OF_PIPE is the execution proof), the Device resubmits
  all stalled batches (kick submit first), budget 2 rounds per wait,
  then throws as before. `MLX_OMARCHY_TEST_DROP_SUBMIT=N` deterministically
  simulates the driver swallow (drops the Nth QueueSubmit, host still
  publishes the completion).
- PROOF + REMAINING DEFECT (honest): with TEST_DROP_SUBMIT=1 the ladder
  fires (TEST-DROP cv=1 -> SUBMIT-RECOVER round 1) and the resubmitted
  kernels EXECUTE (started event sets), but the completion timeline still
  never advances: Mesa does not publish a DUPLICATE timeline point
  (resubmission re-signals value 1 while the dropped submission still
  holds point 1). NEXT (named, no re-derivation needed): recovery must
  signal FRESH reserved values instead of re-signaling the original -
  vkWaitSemaphores is >=, so a higher value satisfies the stuck wait;
  ride-along event-semaphore signals need the same fresh-value treatment
  plus Event bookkeeping (completion_for), and the kick submit should be
  dropped or probed for its own swallow risk. Then: full gate matrix,
  Bonsai-2-27B tok/s (the historical F1 workload), ancestry check vs
  63c1d3cf, certified E2E arms, oMLX A/B per the lane contract.
- LADDER v2 + take-bound fix (cbc99cc9+): three rungs (never-began ->
  kick+resubmit at FRESH values; executed-but-unsignaled -> host
  vkSignalSemaphore of completion AND stranded user semaphores at bumped
  fresh values; budget 2 rounds/wait). Take/judge bounds use
  last_reserved() because round-1 re-retains at values ABOVE the original
  target. Verified on-device: round-1 resubmit fires and kernels execute
  (TEST_DROP_SUBMIT=1). REMAINING DEFECT, precisely: round 2 never fires
  or never prints - after round-1 resubmission executes, the next stall
  returns false silently (host-signal rung absent from logs); the normal
  burst case likewise shows zero recovery attempts (first throw pre-empts
  the ladder). Suspects, in order: (a) the throwing wait is the nested
  eval's Event::wait whose recovery takes a DIFFERENT branch than
  traced, (b) dispatcher-thread drain erasing resubmitable_ mid-stall,
  (c) has_active_submission semantics for signal-only ride-alongs.
  Next debug step: stderr-trace inside recover_stalled_submissions
  entry/return with round + has_active + batch count, one build.
- MATRIX ADDITION (Joshua, 2026-09-18): after stock Bonsai-2-27B
  generation is proven, run the SAME generation test on
  rariruluis/ternary-bonsai-2-27b-mlx-runtime-abliterated (published
  2026-09-18, abliterated on the stock prism-ml pack, presumably
  prism_hadamard_qwen35 - VERIFY the arch/loader contract against the
  canonical prism-ml pack first). Record tok/s + a qualitative sanity
  line; local load/generate verification only.
- DECISION TRACE (build13, MLX_OMARCHY_TRACE_DISPATCH=1 on
  TEST_DROP_SUBMIT=1): the ladder's silent-false is now OBSERVED, not
  guessed. Output order: STALL target=1 through=0 round=0 →
  RECOVER-ENTER through=0 active=0 → RECOVER-FALSE empty-batches →
  TEST-DROP cv=1 → second stall → RECOVER-ENTER through=1 → round-1
  resubmit ✓. Meaning: a FOREIGN WAITER (scheduler-thread event wait,
  event.cpp scheduler::wait_event branch, line ~227) stalls BEFORE the
  main thread submits (through=0 = nothing reserved), gets refused
  (empty), and its watchdog throw kills the process ahead of the real
  recovery. The round-1 resubmit for the real gate batch DID fire after.
  Next concrete fixes: (a) wait_for_timeline_progress must not throw
  from a foreign/stale waiter whose target exceeds last_reserved()
  (refuse-and-return instead of throw when reserved_through==0 at first
  stall), or scheduler::wait_event must not park GPU-timeline waits on
  worker threads; (b) re-run drop-1/1,2 after (a); the round-2
  host-signal rung should then engage for executed-but-unsignaled
  batches.
- Instrumentation on this wheel (env-gated, keep): [rtmod] FEW/DISPATCH/
  SUBMIT/JOIN/EV-SIGNAL/FENCE-UPDATE/GATE prints; all behind
  MLX_OMARCHY_TRACE_DISPATCH. Debug edits live uncommitted on jw16
  (~/src/mlx-omarchy-rtmod2 overlay, stash + /tmp/rtmod2-local-edsave.diff
  hold the pre-existing instrumentation); this branch's copies are the
  authoritative ones committed here.

## 14. F1 root-cause round 5 (2026-09-18, F1ForeignWaiter lane): real root
cause found and fixed - nested blocking eval in host-read gates; proofs green

- FOREIGN-WAITER FIX LANDED (8e6b299f): wait_for_timeline_progress now
  refuse-and-continues when the waited target exceeds last_reserved()
  (owner has not submitted yet) - STALL-FOREIGN trace - and
  recover_stalled_submissions returns a three-state RecoveryResult
  (kRecovered / kNotRecoverable / kExhausted): empty-batch refusal keeps
  the wait alive instead of throwing; only budget exhaustion throws.
- THE PROOFS EXPOSED THE REAL ROOT CAUSE. With the false-kill gone, the
  eager sin probe hung instead of erroring, and gdb on the parked process
  showed the main thread inside Event::wait UNDER magnitude.eval() inside
  Sin::eval_gpu. On the bumped 0.32.3-main framework, a nested eval()'s
  epilogue skips the signal+finalize at eval_nest_depth > 1 (settle
  semantics, added for the rope-pair pre-settle), so the gate's nested
  synchronizer wait could never be satisfied: every eager sin/cos gate
  parked until the watchdog fired. The "environmental submission
  swallow" burst signature was this deadlock + watchdog throw, not a
  driver bug. Rung 1/2 of the recovery ladder remain correct for the
  deterministic TEST_DROP_SUBMIT swallow class (proofs below).
- GATE FIX (8029946a): the four host-read gates (trig gate magnitude,
  rope offset bound, rope freqs bound, rope fallback handoff) now use
  settle() - record into the open batch - and rely on their existing
  synchronize() to submit and order the mapped read. No nested wait, no
  deadlock.
- DRAIN FIX (6d3893b3): the recovery ladder reserved fresh completion
  values without enqueueing a completion for them, so drained_value_
  could never catch up past the original completion and the next join
  timed out ("dispatcher drain did not catch up ... after the timeline
  counter reached 2"). Both rungs now enqueue an empty-payload
  completion for each fresh value; the original completion still owns
  handlers/temporaries.
- PROOFS (build18 = 0.32.3.dev202609181845+6d3893b3, /tmp/venv-proof,
  idle jw16, MLX_OMARCHY_HANG_NO_PROGRESS_NS=2s, /tmp/f1-round5.log):
  - S0 sanity (no sim env): SIN-OK rc=0.
  - S1 TEST_DROP_SUBMIT=1 (never-began swallow): TEST-DROP cv=1 ->
    SUBMIT-RECOVER round 1 fresh signals -> SIN-OK rc=0 (rung 1
    end-to-end, values correct).
  - S2 TEST_DROP_SUBMIT=1,2 (consecutive swallows): TEST-DROP cv=1 ->
    recover; TEST-DROP cv=3 -> recover; TWOOP-OK rc=0.
  - S3 TEST_DROP_SIGNAL=1 (all signals stripped): Mesa treats the
    signal-less batch as never-began too (started event never sets), so
    rung 1 recovered it; SIN-OK rc=0. Rung 2 (host-signal) stays a
    field-only defensive path: a batch Mesa actually executes always
    publishes its completion, so started-SET + completion-lost is not
    deterministically simulatable at submit time on this driver.
- OPERATIONAL NOTES: a leaked pre-fix probe (PID 256748) held the GPU
  device across several runs and was killed; venv-trace was overwritten
  with build14+ by an earlier misdirected pip (cloned venv shebang still
  pointed at venv-trace) - /tmp/venv-proof is the proof venv (build18);
  llm-inference was stopped/restarted around each run and confirmed
  healthy (HTTP 200) after the last one.

- STORM ARM (in flight at yield, /tmp/f1-round6-storm.log): primitive
  suite completed 2,700,949 assertions with exactly 1 failure and NO
  watchdog/timeline kill: the cholesky error-string check
  (test_primitives.cpp:1225, 'float64' substring) - an error-copy
  assertion, not F1 class; remaining 11 suites still running under the
  8-hog storm at yield time. Lane handoff: triage that string vs the
  0.32.3 dtype-name change, then finish the storm arm and continue the
  chain (Bonsai-2 stock + abliterated tok/s, land to main vs 63c1d3cf,
  4 certified E2E arms, oMLX A/B).
