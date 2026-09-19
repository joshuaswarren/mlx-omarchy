# Vulkan timeline serving stall: root cause fixed, serving verified on jw16

2026-09-19 · jw16 (T6001, Asahi Linux) · branch `timeline-stall-fix` (base 87a84b40,
fix commit c08cf2ed, fast-follow v0.7.1 candidate — not tagged) ·
closes the fix lane opened by
[2026-09-19-mlxlm-server-hang-jw16.md](2026-09-19-mlxlm-server-hang-jw16.md).

## Root cause (as diagnosed, then as proven)

The diagnosis receipt's working theory — "the scheduler reserves a timeline value but
never enqueues the signalling batch" — pointed at the completion dispatcher. The live
instrumented repro (event-pointer tracing through EV-SIGNAL → SUBMIT → EV-WAIT, one
deterministic `probe_bg.py` run) relocated the leak one layer up:

**The deadlock is a mid-tape host readback blocking on its own pass's event latch.**

1. mlx-lm's BatchGenerator split step runs the whole model graph inside one
   `async_eval` pass (`mx.async_eval(self._next_tokens, ...)` — the logits were never
   dispatched before this point, so the pass's tape contains every layer).
2. `eval_impl(async=true)` attaches the pass's per-stream event to every output it
   dispatches, including the padded-batch cache offset array produced inside the same
   tape. The event signals only in the pass's epilogue (ride-along on the epilogue
   commit, or host-signal when idle).
3. The first RoPE layer's `rope_trig_gate` reads the offset on the host. The offset is
   not a host constant (padded batch path), so the gate synchronizes (commits + joins
   the completion timeline — data-safe), then calls `offset.item<int>()`.
4. `item()` → `array::wait()` → the offset's event latch: counter 0, and the ONLY
   signaler is the pass epilogue — still ahead on the call stack. The dispatching
   thread blocks forever. The 10 s watchdog fires
   `Vulkan timeline counter failed to advance (last observed=0, target=1)` — the error
   text made it look like a scheduler submission loss; it is an unsignalable event
   latch. The recovery ladder's refusal (`empty-batches`) is CORRECT behavior: every
   retained submission had drained; there was no reservation leak and no dropped
   submission. The refusal-bounded typed throw (landed with the F1 chain) then killed
   the generation thread; on 0.31.3 the request thread parked in `queue.get` forever —
   the reported serving hang on both endpoints.

The trace that pinned it (bg-thread-stream variant, the stall iteration): the previous
step's event wait returns normally (ride-along landed with the step's batch), the pass
records the decode forward into the open command buffer, `SUBMIT cv=87 waits=1 sigs=1`
commits it from inside the gate, `JOIN reason=rope_offset_scalar last=87` completes,
then `EV-WAIT ev=0xfffe3c0d4030 val=1 cnt=0` stalls on the CURRENT pass's fresh event —
signaled only later by the exception handler's `EV-SIGNAL ... path=host-signal` after
the watchdog threw. Healthy steps all show `EV-SIGNAL path=queued` riding their
epilogue commit (`sigs=2`).

Only the gate's scalar path was broken: the vector path already uses
`settle()` + `synchronize()` + read on a fresh event-free node, and the other host
readbacks guard on `Status::available && !has_primitive()` (host constants only).

## The fix (c08cf2ed, 2 files, +53/−11 with docs)

`overlay/mlx/backend/omarchy/primitives.cpp` — `rope_trig_gate` scalar path mirrors the
vector path:

```cpp
if (!host_constant) {
  settle({offset});                                   // schedule if unscheduled (event-free)
  omarchy::get_command_encoder(stream).synchronize("rope_offset_scalar");
  offset.detach_event();                              // stale latch: only the pass
}                                                     // epilogue could signal it, and the
worst_offset = ...offset.item<int>()...;              // data is proven final above
```

`overlay/mlx/backend/omarchy/encoder.cpp` — `CommandEncoder::submit()` catch path
retires the reservation it made: a failed `QueueSubmit` now publishes the empty
completion entry, so waiters get the typed watchdog error instead of blocking on a
`drained_value` that can never advance (every reserved value must have a completion
entry — the literal "reserved but never enqueued" leak, on the failure path).

## Verification ladder (jw16, fixed wheel c08cf2ed)

Wheel `mlx_omarchy-0.32.3.dev202609190724+c08cf2ed-cp314-cp314-linux_aarch64.whl`,
sha256 `9c25ee545e61cd45bfe6fcf981632b0d4434587d13fbc4d5778b0d7bbe2eefa3`. All GPU work
inside `flock /tmp/m1-gpu.lock` windows with llm-inference stopped before and
restarted + confirmed after (windows end `RESTORE service=active health=200` at
02:48:01-05:00). Artifacts: `/tmp/tlfver/` on jw16 (probe logs ×3, suite logs, server
logs + response bodies, summary.txt + summary2.txt).

| # | leg | before | after |
|---|---|---|---|
| A | `probe_bg.py` ×3 runs, 3 BatchGenerator variants each (main / bare thread / thread default stream) | 3/3 deadlocks per run, ~40 s to the typed throw | **exit=0 ×3, all variants ok** (02:35:16–02:35:28) |
| B | `omarchy_primitive_tests` | — | **103/103 rc=0** (2,700,952 assertions) |
| B | `omarchy_runtime_tests` | — | **41/41 rc=0** (22,694 assertions) |
| C | single-sequence `stream_generate` control, Qwen2.5-0.5B | 82–84 tok/s baseline | **143.6 tok/s**, 128 tok in 0.89 s — no regression |
| D | mlx_lm.server 0.31.3, batched (no seed): 12 completions + 12 chat | first request hangs 000 @ 300 s, generation thread dead | **12/12 completions 200 with text; 12/12 chat HTTP 200; thread_deaths=0, stalls=0** |
| E | mlx_lm.server 872ae88: 12 chat | "generation thread died" fail-fast | **12/12 chat 200 with content** ("The sky appears blue because sunlight is scattered..."), thread_deaths=0, stalls=0 |
| F | oMLX 0.6.4 serve, Ministral-3-8B: 6 requests | prefill stall on EVERY request (3/3 empty 200s) | **6/6 answers, `Paris` ×3 checked, prefill_stalls=0** |
| G | Parakeet pinned-reference E2E (mel → ANE islands → TDT → text) | — | **status=match, checks_failed=[], 104/104 emissions** |

Honest note on D's chat rows: 6 of 12 chat responses carry `content: null` because
max_tokens=32 truncated inside the model's reasoning span (`finish_reason: length`,
`reasoning` populated). They are HTTP 200 with valid completion bodies and zero thread
deaths; the completions rows and the E leg carry plain text end to end. The model dir
`/tmp/ab-models/qwen25-05b` is Qwen2ForCausalLM per config.json; the servers' registered
HF ids are cosmetic registry labels and do not affect routing.

## Bookkeeping

- Branch `timeline-stall-fix`: base 87a84b40 (origin/main), ancestry guard
  `git merge-base --is-ancestor 63c1d3cf timeline-stall-fix` → exit 1 (PASS: the
  parakeet-mel-exact tip is not in the branch).
- docs/known-defects.md: entry flipped to FIXED with fix commit + verification numbers;
  the degraded P1 workaround (seeded requests → `_serve_single`) is obsolete — batched
  serving works.
- Fast-follow v0.7.1 candidate: merged to main, NOT tagged (per lane instructions).
