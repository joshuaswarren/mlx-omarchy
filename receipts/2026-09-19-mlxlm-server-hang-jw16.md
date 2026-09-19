# mlx_lm.server / oMLX serving hangs: one runtime scheduler stall, root-caused

2026-09-19 · jw16 (T6001, Asahi Linux, recert wheel `b283a16` + libmlx16 `38adf6efb80ee850`) ·
closes the "mlx_lm server hang" parker from
[2026-09-18-f7-gdn-correctness.md](2026-09-18-f7-gdn-correctness.md) and unblocks the
oMLX-vs-mlx_lm comparison for docs/serve.md.

## Verdict

Every serving hang observed on this stack — `mlx_lm.server` on PyPI mlx-lm **0.31.3**, on
git-main **872ae88**, and **oMLX 0.6.4's own engine** — is a single omarchy-runtime defect:

**The Vulkan scheduler leaves a timeline value reserved with no signalling batch ever
enqueued.** The 10 s no-progress watchdog throws

```
RuntimeError: [omarchy] Vulkan timeline counter failed to advance for 10000 ms
(last observed=0, target=1). The device may be hung; no CPU fallback is available.
```

and the recovery ladder refuses to act because the retained-batch set is empty — the code's
own contract for "a stall whose target is already reserved but whose signalling batch does
not exist" (overlay/mlx/backend/omarchy/device.cpp, wait loop + `recover_stalled_submissions`).

The client-visible shape depends only on which server wraps the stall:

| server vintage | stall | client sees |
|---|---|---|
| mlx-lm 0.31.3 server | generation thread dies on the RuntimeError | request thread blocks on `queue.get` **forever** → the reported >180 s infinite hang (measured: HTTP code `000` at the 300 s client timeout on **both** `/v1/chat/completions` and `/v1/completions`) |
| mlx-lm 872ae88 server | identical stall (cv≈29, same frame) | fail-fast: server logs `mlx_lm.server generation thread died`, raises to the client; `/v1/completions` removed (404) |
| oMLX 0.6.4 (VLMBatchedEngine) | `Prefill failed for <uid>: [omarchy] Vulkan timeline …` | empty 200 response after ~10 s, per request |

Two consequences of the evidence:

1. **The known issue's "chat path hangs, completions fine" framing was an artifact.** Both
   routes go through the server's BatchGenerator; both hang identically on 0.31.3.
2. **There is no upstream floor version.** mlx-lm 872ae88 improves the failure mode
   (typed error, health-liveness tie, `StopSequences`) but the stall is unchanged. The fix
   belongs in the omarchy runtime's scheduler/event bridging.

## Isolation ladder (all runs inside `flock /tmp/m1-gpu.lock` GPU windows, service stopped/restarted+confirmed)

| # | workload | engine path | model | result |
|---|---|---|---|---|
| W1 | mlx_lm.server 0.31.3, `/v1/completions` then `/v1/chat/completions` | BatchGenerator (batched) | Qwen2.5-0.5B | both `000` @300 s; generate thread dies at first decode `GenerationBatch._step` after 53 prefill submits |
| W1 | mlx_lm.server 872ae88, chat | BatchGenerator | Qwen2.5-0.5B | `000` @21 s, fail-fast "generation thread died" |
| W2 | `stream_generate` ×3 (main / bare thread / `StreamContext(main)`) | single-sequence | Qwen2.5-0.5B | **all pass**, 82–84 tok/s — threads and per-thread streams (2, 4) exonerated |
| W3 | `BatchGenerator` ×3 (main / bare thread / thread default stream) | batched prefill → split → decode | Qwen2.5-0.5B | **all three stall identically**, `[rtmod] STALL target=1 observed=0 round=0 recovery=1` — main thread included, so threading is not the trigger |
| W5 | `stream_generate` direct | single-sequence | Ministral-3-8B | OK, 5.28–5.42 tok/s |
| W4/W6 | mlx_lm.server 872ae88 with request `seed:0` → `_serve_single` | `stream_generate` | Ministral-3-8B / Qwen2.5-0.5B | serves fine (4.3 tok/s / 44–57 tok/s e2e) |
| W4/W6 | oMLX 0.6.4 `serve` | own scheduler | Ministral-3-8B | **prefill stall every request** (3/3), empty 200s |
| W6 | oMLX 0.6.4 `serve` | own scheduler | Qwen2.5-0.5B | **serves clean**, 0 prefill failures — faster e2e than mlx_lm single path |

Mechanistically the failing frame is always the same transition shape: work is planned on a
stream, a waiter reserves/waits timeline value 1 on a semaphore whose counter never leaves 0,
and the batch that would signal it is never handed to the driver. Which (engine pattern ×
model graph) pairs hit it varies — mlx-lm's batched-prefill→`split()`→first-decode-step on
Qwen2.5-0.5B, oMLX's VLM prefill on Ministral-3-8B — while single-sequence decode escapes on
both models. This is a submission-planning defect in the runtime, not a library-usage bug.

## Minimal repro (main thread, deterministic, ~30 s)

```python
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import BatchGenerator
from mlx_lm.models.cache import make_prompt_cache

model, tok = load("/…/snapshots/a5339a41…")   # Qwen2.5-0.5B-Instruct-4bit
bg = BatchGenerator(model, prefill_step_size=512)
prompt = tok.encode("Say hi in one word.")
bg.insert_segments([[prompt]], [24], [make_prompt_cache(model)], [prompt])
bg.next()   # prefill completes; first decode GenerationBatch._step raises:
```

```
generate.py:1809 _next → 1186 generate → 1288 __init__ → 1369 _step
    mx.async_eval(self._next_tokens, self._next_logprobs, token_context)
RuntimeError: [omarchy] Vulkan timeline counter failed to advance for 10000 ms
(last observed=0, target=1)
```

Full logs: `ab/probe-bg.log` (W3), `ab/probe-thread.log` (W2), `ab/ref-ministral.log` (W5)
in this receipt directory; window ledgers in `ab/window*.status`.

## Working configurations today

- **mlx_lm.server (any vintage): send requests with `"seed": <int>.** `_is_batchable` is
  false for seeded requests → `_serve_single` (`stream_generate`) → the safe path. This is a
  degraded configuration (no request batching; per-token host work makes 0.5B e2e ≈
  44–57 tok/s vs oMLX's 57–111), i.e. a **P1 workaround** until the runtime fix lands.
- **oMLX 0.6.4 works on Qwen2.5-class graphs** with the venv fixed (below), and fails on
  Ministral-3 prefill (same stall).
- oMLX venv on jw16 was broken independent of the stall: `omlx 0.6.4` imports
  `SequenceStateMachine` from `mlx_lm.generate`, present in the 0.31.3-era API but gone in
  0.32.0 and 872ae88 (moved/renamed). Fixed by installing the pyproject-pinned
  `mlx-lm @ ab1806e8` into `/tmp/venv-omlx`; `omlx.server` then imports and serves. The
  `StopSequences` API (12 refs in generate.py at 872ae88) is required by the **newer** oMLX
  line, not by deployed 0.6.4.

## oMLX-vs-mlx_lm A/B (non-streaming, temp 0, max_tokens 128, warmup 1, tok/s = completion_tokens / total e2e incl. prefill)

Qwen2.5-0.5B-Instruct-4bit (both stacks functional):

| prompt | mlx_lm 872ae88 single-path tok | tok/s | oMLX 0.6.4 tok | tok/s |
|---|---|---|---|---|
| "Say hi in one word." | 3 | 57.1 | 2 | 10.8 |
| "Explain in one sentence why the sky is blue." | 67 | 50.7 | 66 | **111.3** |
| "Name the capital of France and its population…" | 21 | 44.1 | 20 | **57.0** |

Ministral-3-8B-Instruct-2512-4bit:

| stack | result |
|---|---|
| mlx_lm.server single path | serves; 4.3 tok/s e2e (39 tok / 9.0 s) — consistent with the 5.3 tok/s direct `stream_generate` reference, i.e. this model's decode is kernel-limited on the runtime, not server-limited |
| oMLX 0.6.4 | non-functional: prefill timeline stall on every request (same runtime defect) |

Raw payloads and client logs: `ab/ab-*.json`, `ab/ab-*.client.log`.

## Fix lane (next work, inputs prepared)

1. Reproduce with `ab/probe_bg.py` under `MLX_OMARCHY_TRACE_DISPATCH=1` (already emits
   `[rtmod] STALL tid target observed round recovery` lines).
2. Instrument the reserve→submit edge in the scheduler: the waiter's `target_value ≤
   last_reserved()` yet `progress->has_active_submission(progress_through)` stays false and
   the retained set is empty — find where the owning batch is dropped between reservation and
   `Encoder::commit()`/submit for (a) the split→first-decode-step pattern and (b) oMLX VLM
   prefill on Ministral.
3. Regression guard: commit `probe_bg.py` as a gate leg (fails in <40 s today) once the fix
   lands, plus an oMLX serve smoke on Ministral.
4. Until fixed, docs/serve.md configurations should send seeded requests to mlx_lm.server.

Relates: the candidate-wheel matrix probe (`/tmp/rtmod-matrix.log`, 2026-09-18) showed the
same signature on `generate()` with Ministral-3-8B while Qwen2.5-0.5B passed — same defect
family on the 0.32.3-bump-candidate wheel.
