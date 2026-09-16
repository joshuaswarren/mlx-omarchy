# 2026-09-15 TDT GPU loop, round 3 — GPU-resident greedy control (agent/tdt-gpu-loop)

Status: **exactness proven end-to-end on both arms and tdt_decode under the
target — LAND (branch only, default decode path unchanged).**

Round 3 target (from receipts/2026-09-15-tdt-gpu-step2.md): remove the
per-emission host sync by moving the greedy TDT control flow onto the GPU.

## Gate results (jwm1-linux, flock -x /tmp/m1-gpu.lock, never stolen)

| arm | gate | result |
| --- | --- | --- |
| native encoder (fused_diagnose --tdt-loop) | first_divergence = null | **None — PASS** |
| native encoder | decision 142 token / logit gap | **7892 / 0.0 — PASS** |
| ANE encoder (fused_e2e --tdt-loop) | 104/104 emissions, transcript db501a8c | **match, sha db501a8c080380ea… — PASS** |
| unit (validate_loop.py, 5 seeds) | emission stream == landed host path, incl. windowed chaining | **5/5 — PASS** |
| perf | tdt_decode < 2880 ms (target < 1000) | **833.3 ms — PASS** |

Timing history for tdt_decode on the same fixture: numpy host path 2880 ms
(gate), step2 queued six-dispatch step 2939.9 ms (NO-LAND), this loop
**833.3 ms** in the e2e stage wall / 896.5 ms standalone (best of 3,
`bench_real.py`), a 3.5x drop vs the gate and 3.4x vs step2.

## What landed on this branch

`overlay/tools/coreml/vulkan_tdt_loop.py` — `run_tdt_loop`: the ENTIRE
`greedy_tdt_decode` state machine in ONE workgroup of ONE custom kernel
dispatch, launched once per decode (single-dispatch mode) or once per frame
window (`frame_stop`), with the host reading tokens/state once at the end.
No per-emission host sync exists.

Kernel design (1024 threads, one workgroup, ~27 KB workgroup memory):

* LSTM layer chains + fold + cell per output lane, `exact_fma16` block
  chains from fp16 0 ascending k, fold b0 assigned then +b1..+b9, bias,
  bit-indexed LUT gate pairing and cell — the step2 contract verbatim, with
  all cross-thread intermediates (h1 staging, x snapshot `s_a`) in
  workgroup memory so no device-memory ordering is required.
* Projector: fp32 ascending `acc = acc + a*b` (precise, no fma), one fp16
  rounding, fp16 bias — staged through `s_relu` to avoid a read-write race
  on `s_pj`.
* Joint head: shared `s_relu`, fp32 ascending chain per output lane.
* In-kernel argmax reproduces `np.argmax`: strict `>` ascending strided
  scan per thread, cross-thread reduce prefers smaller index on ties, first
  NaN wins; duration logits land in `s_dval` and reduce the same way.
* Control in workgroup `s_ctl`: blank → `frame += max(duration, 1)`;
  emit → record (token, frame, duration), `symbols++`, advance on
  `duration > 0`; symbols cap → `frame += 1`; duration index outside the
  class count raises on the host (`TdtControlError` analogue).
* Window chaining: `frame_stop` bounds a launch; hidden, cell, decoder
  state (`s_pj`), input token, validity and frame start chain through the
  `initial_*` arguments, so a windowed decode reproduces the single-launch
  emission stream exactly (validated).

Why this removes the step2 blocker: step2's per-emission cost was the
fork's solo-dispatch round trip (~17x the queued cost) paid on every
emission because the host needed each argmax. Here the host never asks —
the argmax and the control run in-workgroup — so the round trip is paid
once per decode instead of once per emission.

## Validation notes

* `validate_loop.py` replays each seed through the LANDED harness callback
  semantics (fused_e2e's `fused["frame"] == frame_index` reuse rule). An
  earlier draft of the validator reused the decode's tok/dur
  unconditionally, which silently used a joint computed at the PREVIOUS
  joint's frame; that made the loop look divergent on seed 20260918 while
  the loop was actually faithful to the real semantics. `probe_replay.py`
  (callback-by-callback log) and `probe_stale.py` (mode-1 joint at stale
  vs current frame) pin this down; keep the distinction in mind for any
  future harness work: after a decode whose frame advanced, the harness
  recomputes the joint at the new frame via the joint-only path.
* `probe_dump.py` (per-decode pre/post state dump, decode-indexed slots)
  proves every in-kernel decode is bit-identical to the chained `run_step`
  path once the reference semantics are matched.
* Emission capacity is `valid_frames * max_symbols_per_step` (frame
  advances >= 1 per outer iteration, <= max_symbols emissions each); a
  breach raises instead of overwriting.

## Artefacts

- `overlay/tools/coreml/vulkan_tdt_loop.py` — `pack_step_weights` reuse
  from step2; public entry `run_tdt_loop(packed, encoder, valid_frames,
  config, ...) -> TdtLoopOutput` (token_ids, frame_indices, durations,
  decoder_state, hidden, cell, chaining fields).
- `receipts/2026-09-15-tdt-gpu-loop/derivation/` — `validate_loop.py`
  (unit gate), `bench_real.py` (fixture bench vs golden token_ids),
  `fused_e2e.py`/`fused_diagnose.py` (harness copies with opt-in
  `--tdt-loop`; default path unchanged), gate wrappers
  (`run_validate.sh`, `run_diag_loop.sh`, `run_e2e_loop.sh`), gate
  evidence (`e2e-report.json`, `token_ids.json`, `transcript.txt`,
  `fused_diagnose_out.json`), and the probes cited above
  (`probe_replay.py`, `probe_stale.py`, `probe_dump.py`, `probe_stream.py`,
  `probe_divergence.py`).
- jwm1 stage: `/var/tmp/TdtGpuLoop/{pkg,e2e-after,e2e-scratch}`
  (`pkg/coreml/vulkan_tdt_loop.py` is the module under test).
- Base: agent/tdt-gpu-step2 d5d250a2. NOT merged; commit 63c1d3cf
  untouched. Default decode path unchanged.

Resolved model: zai/glm-5.3-flash
