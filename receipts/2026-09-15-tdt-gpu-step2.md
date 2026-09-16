# 2026-09-15 TDT GPU step, round 2 — queued dispatch split (agent/tdt-gpu-step2)

Status: **exactness proven end-to-end on both arms; perf gate still unmet —
NO-LAND. Default decode path unchanged.**

Round 2 target (from receipts/2026-09-15-tdt-gpu-step.md): lane-parallel +
exact block-fold k-split, faster than the numpy path while bit-exact.

## Gate results (jwm1-linux, flock -x /tmp/m1-gpu.lock, never stolen)

| arm | gate | result |
| --- | --- | --- |
| native encoder (fused_diagnose) | first_divergence = null | **None — PASS** |
| native encoder | decision 142 token / logit gap | **7892 / 0.0 — PASS** |
| ANE encoder (fused_e2e) | 104/104 emissions, transcript db501a8c | **match, sha db501a8c080380ea… — PASS** |
| perf | tdt_decode < 2880 ms | **2939.9 ms — FAIL (−2%)** |

Unit contract: validate_step2 24/24 trials bit-exact vs the landed numpy
BNNS LSTM and run_joint logits, including recurrent chaining.

Final timings (e2e-report.json in this dir): decoder_mean 16.38 ms
(numpy baseline 16.22), joint_mean 3.86 ms (baseline 3.60), tdt_decode
2939.9 ms (baseline 2880). The GPU path reached per-call parity with
numpy but not the required sum.

## What landed on this branch

`overlay/tools/coreml/vulkan_decoder_step.py` — queued six-dispatch step:

1. layer-0 block chains (25600 threads, one 128-wide BNNS block chain each)
2. layer-0 fold + gate pairing + cell (640 threads, one group)
3. layer-1 block chains
4. layer-1 fold + gate pairing + cell
5. projector chain + relu (640 threads)
6. joint head (8448 threads, one output lane per thread)

Argmaxes run host-side over the returned fp32 logits (np.argmax =
first-max = the landed in-kernel strict-`>` ascending scan). One host
sync per step; the joint-only callback is a single joint dispatch that
rebuilds the relu from fp16(dec_in).

Exactness is by construction at every split: the ten 128-wide k-block
chains of a (lane, gate) are independent up to the fold, so any thread
grouping preserves the contract; the fold is sequential b0..b9 with the
first term assigned (not added, preserving -0); cell, projector, joint
keep the landed op order including the fp16 rounding points.

## Measurements that constrain any future attempt

All numbers from the probes in `derivation/` (jwm1, Honeykrisp fork,
mlx-omarchy 0.32.2):

* `tidx_probe.py`: on this fork `grid=` is the TOTAL thread count and
  `threadgroup=` the group size; `thread_index_in_threadgroup` then spans
  correctly. The first queued build launched 40/640/33 threads instead of
  25.6k/640/8448 and produced wrong/NaN outputs.
* `submit_probe.py`: host submit is 7 µs/call. 50 queued layer dispatches
  drain at 0.16 ms each; a SOLO synced dispatch of the same kernel costs
  6.5 ms. The fork's solo-vs-queued round trip (~17x) is the dominant
  per-emission cost and explains why the round-1 single-dispatch step was
  slow: it was one solo dispatch of a huge kernel.
* `dispatch_bisect.py`: the synced layer cost does NOT scale with content
  (inlining, interleaving `interleave_probe.py`, software-pipelined loads
  `pipeline` variants, hw-fma instead of exact_fma16, math_mode fast, or
  geometry 640-25600 threads all land 4.7-6.6 ms). exact_fma16 costs only
  ~21% over a raw hw fp16 fma chain (`opcost_probe.py`), so there is no
  emulation overhead to remove.
* `geometry_probe.py`/`single_probe.py` sweeps: wall time is flat in
  group count and thread count; the fork serializes groups solo.
* Group-count reduction is the only lever that moved the solo joint
  (33x256 = 2.8 ms floor); speculation across frames (`_JOINT_SPEC`
  experiments, since removed) moved work between buckets but the e2e
  total stayed 3060-3380.
* Environment drift: identical code benched 16.7 ms/step and 24-25
  ms/step an hour apart on an idle 28 C machine (`np_bench.py` shows the
  CPU side unchanged) — absolute tdt numbers carry ±10% machine mood.

## Why the perf gate still fails

The greedy TDT control forces one host sync per emission (the next
frame/token decision needs the argmax), so the step can never amortize
the solo round trip. At 2 dispatches the step is ~16.4-17.0 ms and the
mode-1 joint ~3.4-4.2 ms; the numpy contract path pays no dispatch round
trip on the decoder (16.2 ms pure CPU) and one mx.matmul round trip for
the joint (3.6 ms). Reaching < 2880 ms needs the per-step total under
~16.5 ms including the joint — i.e. beating the fork's solo dispatch
latency itself. The candidates that would do that are fork-level work
(submission batching across the sync, or a persistent-kernel step with
device-wide sync), not kernel restructuring.

## Artefacts

- `overlay/tools/coreml/vulkan_decoder_step.py` — the queued six-dispatch
  step (same public API: `pack_step_weights`, `run_step`).
- `derivation/` — all probes cited above, `validate_step2.py`,
  `bench_step2.py`, `stage_times.py`, `debug_nan.py`, final
  `e2e-report.json` (ANE arm) and `fused_diagnose_out.json` (native arm).
- jwm1 stage: `/var/tmp/TdtGpuStep2/{pkg,e2e-after,e2e-scratch}`,
  wrapper `/var/tmp/TdtGpuStep2/run_fused_e2e2.sh`.
- Base: agent/tdt-gpu-step a12587d6. NOT merged; commit 63c1d3cf
  untouched.

Resolved model: zai/glm-5.3-flash
