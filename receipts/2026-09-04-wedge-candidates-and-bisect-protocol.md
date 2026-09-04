# 2026-09-04 — single-full-sequence-eval wedge: candidates and bisection protocol

Companion to docs/known-defects.md (v0.3.4, "A single large evaluation
can wedge the GPU queue"). This receipt enumerates the code-level
candidate mechanisms and lays out the bisection protocol; the M1 hardware
verification itself is queued behind Main's three-arm hotfix session on
BenchQueueM1 at the time of writing.

## What the receipt records

- **Confirmed discriminator**: `differential_compile --mode realpath
  --steps 1` (one `mx.eval` over the full 2,048-token eager forward)
  wedges Honeykrisp at the 10 s watchdog with the counter frozen at 0.
  `mlx_lm.generate` at 6,009 tokens completes in 183 s (chunked
  prefill). Evidence: receipts/2026-09-04-hang-watchdog-hardware.md.
- **What the watchdog sees**: `last_observed = 0`, `target = 1`. The
  timeline semaphore counter has not advanced at all across 10 s, so
  the watchdog classifies it as a wedged queue, not slow work
  (device.cpp:841-893).
- **What is NOT yet proven**: the precise mechanism (which Vulkan state
  in the submission path triggers the wedge). The M1 bisect will pin it.

## Code-level candidate enumeration

Reading overlay/mlx/backend/omarchy/encoder.cpp, device.cpp, and
allocator.cpp with the discriminator in hand, the candidates for "ONE
submission's size or content" reduce to:

### A. Single dispatch exceeding device-reported `maxComputeWorkGroupCount[axis]`

- Honeykrisp's report of this limit is not collected (device.h:92-99 has
  `maxComputeWorkGroupInvocations` and `maxComputeWorkGroupSize[3]` but
  no `maxComputeWorkGroupCount[3]`).
- The encoder clamps `group_count_x/y/z` to the constant
  `kMaxComputeGroupCountX = 65535` (encoder.cpp:241-243, compute.h:17).
  65535 is the Vulkan spec floor; a device reporting a smaller limit
  would silently accept a larger dispatch.
- For Qwen 0.5B SDPA at q_len=2048 (kv_heads=2, repeats=7, head_dim=64),
  `dispatch_matmul` records `group_count_z = batch_count = 14`
  (primitives.cpp:448). All axes under 65535, so this is NOT the
  candidate by itself. But the QK^T scores buffer scales with
  `q_len * k_len`; if a different primitive at q_len=2048 asks for
  `q_len*k_len/256 = 16384` workgroups, that exceeds any device limit
  below 16384.

### B. `vkCreateBuffer` size exceeding device `maxBufferSize`

- A single tensor scales with q_len. The QK^T scores buffer at
  q_len=2048 is `1 * 2 * 7 * 2048 * 2048 * 4 bytes (f32) ~= 235 MiB`
  (the SDPA path's `scores = qs @ keys_t`,
  primitives.cpp:6896-6901).
- Vulkan guarantees `maxBufferSize >= 2^31 - 1` (the spec floor);
  Honeykrisp's specific report is unverified. If the device reports a
  smaller limit, `CreateBuffer` would return `VK_ERROR_OUT_OF_DEVICE_MEMORY`
  or `VK_ERROR_INVALID_PARAMETER` and `VKX_CHECK` (allocator.cpp:126)
  throws a typed error — NOT a wedge.

### C. `vkAllocateMemory` size exceeding device `maxMemoryAllocationSize`

- The scores tensor backed by a 235 MiB `AllocateMemory` call. Spec
  floor is 2^31-1; if Honeykrisp reports less, `AllocateMemory` fails
  with a typed error (allocator.cpp:156) — NOT a wedge.

### D. A descriptor `range` exceeding device `maxStorageBufferRange`

- Vulkan spec floor `(2^31 - 1)`. Same pattern as B/C: the driver fails
  the call, the backend throws. Not a wedge by itself.

### E. Command buffer recording crossing a driver-side limit

- A single command buffer records every dispatch between commits. With
  `kBatchNodeBudget = 100` (encoder.h:52), the evaluator flushes every
  100 nodes, so one submission carries ≤100 dispatches. The Vulkan spec
  guarantees command buffers handle at least 2^16 dispatches, so this is
  not the candidate by itself.

### F. A driver-side hang on a SPECIFIC shader pattern at SPECIFIC shape

- Honeykrisp has 5+ documented miscompiles (see
  docs/known-defects.md entries). The fifth was the bf16 compiled-tape
  memory-visibility class. A sixth family could exist that activates at
  one specific dispatch shape (e.g., SDPA at q_len=2048 GQA reshape).
  This is the candidate with no spec-level defense; it requires M1
  measurement to characterize.

### G. Submission ordering or cross-submission wait pathology

- Each submission waits on the previous submission's completion
  timeline value (encoder.cpp:423-426). The first submission has
  `last_completion_ == 0` and no wait. Not a candidate.

### H. Pre-submission host stall (vkCreatePipeline, vkAllocateDescriptorSets)

- Pipelines are cached on first use (compute.cpp:735-745). The wedge's
  single mx.eval triggers many first-use pipeline creates on a fresh
  process. `CreateComputePipelines` can stall on a slow shader compile
  path, but a stall of >10 s would surface as a host-side hang (the
  host is the one stuck), and the watchdog measures only the timeline
  counter — which means the wedge would not surface as
  `last_observed=0` if the host is the stuck party. So this is NOT the
  candidate from the symptom shape.

### I. Descriptor-pool exhaustion path

- `kDescriptorSetsPerPool = 2048` (encoder.h:200) and a single full-
  sequence forward at q_len=2048 across 14 layers with multiple
  dispatches per layer — estimate ~300-800 descriptor sets per
  submission window. Well under 2048; not the candidate.

## Where this leaves us

The code-level enumeration narrows the wedge to one of:

- **F (driver-side shader hang at specific shape)** — the strongest
  plausible candidate. Cannot be defended against in pure code; can
  only be characterized on the M1.
- **A (`maxComputeWorkGroupCount[axis]` exceeded)** — defensible with
  a per-axis refusal if the device reports a smaller-than-65535 limit.
  Not currently collected.
- **B/C/D (allocation/buffer/descriptor-size limit)** — would surface
  as typed errors, not wedges; ruled out unless Honeykrisp has a
  silent-accept path on these limits.

The bisect isolates which one.

## Bisection protocol (committed, awaiting M1 run)

Two scripts committed at origin/wedge/q1-2048 commit 94c1dfc:

### scripts/wedge_bisect.py

Two-phase harness. Every probe is a fresh subprocess because the wedge
poisons process state.

- Phase 1: sweep prompt token length N from 1 to 4096 in a fixed grid,
  one mx.eval per N, eager path, watchdog at 10 s default. The
  smallest N that wedges gives the threshold.
- Phase 2: at the threshold, run single-factor variants:
  - baseline_eager (the wedge)
  - baseline_compiled (does compilation move the threshold?)
  - threshold_quarter / threshold_double (does the threshold sit at
    q_len or at total tensor bytes?)
  - short_watchdog (does the wedge trip earlier?)

Exit code per probe: 0 = OK, 124 = wedge, 1 = other.

### scripts/wedge_primitive_probe.py

Isolates which primitive the wedge lives in. Runs three primitives
separately, each sweeping q_len:

- sdpa: full Qwen-shape SDPA via `mx.fast.scaled_dot_product_attention`.
- qk_matmul: just the QK^T matmul on the post-reshape f32 buffer.
- softmax: the post-softmax reduction over the scores tensor.

If sdpa wedges at q_len=2048 but qk_matmul and softmax complete at
q_len=2048, the wedge lives in the SDPA composition (softmax + matmul
+ memory layout), not in any single op. If qk_matmul wedges alone, the
wedge lives in the matmul's submission shape. If softmax wedges alone,
the wedge lives in the reduction kernel.

## M1 run status

Protocol shipped to BenchQueueM1 (queue wedge-bisect v1) on
2026-09-04 evening CDT. At time of writing, the box is occupied by
Main's three-arm hotfix session; the wedge protocol is queued behind
it, not indefinitely parked. BenchQueueM1 will run it against the
post-watchdog origin/main + 94c1dfc wheel (sha 79e4036...; already
built) when the box is free.

When the result lands, the follow-up is:
1. Pin the candidate (F / A / B / C / D) from the bisect tables.
2. For A: add `maxComputeWorkGroupCount[3]` to CapabilityReport and
   refuse by name in `dispatch_compute` when a per-axis group count
   exceeds the device limit. Regression: a unit test that asks for a
   known-too-large dispatch and asserts the named refusal.
3. For B/C/D: confirm by M1 measurement that the driver returns an
   error code (not silent accept). If the driver silently accepts and
   hangs, add the same per-allocation size refusal.
4. For F: write the receipt; the fix is upstream in Honeykrisp. The
   defense-in-depth here is the watchdog itself (already shipped).
5. Update docs/known-defects.md: change the v0.3.4 entry from OPEN to
   FIXED-OURS (with refusal) or OPEN-WITH-WORKAROUND (with a "chunk
   prefill" recommendation), whichever the bisect supports.

## Not committed (and not yet to commit)

- No overlay changes. The fix depends on the bisect outcome.
- No docs/known-defects.md update. The entry stays OPEN until the
  M1 run lands.
- No regression test. The test's exact shape depends on which
  primitive trips the wedge; writing it speculatively risks pinning
  the wrong shape and shipping a false-positive guard.

## Verdict

The mechanism identification is incomplete: the M1 bisect is the
load-bearing measurement and the box is occupied. Code-level analysis
narrows the candidates to F (driver-shape) and A (workgroup-count
limit) as the two most plausible. The bisection scripts are ready
and committed; the M1 run will resolve which one is the cause.
