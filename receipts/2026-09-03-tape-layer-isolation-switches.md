# Tape layer-isolation debug switches

Date: 2026-09-03. Branch worktree off `origin/main` (`045ef20`).
Scope: compiled-tape execution only. No behaviour change when no switch
is set; verified by the unchanged default path and by the new battery
case's default-state assertions.

## Why

On Honeykrisp (M1), compiled tapes return wrong values while eager is
correct. Established facts (do not re-derive):

- The refused magnitude differs every run (8.08e8, 8.99e8, 9.53e8,
  9.60e8): a race signature, not a deterministic miscompile.
- Under GPU-assisted validation (serialized queue) the same run is
  correct. Slowing the device removes the wrong values.
- `omarchy_compiled_tape_tests` passes 8/8 cases, 343/343 assertions on
  the same M1 and driver: the defect needs a graph bigger than the tests
  build.
- The differential harness localises the failure to a broadcast Sigmoid
  at shapes like [1,30,4864]; an isolated qwen2 swiglu passes.
- Intra-command-buffer barriers are broad and look correct
  (encoder.cpp pre- and post-dispatch barriers). A missing barrier is
  NOT the hypothesis.
- Cross-submission ordering was fixed in ff4b05a.

The structural difference that remains: eager runs one dispatch per
command buffer per submission; the compiled tape records many dispatches
into ONE command buffer and submits once. Three switches isolate the
candidate layers inside that difference.

## The switches

Documented in docs/install-omarchy.md, "Compiled-tape debug switches".
All default off, read once per tape evaluation, one stderr line the
first time a tape runs with any of them active.

| Switch | Layer isolated | Where |
|---|---|---|
| `MLX_OMARCHY_TAPE_PER_NODE_SUBMIT=1` | many-dispatches-per-command-buffer | compiled.cpp, per-node `encoder.commit()` |
| `MLX_OMARCHY_TAPE_FULL_BARRIERS=1` | in-buffer dependency honouring | encoder.cpp `dispatch_compute`, full barrier both sides of every dispatch |
| `MLX_OMARCHY_TAPE_NO_REUSE=1` | aliasing and lifetime | allocator.cpp `malloc`/`free` cache bypass + encoder.cpp `acquire_descriptor_set` per-dispatch pool |

Plumbing: the env vars are read once per tape eval in
`eval_compiled_tape` and published through `TapeDebugScope` (device.h /
device.cpp, two relaxed atomics) so the encoder and allocator pay no
environment lookup on the unset hot path.

## Hypothesis space and what each outcome proves

Ordered decision tree for the M1. Every run needs
`MLX_OMARCHY_ALLOW_UNSAFE_COMPILE=1` (compiled tapes auto-disable on
Apple GPUs). Each run is the differential harness
`scripts/differential_compile.py`, which aborts at the accuracy gate
when the corruption fires; a run that finishes and answers correctly
counts as "corruption absent" only if the harness reports agreement.

1. Baseline: override only. Expect: corruption reproduces (abort at the
   Cos gate, nondeterministic magnitude). If it does NOT reproduce,
   stop and re-baseline the harness itself; do not interpret any
   switch result.
2. Run A: `MLX_OMARCHY_TAPE_PER_NODE_SUBMIT=1` (plus override).
   - Corruption gone -> the defect lives in many-dispatches-per-
     command-buffer recording. Go to 3 to separate barriers from
     resources.
   - Corruption persists -> the command buffer boundary is innocent.
     The defect is in the tape's resource handling. Go to 4.
3. Run B: `MLX_OMARCHY_TAPE_FULL_BARRIERS=1` (plus override, batched
   recording as usual).
   - Corruption gone -> the driver drops an in-buffer dependency the
     regular barriers already express. That is a Honeykrisp finding:
     write a minimal shader reproduction (the project has already
     isolated five such miscompiles).
   - Corruption persists (with A clean and B dirty) -> the defect
     tracks the submission boundary itself (fence/semaphore/ring-slot
     interaction on Honeykrisp), not in-buffer barriers. Compare
     per-node submission counts against main AFTER TinyWriteFix's
     empty-submission fix lands, or the baseline shifts.
4. Run C: `MLX_OMARCHY_TAPE_NO_REUSE=1` (plus override, batched).
   - Corruption gone -> aliasing or lifetime: a recycled buffer or a
     shared descriptor pool is involved. Bisect further with the
     allocator cache alone (clear_cache between decode steps) versus
     the per-dispatch descriptor pool.
   - Corruption persists everywhere and A persists too -> the defect is
     none of the three isolated layers; next suspect is the completion
     timeline / ring-slot reuse or the allocator's in-flight window,
     which no switch currently isolates.

One pass through A, then B or C, names the layer.

## Intermediate lifetime during recording (the specific hazard)

Question: can an intermediate's buffer be freed and re-handed to a
later node DURING recording of the same command buffer?

Answer: no, by construction, in the current code.

- `eval_compiled_tape` keys every node output into `resolved`
  (unordered_map, strong array references) for the whole function body.
  No intermediate drops to refcount zero during the loop, so the
  allocator can never recycle one mid-recording.
- Non-output intermediates are additionally pinned by
  `encoder.add_temporary` until their submission completes on the GPU
  (completion timeline), which outlives the recording.
- Buffer donation (`set_unary_output_data`) cannot fire inside the
  tape: a tape node's input always has at least two references
  (`resolved` plus the per-node `node_inputs`), so `is_donatable` is
  false and every node output takes a fresh allocation.
- The residual exposure sits one step out: the allocator's cache has no
  in-flight tracking. It trusts host refcounts plus `add_temporary`
  pinning. `MLX_OMARCHY_TAPE_NO_REUSE=1` removes that trust for the
  tape window entirely, which is exactly what Run C tests.

## Verification

Development device: llvmpipe (`VK_ICD_FILENAMES=lvp_icd.x86_64.json`,
`MLX_OMARCHY_ALLOW_NON_APPLE=1`) on the x86 dev box. llvmpipe cannot
reproduce the defect - this only proves the switches run and stay
correct. Development device, not Honeykrisp.

Build: `ninja omarchy_compiled_tape_tests` at this commit, clean.

`omarchy_compiled_tape_tests` (11 cases, 742 assertions), one run per
configuration:

| Configuration | Result |
|---|---|
| no switch | 11/11 cases, 742/742 assertions, SUCCESS; no switch notice until the battery itself arms one |
| `MLX_OMARCHY_TAPE_PER_NODE_SUBMIT=1` | 11/11, 742/742, SUCCESS; notice printed; submission-delta assertion held |
| `MLX_OMARCHY_TAPE_FULL_BARRIERS=1` | 11/11, 742/742, SUCCESS; notice printed |
| `MLX_OMARCHY_TAPE_NO_REUSE=1` | 11/11, 742/742, SUCCESS; notice printed |

The new battery case asserts both scoped defaults false with no env set
and - under `MLX_OMARCHY_TAPE_PER_NODE_SUBMIT=1` - that one 3-node
swiglu tape queues at least 3 submissions (decisive-bisector property:
the shape really changes). Under each switch the swiglu-shaped tape
(sigmoid, broadcast multiply, broadcast scalar add) matches eager at
1e-6 in float32.

Honest limits: on llvmpipe, correctness under the switches is expected
and proven; the defect itself needs the M1 protocol above. The barrier
and descriptor-pool branches are exercised on every dispatch while
their flags are set, so the llvmpipe runs do execute the switched
Vulkan recording paths, not just the env plumbing.
