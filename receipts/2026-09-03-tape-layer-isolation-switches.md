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
Apple GPUs; the native mlx_lm path arms itself).

Detector, amended on hardware report 2026-09-03 (BenchQueueM1): at this
commit the differential harness `--mode realpath` cannot discriminate -
it refuses at `[omarchy] broadcast Sigmoid is not implemented ...
shape=[1,36,4864]` (same refusal as d1a6bfd). The tree therefore runs
on the NATIVE mlx_lm decode instead: detection is the loud abort
(Cos accuracy gate, rc != 0, magnitude line) versus a correct
completion (rc = 0, correct answer). Bitwise per-step comparison is
lost until the broadcast-Sigmoid refusal is fixed; that fix is a named
follow-up, not a blocker for the layer verdict. Because the corruption
is a race, every step runs the native decode 5 TIMES: a switch counts
as "gone" only on 5/5 correct completions; any abort counts as
"persists"; baseline establishes the abort (observed at 7c25feb:
rc = 1, magnitude 855782848, and 4 earlier runs with differing
magnitudes). Magnitude nondeterminism is expected and irrelevant - the
verdict keys on abort versus correct answer.

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

## MEASURED OUTCOME (M1, native tree, BenchQueueM1)

Ran at 7c3d6b4 (contains 7c25feb plus TinyWriteFix's batching changes),
wheel provenance gate green, 5 runs per step, any abort = persists.
Baseline Cos abort: magnitude 8.56e8.

| Step | Switch | Result | Magnitude |
|---|---|---|---|
| A | `MLX_OMARCHY_TAPE_PER_NODE_SUBMIT=1` | PERSISTS, 5/5 aborted | 1.46e11, constant across all 5 |
| B | `MLX_OMARCHY_TAPE_FULL_BARRIERS=1` | PERSISTS, 5/5 aborted | 1.46e11, constant across all 5 |
| C | `MLX_OMARCHY_TAPE_NO_REUSE=1` | PERSISTS, 5/5 aborted | 7.97e8-9.00e8 (baseline band) |

Verdict: none of the three isolated layers removes the corruption. The
defect is not many-dispatches-per-command-buffer, not an in-buffer
dependency (B already used the heaviest correct barrier), and not
in-window resource reuse. Remaining named space: host run-ahead across
the tape boundary, cross-window buffer recycling, ring-slot command
buffer identity, completion timeline.

The band datum is load-bearing: the refused magnitude TRACKS the switch
class. A and B (serialization/timing changes) move the garbage to a
CONSTANT 1.46e11 = exactly 34 x 2^32 in all runs - a deterministic
stale word, not random noise; C and baseline (timing unchanged) stay in
the jittery 2^29.6-2^29.8 band. Two consequences: (1) the corrupted
read picks up foreign memory whose address lands differently under
different execution timing; (2) A or B is the reproduction workhorse
for the next bisect - deterministic garbage is traceable, jittery
garbage is not.

C's same-band result also exonerates in-window recycling specifically:
fresh device memory for every tape allocation changed nothing, so the
stale bytes come from buffers whose lifetime is managed outside the
tape window - the cross-window cache channel or a non-tape buffer.

Detector notes (TapeCorruptionFix evidence + bit arithmetic):
- The gate's unordered host read was fixed at d8cb4f6 (09:26), yet
  sync-val runs at ff4b05a (14:52) and 7cf5e9f (16:45) still aborted
  with f16-impossible magnitudes (~8-9.6e8 > f16 max finite 65504), so
  those aborts read genuine corrupted f32 values (RoPE-arg path), not
  the gate's own race. The corruption was live at 16:45.
- 1.46e11 = exactly 34 x 2^32 (arithmetic verified), constant across
  5/5 runs under A and B. As float32 bits that is 0x52100000; as two
  packed f16 halves it reads [0x5210 = 33.0, 0x0000 = 0.0] - a small
  integer in sequence-position territory (36-token prompt), consumed
  through a float view. Inference, not verified: the stale word may be
  a position id or position-derived index on a RoPE argument buffer.

Next bisects (implemented at this commit): STEP E
`MLX_OMARCHY_NO_BUFFER_CACHE=1` - process-wide, the allocator never
recycles (every free destroys); tests cross-window recycling with
in-flight staleness. STEP D `MLX_OMARCHY_TAPE_SYNC_EVERY=1` - the
stream drains after every tape eval; tests whether host run-ahead
(GPU executing while the host records/queues) is load-bearing. Run E
first, then D; outcomes are independent. If both persist, recycling
and asynchrony are innocent and the space narrows to ring-slot
command buffer identity versus the completion timeline, under A or B's
deterministic-garbage configuration.

## POSTMORTEM: the red submission assertion (same night)

At 7c3d6b4 the PER_NODE_SUBMIT case went red (delta=2, wants >=3,
deterministic, reproduced on llvmpipe). Investigation:

1. The switch was NOT defeated. Instrumented run: every dispatching
   tape node still forces its own submission (pre-commit nodes=1);
   CommandEncoder::commit still submits; 7c3d6b4 did not touch
   compiled.cpp.
2. The old assertion counted the wrong thing. The test's swiglu lambda
   captured a stream AND fed a shared intermediate to two outputs; the
   tracer splits at multi-consumer nodes, so the fn's compiled tape was
   a 2-node [Broadcast, Add] remnant with the sigmoid/multiply running
   eager inside the compiled call (measured: tape n=2 in=2
   ops=Broadcast,Add). Under the OLD per-op evaluator the >=3 bound was
   satisfied by eager per-op commits - it was passing for the wrong
   reason. Under batching those commits are gone.
3. Corrected assertion, written against the confirmed batching
   contract (flush sites: finalize, event contracts, 100-node budget,
   host-read sync; explicit commit() still force-submits; one
   submission per dispatching node under the switch): a single-output
   LINEAR variant (sigmoid, multiply, add - one consumer per node)
   tapes fully as [Sigmoid, Multiply, Broadcast, Add]; the test
   measures baseline (switch off) and switched deltas for the same fn
   and requires strict dominance (delta_off < delta_on) plus
   delta_on >= 3. Dominance is the property the hardware bisect relies
   on; a fixed constant would re-break on the next tracer change.
4. Semantics note for the protocol: per-node means per-DISPATCHING
   node. View-only tape nodes (Broadcast) record zero commands and
   produce no submission under either mode.

## ROOT CAUSE (supersedes the race interpretation above)

While the switches were in flight, TapeCorruptionFix pinned the actual
defect, and it is NOT a race: eval_compiled_tape materializes every
node at the TRACED shape. Upstream compile_replace derives shapeless
node shapes at eval time; this interpreter did not, and the shapeless
compile cache keys on ndim+dtype, so decode legally reuses the
prefill-traced fragment. Bigger traced shape than eval input = the node
output is sized at the traced shape and reads PAST the eval input
buffer. Lavapipe serves zeros past the buffer (looks clean, matches
eager); Honeykrisp serves recycled pages - which is the ~8-9e8 jittery
band, the constant 34x2^32 position word under the serialization
switches, the f16-impossible magnitudes, and 'Parisse'. Every datum in
MEASURED OUTCOME is consistent with OOB-into-recycled-pages; the
race interpretation of the differing magnitudes is retracted. The
broadcast-Sigmoid refusal is the same root cause in the opposite
direction: a traced node shape larger than the eval input surfaces as
the named refusal instead of silent OOB. The fix (eval-time shape
derivation in eval_compiled_tape) is TapeCorruptionFix's; the A/B/C/D/E
verdicts above all stand as exonerations - the defect was never in the
submission/buffer layers the switches isolate.

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
| `MLX_OMARCHY_TAPE_SYNC_EVERY=1` | 11/11, 742/742, SUCCESS; notice printed |
| `MLX_OMARCHY_NO_BUFFER_CACHE=1` | 11/11, 742/742, SUCCESS; init notice printed (armed before runtime init, as required) |

The two new switches bisect the remaining space named by the M1 verdict
above. `MLX_OMARCHY_TAPE_SYNC_EVERY` is read per tape eval and is
covered by the in-battery loop; `MLX_OMARCHY_NO_BUFFER_CACHE` is read
once at runtime init, so it is verified by a full-process env run (the
in-battery setenv cannot arm it after init - the test comments say so).

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
