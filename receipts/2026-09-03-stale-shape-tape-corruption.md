# 2026-09-03 — the Honeykrisp compiled-tape corruption: root cause, fix, proof

Branch `tape-corruption-fix` (merged to main by CompiledFailClosed): commits
`13d83f7` (the fix), `650e324` (the battery regression case), `6cc0c07`
(the recycled-storage poison detector).

## The defect, named

`eval_compiled_tape` materialized every tape node output at the node's
**traced** shape. Upstream's shapeless compile cache matches entries by
ndim and dtype only (`mlx/compile.cpp`, `has_same_shape_and_dtype`), so
one traced fragment legally serves every input shape: a decode call at
`[1,1,...]` reuses a prefill-traced tape. Upstream handles this -
`compile_replace` recomputes each node's shape at eval time through
`primitive().output_shapes(real_inputs)` for shapeless tapes
(`mlx/compile.cpp:1074-1097`), and the Metal fused kernel consumes
eval-time shapes at launch. The Omarchy interpreter did neither: node
outputs kept prefill sizes.

Consequences, all from one mechanism:

- Outputs computed into the wrong shape (grow direction).
- Inputs read **past their buffers** (decode direction: output allocated
  at prefill size, kernel reads `out.size()` elements from a
  decode-sized input).
- `dispatch_elementwise` refusing with the recorded
  `broadcast Sigmoid is not implemented` error whenever the stale shape
  was not trailing-broadcastable - the fence that masked everything on
  current main.

**This was never a race.** The refused Cos-gate magnitudes were
f16-impossible (f16 max finite is 65504; observed band 8.08e8-9.60e8):
the eager trig gate faithfully measured memory read past the undersized
eval inputs. On lavapipe those pages are zero, which is why llvmpipe
always matched eager, the 8/8 battery (traces and evals at one shape)
never fired, and the defect needed "a real graph" - prefill and decode
having different sequence lengths IS the trigger.

## Hypotheses eliminated, with evidence

Recorded so no one retries them.

1. **Nondeterministic magnitude = race in the tape path.** Reinterpreted:
   the jitter is the recycled-page contents past the buffer. Batched
   submissions vary allocation timing -> jittery band (~8e8). Per-node
   submission pins the timing -> the same page lands there every time ->
   constant 1.46e11 = exactly 34x2^32 = float32 bits 0x52100000 = packed
   f16 [33.0, 0.0], a position-id-shaped word (TapeLayerIsolation's
   bit-level read, arithmetic verified). A stale read of
   timing-dependent memory, not a race.
2. **Many-dispatches-per-command-buffer (decision tree, many layer).**
   Eliminated by measurement: `MLX_OMARCHY_TAPE_PER_NODE_SUBMIT` persisted
   on the M1 (constant 1.46e11 band, 5/5). Switch verified genuinely
   active on the tested build (submission-count dominance asserted; the
   red delta assertion at HEAD was a test artifact, fixed in `26e8085`).
3. **Driver drops an in-buffer dependency (barrier layer).** Eliminated:
   `MLX_OMARCHY_TAPE_FULL_BARRIERS` persisted on the M1. Intra-CB barriers
   were verified broad and correct beforehand; no minimal shader
   reproduction exists because there is no sixth shader miscompile here.
4. **The submission boundary itself (fence/ring slot).** Eliminated:
   corruption survived `ff4b05a` (per-submission ordering wait) and
   switch A.
5. **Allocator in-window aliasing / lifetime (resource layer).**
   Eliminated as the primary cause: `MLX_OMARCHY_TAPE_NO_REUSE` persisted
   on the M1 (familiar ~8-9e8 band) - consistent in hindsight, because
   the OOB is past the eval *input* buffers, which NO_REUSE does not
   touch. The allocator's lack of in-flight-GPU tracking remains a real
   residual exposure; `MLX_OMARCHY_POISON_FREED=1` (`6cc0c07`) is the
   standing tripwire for it.
6. **The trig gate's own unordered host read.** A real bug, fixed at
   `d8cb4f6` 09:26 (mapped read raced the gate's own submission) - but
   not the abort cause: the ff4b05a 14:52 and 7cf5e9f 16:45 aborts
   persisted with the same magnitudes after that fix.
7. **Stale wheel generation.** The morning silent-corruption reports
   (CJK fragments, `<|endoftext|>` repeats) were measured in venvs
   poisoned by a `@` direct-URL pin reinstalling a stale 06:06 wheel.
   Confirmed artifact, retracted in docs/differential-harness.md. The
   provisioning rule (hash-gate every venv) came from this.
8. **Broadcast Sigmoid as the corruption locus.** Wrong localisation
   (issued from Main, corrected by TapeLayerIsolation). The refusal was
   the stale-shape symptom for shapes that are not trailing-
   broadcastable; the corruption class behind it never needed a race.

## Reproduction and fix

`scripts/probe_shapeless_reuse.py` (in `13d83f7`) on llvmpipe, 4-bit
wheel at `7c3d6b4` (pre-fix): swiglu traced `[1,8,64]` evaluated at
`[1,36,64]` refuses with the exact M1 error naming the TRACED shape;
a longer chain silently returns NaN. With the fix: 4/4 cases clean, and
the stderr reuse notice proves the shapeless path ran.

The fix (`13d83f7`): the interpreter derives each node's output shape
from the eval-time inputs as the trailing broadcast of their shapes -
the same contract upstream applies to fused fragments - with Broadcast
nodes keeping their own rule. At the trace shape this equals
`node.shape()`, so exact-shape tapes are byte-for-byte unchanged. A
first-reuse stderr notice names traced and serving shapes.

Upstream-contract answer (assignment item 5): upstream's own backends
honour `compile_replace`'s recomputed shapes; the shapeless cache key
(ndim + dtype) is upstream contract, not a bug. Our interpreter ignored
the contract. Nothing to report upstream.

## Verification

Local (llvmpipe, x86 dev box):

| Check | Pre-fix `7c3d6b4` wheel | Post-fix `6cc0c07` wheel |
|---|---|---|
| probe_shapeless_reuse.py | exit 3: refusal + NaN mismatch | exit 0, notice fires |
| probe with MLX_OMARCHY_POISON_FREED=1 | n/a (no poison code) | exit 0 |
| realpath differential (France, template) | bitwise-clean (OOB-prefix artifact: zeros) | bitwise-clean, notice fires |
| omarchy_compiled_tape_tests | (11 cases at that commit) | 12/12 cases, 1185/1185 assertions; poison-armed run also 12/12 |

Hardware (jwm1, Honeykrisp; BenchQueueM1 TCF-1, logs
`jwm1:~/benchq/logs/tcf-*`):

- Wheel `mlx_omarchy-0.32.2.dev202609032217+6cc0c07-cp314-cp314-linux_aarch64.whl`,
  sha256 `1fab661e3e109b82134996dd88ecc5560318df9122ff563fce177747178f45a5`,
  provenance gate PASS (installed libmlx.so == wheel member).
- Unit probe: rc=0, 4/4 clean, reuse notice verbatim (tape-ran proof).
- France prompt with `MLX_OMARCHY_ALLOW_UNSAFE_COMPILE=1`, unset
  `MLX_DISABLE_COMPILE`: **25/25 runs rc=0 "Paris"** (5 baseline +
  20 acceptance), shapeless-reuse notice present in every run - the
  tape demonstrably executed in each.
- Differential: `probe_tape_eager.py` rc=0 (32 iterations, bitwise-clean,
  all variants); `differential_compile.py --mode realpath --steps 32`
  rc=0, "per-step logits and tokens match bitwise over 32 steps" - the
  broadcast-Sigmoid refusal is gone at `6cc0c07`.
- C++ batteries from source at `6cc0c07` with the override:
  `omarchy_compiled_tape_tests` 11/11 cases / 349 assertions including the
  new shapeless-reuse case on Honeykrisp; `omarchy_eq_math_tests` 7/7
  (116); `omarchy_runtime_tests` 25/25 (6250); stream-overlap assertion
  holds (`81613a1` fix).
- Poison regression leg: 5/5 "Paris" with `MLX_OMARCHY_POISON_FREED=1`,
  zero 123456789-signature aborts - no recycled-storage read served any
  run.

Deviations: graph/model differential modes deferred to the TCF-2 batch
(realpath is the load-bearing mode; the synthetic modes close the
deviation there). Protocol batch script halts at first failure; no
post-failure runs existed.

## Status

- Defect: root-caused, fixed, regression-guarded, proven on hardware.
- Re-enable: ordered by Main; CompiledFailClosed merges the branch and
  lands the re-enable (auto-eager hook + gate removal) with the
  retirement of `MLX_OMARCHY_ALLOW_UNSAFE_COMPILE` and the test/doc
  flips.
- Open on my side: TCF-2 compiled-vs-eager measurement (decode + prefill,
  pinned length, provenance wheel) - the number lands in the README when
  measured; graph/model differential modes to close the deviation.
