# Known defects, by release

This page records silent wrong values and crashes in releases and development builds. Each entry states the observed version and platform, the symptom, and the fix status. Upstream MLX tests exposed the initial list on 2026-09-02; per-case evidence sits in [the suite receipt](../receipts/2026-09-02-upstream-suite-coverage.md). The M1 findings are recorded in [the hardware receipt](../receipts/2026-09-02-m1-red-suites-root-cause.md).

The backend must refuse unsupported operations by name rather than return a wrong number. Known silent failures take priority over coverage expansion; this ledger is not proof that unlisted paths are correct.

## Why platform matters here

Two of the worst v0.3.0 defects never appeared on a Linux development box. They are real-M1-only, and the full dev-box battery - 24 binaries, 407 cases, 828,139 assertions - was green the whole night they shipped. A Vulkan capability query, a shader miscompile, and a submit-thread ordering are all per-driver questions: llvmpipe, lavapipe, and Honeykrisp answer them differently. **A green run on a software driver is not proof about the Apple GPU, and this ledger now records where every defect was observed.** Anyone contributing: your llvmpipe battery passing is the start of verification on this project, not the end of it.

## Fixed in development

### Idle-stream events destroyed their semaphore while a submit still used it

Observed on: llvmpipe development host, in every release through v0.3.5
(the code path is platform-independent). Status: FIXED at `eb9571a`.

A GPU-stream `Event::signal` on an idle encoder claimed to signal from
the host but called `Device::signal_timeline`, a signal-only
`vkQueueSubmit`. `Event::wait` then observed the timeline value and the
event destructor destroyed the semaphore while the driver still
referenced it from that submission. The symptom was a timing-sensitive
SIGSEGV in the full reduction suite after about 2,600 passing assertions,
at the boundary after a named-error evaluation; GDB and ASan runs never
reproduced it. Vulkan validation pinned it: exactly three
`VUID-vkDestroySemaphore-semaphore-01137` errors at the three named-error
boundaries preceding the crash.

The fix routes that path through the existing `vkSignalSemaphore` host
signal, so no queue submission references the semaphore. Verification on
the integrated source: three plain reduction runs at 26/26 cases and
7019/7019 assertions each, one validation-layer run at the same counts
with zero VUID-01137 matches, and the runtime suite at 29/29 and
6256/6256 (`receipts/2026-09-05-reduction-lifetime-integrated.log`).
The validation layer still reports pre-existing LocalSizeId and
device-teardown leak diagnostics; those are not this defect.

### bf16 generation emits repetitive fragments

Observed on: real M1, Honeykrisp, development wheel built from `4f27136`.
Status: FIXED in the tested eager path at `2f54fcb`. The affected release range is not established; published wheels have not been qualified with this fix.
The RoPE synchronization guard was present; this is not the guard-removal
experiment described below.

Both backends used upstream MLX 0.32.2 as their source version, mlx-lm
0.31.3, model revision `56d07e766edd7159fbe12ed12d9cf114bf38bf1e`,
the prompt `Hi` (30 actual template tokens), eager execution
(`MLX_DISABLE_COMPILE=1`), temperature 0, seed 0, and four warmup tokens.
The harness suppressed EOS to capture exactly 32 tokens.

Linux started with token 978 (` spire`) and continued with fragments such
as `the blank spire`. Native MLX on an M1 Max started with token 9707
(`Hello`) and answered `Hello! How can I assist you today?` before EOS;
its later template tokens are an artifact of suppressing EOS. The first
difference is at zero-based index 0. The stable Linux digest
`635bc7f4bbaa48a4` proves repeatability, not correct generation; the native
digest is `7fc0f968789b1882`. Neither timing nor the cause is inferred
from this comparison. The native machine was contended and is not the
same chip as the Linux target.

For comparison, the 4-bit long128 run at model revision
`a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3` shared its first 20 tokens.
Native then selected ` carefully` and Linux selected ` review`; both
continued coherently, with different text. Their digests were
`254d73fd93164b98` and `4cc08910089477fd`, respectively. Token agreement
alone is not a general numerical correctness test.

The default bf16 RoPE path promoted its input to dense float32 but retained
the original view's strides and transpose classification. RoPE now derives
layout from the buffer it actually binds, after promotion, and keeps the
normalized array alive through dispatch. The existing copy path already
normalizes strided casts; no extra copy or shader change was needed.

On M1 at `2f54fcb`, whole-sequence and 29+1 chunked RoPE results agree
exactly for both Q and K. The ordinary eager model run now answers
`Hello! How can I assist you today?`, with every token through the first
EOS identical to native MLX. Its full EOS-suppressed 32-token stream has
digest `f26175202f3dabe9`; the first difference from native is index 14,
after EOS at index 9. This fixes the observed first-token corruption, not
general cross-backend token equality. The synchronization guard stays in
place, and compiled bf16 is not qualified by this eager check.

Evidence: [original full-token comparison](../receipts/2026-09-04-native-output-comparison.json) and [fixed-source M1 values and full-token comparison](../receipts/2026-09-04-bf16-rope-layout-m1.json).

## Live in v0.3.5

Open in the current release.

### bf16 decode still requires the RoPE synchronization guard

Observed on: real M1 (Honeykrisp), in the development experiment that
removed the bf16 queue drain. Status: OPEN, guard retained. The affected
release range has not been established.

Without the drain, the scalar RoPE offset contained unexpected words
such as `be450000 3f670000 3fc90000`, and the accuracy gate refused the
argument. Interpreted as float32, those words have zero low 16 bits and
are exactly representable in bf16. They do not identify the writer or
prove premature release. Ownership, aliasing, and producer ordering
remain candidate causes. The recorded 4-bit runs did not show this
failure; that is not proof that all f16/f32 paths are unaffected.

`rope_trig_gate` retains synchronization for bf16 output. The same change
also pinned the dense temporary used by `copy_gpu` for strided dtype
casts, but that fix did not resolve the bf16 observation without the
drain. Keep the guard until the relevant lifetime and readiness
contracts are established and ordinary numerical checks pass.

Evidence: [RoPE gate receipt](../receipts/2026-09-04-rope-gate-drain.md).


### 4-bit decode runs at 0.21 tok/s (v0.3.4 only)

Affected: v0.3.4 only; fixed in v0.3.5. Observed on: real M1 (Honeykrisp), on the published
aarch64 asset. Status: FIXED on main at 0535e62; v0.3.5 carries it.

The hang watchdog replaced the blocking semaphore wait with a sleep-then-read
loop, so every host wait on GPU completion cost a full 100 ms tick. About 48
waits per token gives 4.8 s/token: 4-bit decode 12.5 -> 0.21 tok/s, 60x. The
release notes' 12.52 was measured on the commit before the watchdog was
cherry-picked in; nobody measured the uploaded wheel. Do not use v0.3.4 for
decode; v0.3.3 is unaffected. Three-arm measurement, fix, and the release
rule it produced: [receipts/2026-09-04-v0.3.4-decode-regression.md](../receipts/2026-09-04-v0.3.4-decode-regression.md).

### A single large evaluation can wedge the GPU queue

Affected: v0.3.4, and every earlier release - the watchdog change in v0.3.4
altered how the failure is reported, not whether it happens. Observed on: real
M1 (Honeykrisp). Status: OPEN, cause unknown.

A single `mx.eval` over a full-sequence forward at 2,048 tokens wedges the
queue. The completion timeline counter stays frozen at 0 - no work retires -
and the submission watchdog correctly declares a hang after ten seconds of no
progress, failing by name with no CPU fallback. This is a genuine hang, not
slow work being misclassified: an earlier reading of this failure assumed the
ten-second bound was simply too short, and hardware testing refuted that.

Chunked work at the same or greater length is unaffected. `mlx_lm` chunks
prefill, so ordinary generation does not reach it - a 6,009-token context
completed normally in 183 seconds. What reaches it is a hand-written
full-sequence forward, which is why no user has reported it and no test caught
it. Evidence and the protocol that produced it:
[receipts/2026-09-04-hang-watchdog-hardware.md](../receipts/2026-09-04-hang-watchdog-hardware.md).

The named failure is not a clean, recoverable error: the wedge poisons the rest
of the process. The decode run immediately after a wedged evaluation measured
0.214 tok/s, roughly twenty times slow, from residual wedged-queue state. So a
process that has hit the wedge must be restarted before any further work in it
is trusted, and any number it produces afterwards is void.

Until the cause is found, evaluate long sequences in chunks rather than in one
operation, and restart the process if a wedge is hit.

## Live in v0.3.1

These are open in the current release. Each fails silently or crashes, so watching for errors cannot catch them.

### Compiled tapes returned wrong values on real Apple GPUs - root-caused, fixed, re-enabled

Affected: observed at commit `ff4b05a` on the mlx-lm decode path; the same
family was measured for bf16 at `fbdd5ed` and `5f8ba16` (see
`receipts/2026-09-02-m1-bf16-compiled-tape.md`). Observed on: real M1
(Honeykrisp). Not observed on llvmpipe, where the differential harness and
the compiled-tape battery match eager exactly.

The measurement that forced this entry: a greedy 4-bit
`Qwen2.5-0.5B-Instruct-4bit` run at `temp 0 seed 0` answered "The capital of
France is Paris." with `MLX_DISABLE_COMPILE=1` and returned `<|endoftext|>`
repeats, CJK fragments, and unrelated English with compilation at its
default - exit code 0, all 32 tokens, normal speed, nothing raised. Commit
`ff4b05a` (each submission waits on its stream's previous submission)
removed an earlier abort at the trigonometric domain gate on this path and
left the wrong answers behind, converting a loud failure into a silent one.
Full conditions and numbers: `receipts/2026-09-03-dispatcher-compile-and-column-replace.md`.
Later provenance work attributed one earlier corruption report to a stale
wheel generation, so whether the silent corruption still reproduces at
current main is unproven in both directions
([docs/differential-harness.md](differential-harness.md)).


**Why it happened.** The tape interpreter materialised every node output
at the node's traced shape. A shapeless compiled fragment serves every
input shape from one trace, so a decode call legally reused a
prefill-traced tape: nodes computed into prefill-sized outputs and read
past their eval-time input buffers, into recycled allocator pages. That
is why the wrong text looked like garbage from nowhere, why magnitudes
were impossible for f16, and why llvmpipe (reading zeros from the same
overruns) never showed it. Not a race; the submission-ordering and
affinity theories are retired. Full hypothesis elimination, the
upstream-contract answer, and the TCF-1 hardware record:
`receipts/2026-09-03-stale-shape-tape-corruption.md`.

**The fix.** Node shapes are derived at eval time from the eval inputs
(trailing-broadcast derivation, commit `13d83f7`), pinned by the
shapeless-reuse regression case (`650e324`), with a recycled-storage
detector (`MLX_OMARCHY_POISON_FREED`, `6cc0c07`). TCF-1 acceptance on
the M1: 25 of 25 greedy runs correct with proof the tapes executed, the
differential harness bitwise-clean, poison 5 of 5, batteries green.
Compilation runs by default again on every device class.

**What was retired with the fix.** The fail-closed device gate in
`eval_compiled_tape`, the discovery-time `disable_compile()` hook, and
the `MLX_OMARCHY_ALLOW_UNSAFE_COMPILE` override all existed to fence an
unpinned defect; the fix pins it, so they are removed rather than left
as switches that no longer switch anything. The protections that remain,
unchanged: the bf16 tape gate (a separate, still-live bf16 defect), the
trigonometric accuracy gate (tape nodes dispatch through their own
`eval_gpu`, which carries it), and the named-error contract for
unsupported tape ops. Compiled-versus-eager speed is not measured yet
(TCF-2); correctness is the proven part.

### `nn.gelu_approx` and `gelu_fast_approx` return values up to 1.47e13 under pytest process context

Affected: v0.3.0-alpha.1 through v0.3.1. Observed on: dev box, Mesa llvmpipe, upstream suite context only.

Upstream's `test_nn.py::TestLayers::test_gelu` measured these gaps at lines 1086 to 1087:

- `max|gelu - gelu_approx| = 1.46602e+13`, against a limit of 0.0005.
- `max|gelu - gelu_fast_approx| = 5.87973`, against a limit of 0.025.

The same primitives pass in a fresh process: 0.000470 and 0.0203. Under pytest process context they return values up to 1.47e13. The receipt classifies this as state-dependent and names a process-order or allocator-reuse race that is not pinned. Two of two in-suite attempts were wrong.

A fresh-process smoke test does not clear this defect. Do not trust `gelu_approx` or `gelu_fast_approx` under test runners.

### The fast SDPA vector path disagrees with its own decomposition

Affected: v0.3.0-alpha.1 through v0.3.1. Observed on: dev box, Mesa llvmpipe, upstream suite context only.

`mx.allclose(ref, out, atol=1e-4, rtol=1e-4)` fails for several masks. The case is `test_fast_sdpa.py::test_sdpa_vector` at line 314. `ref` is `mlx_primitives_sdpa` on the same inputs. The fast path contradicts the backend's own primitive decomposition, so the wrong value is backend-internal. No standalone repro exists yet; the case only fails inside the upstream suite.

The receipt also holds one unconfirmed suspicion in this family: `test_sdpa_fully_masked` expects a sentinel such as `-inf` and our result does not match. No standalone repro exists, so it stays a follow-up, not a defect entry.

### One vjp path is one ulp off

Affected: v0.3.0-alpha.1 through v0.3.1. Observed on: dev box, Mesa llvmpipe.

Upstream's `test_autograd.py::TestAutograd::test_eval_in_grad` got `vjp = 12.000000953674316`. The exact value is `12.0`. This is one ulp of float32 accumulation. Upstream Metal is exact here. Severity is low. It surfaces only where code pins exact equality.

### Boolean reductions (`mx.all` AND `mx.any`) returned wrong results once data crossed the first 32-bit word - fixed in v0.3.2

Affected: confirmed in the published v0.3.0 and v0.3.1 aarch64 wheels; v0.3.0 predates the byte-extraction work, so this is a long-standing gap, not a regression. Observed on: real M1 (Honeykrisp), fresh process, deterministic. On llvmpipe the same code is correct, which is why every dev-box battery stayed green.

```python
import mlx.core as mx, numpy as np
mx.all(mx.array(np.array([True] * 33))).item()   # False; must be True
```

Both `mx.all` and `mx.any` are affected, for whole-array reductions and axis reductions alike. BoolAllFix's hardware map: whole-array `mx.all` of all-True input is wrong at every size from 5 up; `mx.any` over an array False everywhere except one True at index 4 or beyond returns False; an axis reduction on a `(2, n)` shape fails from `n = 3`, because row 1 begins in the second word; and the per-position map at 33 elements shows positions 0-3 and 16-19 correct while every other position past the first word reads falsy. There is no clean size-based workaround - the failure is positional, not a simple count.

The cause is pinned: `reduce_general.comp`'s `load_truthy` uses a dynamic shift-then-mask byte extraction, which this driver miscompiles - the fifth confirmed site of the same dynamic byte-extraction family as the masked-scatter defect and the four sites fixed in `959c7a0`. `mx.logical_and` is green on the same buffers, which proves the input layout is fine and isolates the defect to the reduction's load. Workaround: `mx.logical_and` substitutes for some uses, or reduce on an explicit CPU stream; the dtype-converting `mx.sum(a.astype(mx.int32))` path refuses by name on the GPU (named gap).

**Fixed in v0.3.2** (commit `cf68e7d`): `load_truthy`'s dynamic byte extraction replaced with the proven constant-shift select chain - device-probed on the M1 rather than trusted by analogy, because `select.comp` had proven the macro form is context-sensitive on this driver. Verification on Honeykrisp: the boundary probe went from 85 of 366 checks failing to 0 of 366, `reduce_ops` from 21/22 to 22/22, and the full battery is green across two passes at 408 cases and 828,679 assertions. A user on a v0.3.0 or v0.3.1 wheel still needs the warning above.



## What the M1 verification reds turned out to be

The first M1 run of the v0.3.1 release candidate (tree at `959c7a0`) reported three findings. All three were real; none of them is a device defect. They are recorded because each one carries a lesson that outlives this release, and because a red suite that gets explained away instead of root-caused is how real defects get missed later.

### The NaN ordered comparison: the test's own reference was miscompiled

`omarchy_primitive_tests` failed one assertion in three of three M1 runs: an ordered comparison against NaN returned true where the host reference said false, at `test_primitives.cpp:5548`. IEEE says every ordered comparison with NaN is false. The device returned the correct answer; the reference was wrong, because the reference was computed into a `std::vector<bool>`.

`std::vector<bool>` is bit-packed, and every assignment goes through a read-modify-write proxy. That proxy is itself compiled code, and g++ 16.1.1 20260430 aarch64 miscompiles it. The isolated trigger, verified independently on jwm1: four interleaved `vector<bool>` writes in one loop, then `ne = {x != 1.0 ...}` over `{1, NaN, 3, NaN}` gives `ne[3] = false` for `NaN != 1.0` at `-O1` and `-O2` (`ne = {1,0,1,0}`, should be `{1,0,1,1}`), while `-O0` is correct. The same comparison written against plain bool arrays is correct at every optimisation level, and a single-comparison loop is correct too - which is what pins the bit-packed proxy, not the comparison, as the trigger.

The durable rule for anyone writing device-versus-host comparisons on aarch64: **a reference computed into `std::vector<bool>` is not a reference.** Keep references in plain arrays.

This is the fifth distinct miscompile this project has isolated, and the first that is not in the Vulkan driver: three Honeykrisp shader miscompiles, the context-dependent byte-extraction inversion above, and now a host-compiler bug in the test harness. The pattern behind all five is the same, and it is the sentence this ledger keeps earning: **on this platform, disagreement between device and host is not evidence about the device until the host side is verified independently.**

### The indexing refusals that stopped refusing: stale test expectations

`omarchy_indexing_ops_tests` failed three assertions that expected a named `[omarchy]` refusal and saw the operation succeed. The serious version of this finding would be silent CPU fallback: the CPU backend shipped in this same release, and an op succeeding where no GPU kernel exists is the one thing the contract forbids. It is not that. The three assertions sit inside `if (!capabilities.shader_atomic_float_add)` guards that still expect the old named refusal, and `959c7a0` turned those paths into working compare-exchange kernels - it updated `test_scatter_determinism`'s expectations and missed these. The runtime had no CPU device, and the values came back correct from the GPU FCAS path. Test bugs, now fixed in the test files; the guarded refusals are gone because the capability gap is gone.

### The rope divergence: the probe, not the primitive

A standalone probe reported `fast::rope` diverging from its host reference by 13.6 at position 12345, contradicting the `959c7a0` claim that rope matches its reference there. The probe hardcoded `offset=0` while computing its reference at the target position, so the two never computed the same function. With the offset set, rope is bit-exact at position 4 and within the documented trig band at 12345 and 100000 on Honeykrisp. The `959c7a0` claim stands.

## A correction on the record: compiled tapes do not bypass the trig gate

Commit `959c7a0`'s message states: "One residual, stated rather than hidden: sin and cos inside compiled tapes bypass the eval_gpu gate and inherit no limit." **That claim is wrong for this tree, and this section retracts it.** A reader who finds the claim in the git log should find this retraction next to it.

Why it is wrong: this backend never lowers a compiled tape into a single fused shader. The tape is a per-node interpreter - `eval_compiled_tape` dispatches every node through its primitive's own `eval_gpu` (`overlay/mlx/backend/omarchy/compiled.cpp:146`) - and `Sin::eval_gpu` and `Cos::eval_gpu` carry the gate. The claim describes upstream's fused-tape model, and it reached the commit message from an investigation report without being run.

Evidence it does not reproduce, gathered on the built wheel: `mx.compile` around `mx.sin`, `mx.cos`, and multi-op chains through them refuses at an argument magnitude of 5e6 with the full named message - five variants, single-op and chained. A bf16 tripwire proves the tapes formed: the same shapes with a bfloat16 intermediate raise `[omarchy] Compiled tape bfloat16 is refused`, an error only the tape interpreter can produce, so the refusals came from inside a real tape, not from an eager fallback.

The general rule this leaves behind, for anyone contributing execution paths: **a path that runs a primitive without going through its `eval_gpu` must carry that primitive's gates.** The day a fused single-shader tape path lands, tape nodes stop reaching `eval_gpu`, and a fused path able to absorb Sin or Cos would silently reintroduce the exact hole this section disproves - only while fusion is enabled. Fusion work must either exclude trig from its fusable set or carry the magnitude gate into the fused shader, pinned by a test that runs both legs of its enable/disable toggle. Until such a path exists there is no hole; if it lands without the gate, this section's defect becomes live.

## Fixed in v0.3.1 (shipped live in v0.3.0)

v0.3.0, the full release, actively shipped these. A v0.3.0 wheel has all of them.

### Semaphore lifetime crash - every primitive could segfault

Affected: v0.3.0. Observed on: dev box, Mesa llvmpipe/lavapipe only - the crash was never observed on Apple hardware. Fixed in v0.3.1 (commit `150927b`); the fix is verified on Honeykrisp, 50 of 50 crash-loop runs green with no signals.

Mesa's queue submit thread signals a submission's semaphores before `vk_queue_submit_cleanup` releases that submission's timeline sync points. Observing the timeline therefore does not prove the driver is finished with it. Our completion dispatcher took the signal at face value, dropped the submission keepalive, and destroyed the Event's timeline semaphore while the driver still held a reference. The submit thread then locked a freed mutex: `pthread_mutex_lock` on a freed mutex, from `vk_sync_timeline_point_release`, from `vk_queue_submit_cleanup`, from `vk_queue_submit_thread_func`.

The bug lives in the shared completion lifecycle, so every primitive shared it, forward and backward. It surfaced in an SDPA backward test only because that case interleaves the most events. Measured crash rate in the v0.3.0 build configuration: 6 of 50 runs of one test binary at `d0c6997`, 3 of 50 at `f449ed2`, on the dev box. After the fix: zero in 250 runs - 150 by the fixing agent, 100 verified independently - all on Mesa lavapipe. The validation layers report nothing, because observing a timeline and then destroying the semaphore is legal-looking at the API level. The fix is conservative: it only ever delays a free, holding one generation of temporaries slightly longer.

### Real-M1-only wrong values: bool scatter, LogicalAnd at 33 elements, select

Affected: v0.3.0. Observed on: real M1 (Honeykrisp) only. Invisible on llvmpipe. Fixed in v0.3.1 (commit `959c7a0`).

The fourth Honeykrisp miscompile family: shift-then-mask byte extraction with a data-dependent shift amount, the same class as the masked-scatter defect. It dropped bool scatter writes across word lanes, corrupted 13 of 33 bytes in a 33-element `LogicalAnd`, and picked the wrong operand in `select` when conditions were broadcast or strided. Four suites were red on the M1 and green on llvmpipe at the same commit; the dev-box battery never saw any of it.

The workaround is per-site and probe-pinned, and that is the finding worth reading. In `select.comp`'s packed-bool path the constant-shift select-chain form is WRONG and the original helper form is correct - the exact inverse of every other site. An eight-variant device probe pins it: macro form wrong at 5 of 17 positions, helper form wrong nowhere. Neither form is safe by default on this hardware, so every byte-extraction site is probed individually. Gates on the M1 after the fix: scatter 21/21, select 11/11, eq_math 6/6, primitive 86/86, 604,733 assertions, three repeated runs each.

### Real-M1-only float scatter refusals

Affected: v0.3.0. Observed on: real M1 (Honeykrisp). Fixed in v0.3.1 (commit `959c7a0`).

`VK_EXT_shader_atomic_float` is not advertised on the M1 at all, so float Scatter Sum and Prod refused by name on the real target while working on the dev box. Twelve refusals became working compare-exchange kernels; Prod's CAS path never needed the extension and is ungated. These failed loudly, not silently - a v0.3.0 M1 user got a named error, not a wrong number - but the real target refused twelve operations its own development box ran fine.

### `mx.sin` and `mx.cos` degrade above 1e5 and collapse from 1e6 on the M1

Affected: v0.3.0 (and v0.3.0-alpha.1, where the recorded symptom was saturation at larger magnitudes). Observed on: real M1 (Honeykrisp); on llvmpipe the built-in stays accurate far higher, which is why the alpha.1 entry pinned the onset at about 1e9. Fixed in v0.3.1 (commit `959c7a0`) for the eager path.

The M1 built-in's range reduction degrades measurably: error 4.5e-4 at an argument of 12345, 4.8e-3 at 123457, and total collapse from 1e6. The outputs stay in [-1, 1], so nothing looks wrong locally; any periodic computation on such arguments is computing noise. v0.3.1 refuses by name above an argument magnitude of 1e5 - one device-independent contract, measured against the real target:

- 1e3 < |v| <= 1e5: built-in, bounded at 5e-3 absolute (measured worst 4.8e-3).
- |v| > 1e5: named refusal, no value. The refusal names the driver's untrusted range reduction and the miscompiling Payne-Hanek fallback.

The limit is chosen for the consumer that matters: `fast::RoPE` composes exactly these calls, its largest term is the position itself, and 1e5 clears a 32k context with threefold margin. Rope at positions 12345 and 100000 matches its reference on the M1. A software Payne-Hanek reduction was tried first and discarded on evidence: on the M1 it returns -7.9e15 for sin(5e6), because its carry chain rides the same dynamic-indexing miscompile class above.

Inside `mx.compile` the same gate applies - see the correction section above, which retracts an earlier claim that compiled tapes bypassed it.

## Shipped in v0.3.0-alpha.1 - fixed in v0.3.0 unless marked otherwise

Release [v0.3.0-alpha.1](https://github.com/joshuaswarren/mlx-omarchy/releases/tag/v0.3.0-alpha.1) was cut on 2026-09-01. The wheels on its release page ship every defect in this section. Each one returns confidently wrong numbers and raises nothing. Unless a heading says otherwise, the entry is fixed in v0.3.0 with value tests at the exact upstream failing shapes, and the record is kept here because the alpha.1 wheels are still installed somewhere. Observed on: dev box, Mesa llvmpipe, via upstream's own suites on 2026-09-02.

### `mx.fast.scaled_dot_product_attention` with grouped-query attention - fixed in v0.3.0

First recorded as a head-dimension defect, because upstream's `test_sdpa_head_dim_72` and `test_sdpa_head_dim_96` were the failing tests. Root-causing found a broader and more serious cause: the defect is **grouped-query attention**, not head dimension. When the key and value heads are fewer than the query heads, the backend produced a 5-D result and then installed those 5-D strides on a 4-D output array. Head dim 64 and 128 appeared safe only because the tests that exercise them use equal head counts.

Grouped-query attention is standard in current language models - Llama, Qwen, and Mistral families all use it - so this affected mainstream inference, not an unusual head size. Errors reached 8.5e5, `inf`, and 2.30e+31. Every mask form was affected: additive, bool, causal, and None. All of float16, bfloat16, and float32. One failing shape: B=1, D=72, 8 query heads, 2 key heads. Upstream Metal passes the same tests. Equal-head-count attention computed correctly at every head dim tested, including 72 and 96.

Regression guard: `test_sdpa_norm_regression.cpp` runs GQA at the upstream failing shapes against host math.

### `mx.fast.layer_norm`'s weight gradient was wrong above 512 columns - fixed in v0.3.0

Upstream's `test_fast.py::TestFast::test_layer_norm_grad` pinned this at line 720. The fast gradient must match the composed gradient within 5e-5. The `mx.fast.layer_norm` VJP path returned `nan`.

Root-caused with a scope worth stating precisely: the weight-gradient kernel did not reset its accumulator per 256-column tile, so it summed the wrong columns. Rows of 512 columns or fewer were correct. At 1024 columns the result was `nan`. At 8192 columns the relative error reached 4.94. That threshold is why the kernel's own unit tests passed - they used 32 columns. A silent NaN gradient reaches every parameter it touches, and a loss curve keeps looking healthy while the model stops learning.

Regression guard: `test_sdpa_norm_regression.cpp` runs the VJP weight gradient at 1024 and 8192 columns.

### Float reductions over the first axis dropped NaN - fixed in v0.3.0

`mx.max(x, axis=0)` over an array containing NaN returned the non-NaN maximum. numpy and Metal return NaN. Float32 and float16 were affected. Axis 0, the non-suffix axis, dropped NaN; the suffix axis and whole-array reductions propagated NaN correctly. The returned values looked plausible and stayed inside the legal output range.

### `cummax` and `cummin` stopped carrying NaN after the first one - fixed in v0.3.0

Once a NaN entered a running max, the running max should stay NaN. It did not. The cumulative-scan analogue of the reduction defect above; it sat in the Scan family, not the Sum family. Severity was high.

### `array_equal(..., equal_nan=True)` returned False for equal arrays - fixed in v0.3.0

Upstream builds an `Equal` primitive with the `equal_nan` flag embedded (`mlx/ops.cpp:2015-2025`). Our `Equal` kernel ignored the flag. NaN compared unequal to NaN, and the trailing `all()` collapsed to `False`. Nothing raised. This function is also the comparison harness upstream's suite uses for float NaN cases, so the defect masked other measurements.

### `mx.sum` over three or more axes of a rank-4 int array returned wrong totals - fixed in v0.3.0

Deterministic repro from the receipt:

```python
import mlx.core as mx, numpy as np
np.random.seed(0)
x = mx.array((np.random.randn(65,65,1,65) * 128).astype(np.int32))
z = np.asarray(mx.sum(x, axis=(0,1,3))); w = np.sum(np.array(x), axis=(0,1,3))
print(z, w)
# got [-13684]  want [48876]    (max-diff 18 280, sign flips)
```

On the tested shape, single-axis sums were correct. The two-axis pairs (0,1), (0,3), and (1,3) were correct. Three-axis sums, full reduces, and `axis=None` were silently wrong - 48 of 60 combinations. Rank-4 int inputs were the broken class; rank-5 shapes refused cleanly with `Sum rank above 4`. Errors reached 1e4 in magnitude and flipped sign.

### `take` with a negative axis index filled zeros - fixed in v0.3.0

Negative indices along an axis were treated as out of bounds, and the gather's bounds check then wrote zeros. From the receipt: `take(int32 [[1,2],[3,4]], array(-1), 0)` returned `[0, 0]` instead of `[3, 4]`. A scalar `-1` at axis 0 returned zeros; `array([0, -1])` zeroed the second row. Floats and ints were both affected. Flat `take` with negative indices and no axis wrapped correctly, and axis values other than 0 refused by name.

### `full_like` with an array scalar and a dtype conversion filled only the first element - fixed in v0.3.0

Repro from the receipt:

```python
import mlx.core as mx, numpy as np
base = mx.array(np.array([1,2,3], np.int16))
np.array(mx.full_like(base, mx.array(7.5), dtype=mx.float16))
# got array([7.5, 0. , 0. ], dtype=float16)   want [7.5, 7.5, 7.5]
```

The first element converted correctly; the rest were zero. float16 and int32 conversions were both affected. Same-dtype `full_like` filled correctly, as did `mx.full` with an array scalar and any Python scalar value.

### `log10` was about one ulp low on every value - fixed in v0.3.0

`mx.log10(1000.0)` returned `2.999999761581421`, not `3.0`. The nearest float32 to `log10(1000)` is exactly `3.0`. `log10(100)` returned `1.9999999`, and `log10(1e6)` returned `5.9999995`. The cause was float32 scaling by `1/ln(10)`. Fixed with a double-double-corrected implementation that returns the exact integer for every f32-exact power of ten and agrees with numpy float32 elsewhere; pinned by `W5` cases in `test_eq_math.cpp`. `mx.log` and `mx.log2` were always correctly rounded.

### `mx.sin` saturated to plus or minus 1 for arguments of about 1e9 and above - superseded by the M1 entry

Repro from the receipt, on the dev box:

```python
big = np.array([1e8, 1e9, 1e10, 1e20, 1e30], np.float32)
for o, w, v in zip(np.array(mx.sin(mx.array(big))), np.sin(big), big):
    print(f"sin({v:g}): got {o:.7f}  numpy {w:.7f}  diff {abs(o-w):.3g}")
# sin(1e+08): got 0.9308981  numpy 0.9316390  diff 0.000741
# sin(1e+09): got 1.0000000  numpy 0.5458434  diff 0.454
# sin(1e+10): got -1.0000000 numpy -0.4875060  diff 0.512
# sin(1e+20): got -1.0000000 numpy  0.6565767  diff 1.66
# sin(1e+30): got -1.0000000 numpy -0.7911634  diff 0.209
```

This was the same defect the M1 investigation later measured properly: the range reduction degrades far below the magnitudes this table shows, and the real-target onset is at arguments of about 1e4 to 1e6, not 1e9. See the fixed-in-v0.3.1 entry above for the measured M1 numbers and the eager gate. An earlier version of this page, and the `959c7a0` commit message, called compiled tapes a live bypass; the correction section above retracts that with evidence.

## What to do

On v0.3.0-alpha.1: treat every operation in the alpha section as untrusted. Upgrade - and note that three of the alpha entries, `gelu_approx`, the SDPA vector path, and the one-ulp vjp, are still live in v0.3.1; they sit in the live section above, so do not expect them to disappear.

On v0.3.0: the semaphore crash can kill any workload, and on a real M1 the scatter, LogicalAnd, select, and sin/cos defects return wrong values or refuse operations your dev box runs fine. Upgrade to v0.3.1.

On v0.3.1, published: the release ships the boolean-reduction defect above on Apple Silicon - both `mx.all` and `mx.any`, positional past the first word, disclosed in a prominent warning at the top of the release page - and a verified fix ships in v0.3.2. The three M1 verification reds resolved to test-side causes before the tag landed ("What the M1 verification reds turned out to be" above). The long-standing live entries remain: `gelu_approx` under test runners and bf16 compiled tapes in mlx-lm; compiled tapes on real Apple GPUs ran behind the auto-eager gate and the fail-closed backstop; the root-cause fix (13d83f7, first live entry above) re-enables them and retires both, so that exposure is closed at the source rather than fenced. Large-argument trig refuses by name, eagerly and inside `mx.compile` alike.


Named `[omarchy] ... is not implemented` errors remain the honest failure mode. The defects on this page are dangerous because they do not fail that way.

## f16 attention score cap (sdpa-f16-scores branch, unreleased)

`fast::scaled_dot_product_attention` on float16 inputs stores attention scores in f16 with f32 accumulation inside the kernel, so scaled scores above 65504 saturate to inf and softmax turns the row into NaN, while the f32/bf16 composition stays finite: the trigger is extreme but valid f16 activations (an untuned fine-tune reaches it; normalised models do not). The additive causal/padding mask uses the f16 finite maximum (-65504), not -inf, so fully masked rows stay defined and match the f32 path. This matches upstream Metal's f16 SDPA, which has the same storage cap; run f32/bf16 if you need overflow-immune attention. Reproduction and tolerance evidence: [receipt](receipts/2026-09-04-sdpa-f16-scores-rework.md) and `scripts/sdpa_equivalence.py` (fully-masked-row and overflow-boundary cases).
