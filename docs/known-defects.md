# Known defects, by release

This page is the project's defect ledger for silent wrong values and crashes. It covers v0.3.0-alpha.1, v0.3.0, and v0.3.1. Each entry states the affected versions, the symptom, the platform the defect was observed on, and the fix status. Upstream MLX's own test suites exposed the v0.3.0-alpha.1 list on 2026-09-02; per-case evidence sits in [receipts/2026-09-02-upstream-suite-coverage.md](../receipts/2026-09-02-upstream-suite-coverage.md). The M1 findings are recorded in [receipts/2026-09-02-m1-red-suites-root-cause.md](../receipts/2026-09-02-m1-red-suites-root-cause.md).

This project's contract is to refuse by name rather than return a wrong number. The entries below break that contract, which is why they outrank every coverage gap. Anything not on this page fails loudly with a named `[omarchy] ... is not implemented` error.

## Why platform matters here

Two of the worst v0.3.0 defects never appeared on a Linux development box. They are real-M1-only, and the full dev-box battery - 24 binaries, 407 cases, 828,139 assertions - was green the whole night they shipped. A Vulkan capability query, a shader miscompile, and a submit-thread ordering are all per-driver questions: llvmpipe, lavapipe, and Honeykrisp answer them differently. **A green run on a software driver is not proof about the Apple GPU, and this ledger now records where every defect was observed.** Anyone contributing: your llvmpipe battery passing is the start of verification on this project, not the end of it.

## Live in v0.3.1

These are open in the current release. Each fails silently or crashes, so watching for errors cannot catch them.

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

## A correction on the record: compiled tapes do not bypass the trig gate

Commit `959c7a0`'s message states: "One residual, stated rather than hidden: sin and cos inside compiled tapes bypass the eval_gpu gate and inherit no limit." **That claim is wrong for this tree, and this section retracts it.** A reader who finds the claim in the git log should find this retraction next to it.

Why it is wrong: this backend never lowers a compiled tape into a single fused shader. The tape is a per-node interpreter - `eval_compiled_tape` dispatches every node through its primitive's own `eval_gpu` (`overlay/mlx/backend/omarchy/compiled.cpp:146`) - and `Sin::eval_gpu` and `Cos::eval_gpu` carry the gate. The claim describes upstream's fused-tape model, and it reached the commit message from an investigation report without being run.

Evidence it does not reproduce, gathered on the built wheel: `mx.compile` around `mx.sin`, `mx.cos`, and multi-op chains through them refuses at an argument magnitude of 5e6 with the full named message - five variants, single-op and chained. A bf16 tripwire proves the tapes formed: the same shapes with a bfloat16 intermediate raise `[omarchy] Compiled tape bfloat16 is refused`, an error only the tape interpreter can produce, so the refusals came from inside a real tape, not from an eager fallback.

The general rule this leaves behind, for anyone contributing execution paths: **a path that runs a primitive without going through its `eval_gpu` must carry that primitive's gates.** The day a fused single-shader tape path lands, tape nodes stop reaching `eval_gpu`, and a fused path able to absorb Sin or Cos would silently reintroduce the exact hole this section disproves - only while fusion is enabled. Fusion work must either exclude trig from its fusable set or carry the magnitude gate into the fused shader, pinned by a test that runs both legs of its enable/disable toggle. Until such a path exists there is no hole; if it lands without the gate, this section's defect becomes live.

## Fixed in v0.3.1 (shipped live in v0.3.0)

v0.3.0, the full release, actively shipped these. A v0.3.0 wheel has all of them.

### Semaphore lifetime crash - every primitive could segfault

Affected: v0.3.0. Observed on: dev box, Mesa llvmpipe/lavapipe only - never observed on Apple hardware. Fixed in v0.3.1 (commit `150927b`).

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

This was the same defect the M1 investigation later measured properly: the range reduction degrades far below the magnitudes this table shows, and the real-target onset is at arguments of about 1e4 to 1e6, not 1e9. See the fixed-in-v0.3.1 entry above for the measured M1 numbers and the eager gate. Compiled tapes still bypass the gate; that is the live entry.

## What to do

On v0.3.0-alpha.1: treat every operation in the alpha section as untrusted. Upgrade - and note that three of the alpha entries, `gelu_approx`, the SDPA vector path, and the one-ulp vjp, are still live in v0.3.1; they sit in the live section above, so do not expect them to disappear.

On v0.3.0: the semaphore crash can kill any workload, and on a real M1 the scatter, LogicalAnd, select, and sin/cos defects return wrong values or refuse operations your dev box runs fine. Upgrade to v0.3.1.

On v0.3.1: the live section above is the complete list of known silent failures. Watch `gelu_approx` under test runners and bf16 compiled tapes in mlx-lm. Large-argument trig refuses by name, eagerly and inside `mx.compile` alike. Everything else refuses by name.

Named `[omarchy] ... is not implemented` errors remain the honest failure mode. The defects on this page are dangerous because they do not fail that way.
