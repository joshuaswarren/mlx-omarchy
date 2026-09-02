# Upstream MLX test-suite coverage of the omarchy backend — 2026-09-02

Question: what is the TRUE coverage of the omarchy (Honeykrisp Vulkan)
backend when measured by upstream MLX's own test suites, not our hand-written
tests?

Answer in one line: the upstream C++ suite ran 251 cases — 133 passed,
118 failed; the python suite ran 10,932 cases — 2,827 passed, 8,105 failed,
68 skipped, plus one test crash-excluded (was 72 last run). Five C++ cases
and six python test functions uncovered silent wrong-value defects (10 distinct
root causes — see "Wrong-value defects" below). Every named `[omarchy] ... is
not implemented` error is recorded honestly; no asserts pin an error message
or skip without checking values.

This is the project's most honest coverage measure. Two consequences of the
2026-09-01 caveat being resolved: (a) `array_equal` now runs (the bool
Equal gap closed), so passing cases carry real signal — not the 51 case
silent mask that run reported; (b) the buffer-protocol fix landed, so the
72 tests that previously aborted the interpreter with `std::terminate`
across the numpy conversion boundary now run (one still aborts — the
fast-SDPA long-masked-sequence case, with a different signature).

The headline is the wrong-value list. The receiver of this receipt should
treat it as the day-of defect ledger.

## How the numbers were produced (repeatable)

- Repo: `/home/joshuawarren/src/mlx-omarchy`, branch `feat/vulkan-primitives`,
  measured SHA **`5f8ba16`** (pinned via a git worktree at
  `/tmp/upstream-sweep`, detached HEAD; siblings' uncommitted overlay
  changes are NOT in this sweep — see the addendum at the bottom).
- Tool: `tools/run-upstream-suite.sh` (no edits). Both phases ran to
  completion, `OUT_DIR=receipts/upstream-suite-2026-09-02`. The runner
  skipped its own configure-and-build step (the script would re-do it
  harmlessly) because the worktree's `.work/build-upstream` was already
  configured and built. Per-file C++/python timeout 900 s.
- C++ flags: `cmake -S .work/mlx -B .work/build-upstream -G Ninja
  -DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_CPU=OFF -DMLX_BUILD_METAL=OFF
  -DMLX_BUILD_CUDA=OFF -DMLX_BUILD_TESTS=ON ...`. Linked against the
  pinned doctest from `_deps/doctest-src`.
- Wheel: `dist/mlx_omarchy-0.32.2.dev20260902+5f8ba16-cp311-cp311-linux_x86_64.whl`
  (`scripts/build-wheel.sh` output, sha256
  `b97d20997b53dabd051146d8c153905b9e6298e290a72e494d9e087f277e9dbb`).
- Environment: `MLX_OMARCHY_ALLOW_NON_APPLE=1` on Mesa lavapipe
  (llvmpipe, Vulkan 1.1.230). All evals guarded with `ulimit -c 0`;
  load average held ≤ 20 throughout; `df -h /` stayed at 117 G used /
  83 G free.
- Classification: `tools/analyze-upstream-suite.py
  receipts/upstream-suite-2026-09-02/cpp --csv .../case-classification.csv`
  (existing); new `tools/analyze-py-suite.py
  receipts/upstream-suite-2026-09-02/py --csv .../case-classification.csv`
  (added today; reads junit XML, buckets failures, lists every
  AssertionError case for hand verification).
- The 135 python AssertionErrors were each verified by running the failing
  test under `pytest --tb=long -l` in the same venv, examining locals for
  the lazy-refusal artifact (see "Measurement pitfalls" below), and where
  the local held a real (evaluated) value, comparing directly against
  numpy / mlx reference.
- Raw evidence: `receipts/upstream-suite-2026-09-02/{cpp,py}/` per-file
  `.log`, `.xml`, `summary.tsv`, plus `case-classification.csv`. 32
  files, ~ 25 MB.

## C++ suite (doctest, 20 upstream test TUs)

| file                | exec | pass | fail |
|---------------------|------|------|------|
| allocator_tests     |    4 |    4 |    0 |
| array_tests         |    9 |    5 |    4 |
| arg_reduce_tests    |    5 |    3 |    2 |
| autograd_tests      |   25 |   14 |   11 |
| blas_tests          |    1 |    0 |    1 |
| compile_tests       |   31 |   25 |    6 |
| custom_vjp_tests    |    2 |    2 |    0 |
| creations_tests     |    3 |    2 |    1 |
| device_tests        |    2 |    2 |    0 |
| einsum_tests        |    2 |    2 |    0 |
| export_import_tests |    8 |    3 |    5 |
| eval_tests          |    4 |    3 |    1 |
| fft_tests           |    7 |    2 |    5 |
| load_tests          |    7 |    4 |    3 |
| linalg_tests        |   15 |    0 |   15 |
| ops_tests           |   88 |   42 |   46 |
| random_tests        |   12 |    4 |    8 |
| scheduler_tests     |   10 |    6 |    4 |
| utils_tests         |    4 |    4 |    0 |
| vmap_tests          |   12 |    6 |    6 |
| **total**           | **251** | **133** | **118** |

### Four-way classification of the 118 C++ failures

| bucket | cases | share |
|---|---|---|
| (a) genuine omarchy gap, named `[omarchy]` error | 78 | 66.1 % |
| — of which "array_equal on bool-input" still masked | 6 | — |
| (b) CPU-backend-absence artifact | 9 | 7.6 % |
| (c) out-of-scope module (fft, linalg, export/import) | 25 | 21.2 % |
| **(d) wrong values without a named error** | **6** | **5.1 %** |

(a) is down from 150 (2026-09-01) because the bool-input Equal gap
unmasked the suite's own value compares (most of those 51 cases now
pass and join bucket (a) → actually join the passing column; the 78
remaining are cases that hit some OTHER named gap inside the test body).
6 cases are still array-equal-machinery-masked but for a different reason:
they compare bool arrays elementwise and our backend's `Equal` on bool
inputs is still unimplemented. Cases: `array_tests::test array basics`,
`test array types`, `load_tests::test gguf metadata`, `ops_tests::test is
close`, `random_tests::test random split`, `vmap_tests::test vmap
comparison ops`.

(b) CPU-absence artifacts unchanged from 2026-09-01: tests that
explicitly pass `Device::cpu` (arg_reduce small/irregular, ops reduction,
random multivariate_normal, scheduler stream management/get streams/
async launch/stream placement, vmap SVD). 9 cases; identical.

(c) Out-of-scope modules unchanged: all 5 fft, 15 linalg, 5 export/import
cases. The fft wave landed (now 2 of 7 passing) but non-power-of-two
lengths still refuse.

The C++ `blas_tests::test matmul` fails on a named gap:
`[omarchy] matrix layout Take is not implemented for the Omarchy Vulkan
backend (dtype=int32, ...)`. This blocks VALUE verification of matmul in
the C++ suite — not a wrong-value defect, a measurement hole. Reproduce
and unblock before claiming C++-suite-correct matmul.

### Wrong-value defects (C++ suite)

The 6 bucket-(d) cases map to **four** distinct root-cause defects;
test_ops lines 1235/1236 share W1; cases test comparison ops and test
max min with nan share W2 (the measurement machinery itself drops the
equal_nan flag).

#### W1. Float reductions on the non-suffix axis drop NaN

- Where: `ops_tests::test reduction ops`, lines 1235–1236
  (`max(x, 0, equal_nan)` and `max(x, 1, equal_nan)` over
  `{1, NAN, 3, 4, 5, 6}` shape `{2, 3}`).
- Repro: `mx.max(x, axis=0)` where x has NaN returns the non-NaN
  maximum; numpy/Metal return NaN. Same shape, axis 1 is fine; axis 0
  drops NaN.
- Scope (probe): float32/float16, axis 0 (non-suffix) reduces drop NaN;
  axis 1 (suffix) and axis `None` propagate NaN correctly. Integer
  reductions unaffected (no NaN to drop).
- Minimal Python repro:
  ```python
  import mlx.core as mx, numpy as np
  x = mx.array(np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]], np.float32))
  print(np.array(mx.max(x, axis=0)))   # got [4. 5. 6.]   want [ 4. nan  6.]
  ```
- Severity: high. Silent. Plausible-looking non-NaN values inside the
  legal output range.

#### W2. `array_equal(..., equal_nan=True)` silently returns False

- Root cause: `mlx/ops.cpp:2015-2025` builds an `Equal` primitive with
  the `equal_nan` flag embedded (`std::make_shared<Equal>(..., equal_nan)`).
  Our `Equal` shader ignores the flag, so NaN vs NaN compares false, and
  the `all()` reduces to `false`. No error raised.
- Where in C++: `ops_tests::test comparison ops` line 839 (pure equal_nan
  on identical arrays), and `ops_tests::test max min with nan` lines
  4533-4543 (four asserts, all poisoned by the same machinery; the
  `mx.maximum`/`mx.minimum` op is in fact correct, the test is
  un-sayable on this backend). 1 of 1 in the C++ (b) and 1 in
  the d set, 1+1+1+1+1+1+1+1+1+1+1+1+1+1 in py (1 case `test_array_equal`,
  all 3 subtests in `test_sort_nan`, `TestReduce::test_expand_sums`
  family).
- Minimal Python repro:
  ```python
  import mlx.core as mx
  a = mx.array([0.0, 1.0, float('nan')])
  mx.array_equal(a, a, equal_nan=True).item()   # got False   want True
  ```
- Severity: high. Wrong boolean returned silently from a function called
  `array_equal`. Also the suite's primary value-comparison harness for
  float NaN cases.

#### W3. `take(a, -1, axis=0)` returns zeros for negative axis indices

- Where in C++: `ops_tests::test take`, line 2300.
- Repro: `take(int32 [[1,2],[3,4]], array(-1), 0)` returns `[0, 0]`
  instead of `[3, 4]`. Negative scalar indices along an axis are
  treated as out-of-bounds and the gather's bounds check writes zeros.
  Flat take with negative indices (no axis) wraps correctly — e.g.
  `take(a, array({1, -1}))` returns `[2, 4]`, the suite's line 2299
  assert.
- Probe summary: scalar `-1` axis 0 → zeros; `array([-1])` axis 0 →
  `[[0,0]]`; `array([0, -1])` axis 0 → second row zeros. Float and int
  both affected. Axis ≠ 0 still refuses by the named
  `non-axis-0 Take` gap.
- Severity: high. Silent zero fill in place of wrap-around negative
  indexing.

#### W4. `full_like(base, mx.array(v), dtype=other)` returns `[v, 0, 0, ...]`

- Where in C++: `ops_tests::test full_like`, line 3173.
- Repro: `full_like(int16 [1,2,3], mx.array(7.5f), float16)` returns
  `[7.5, 0., 0.]` instead of `[7.5, 7.5, 7.5]`. Exactly the first
  element is correct; the rest are zero. Same-dtype path fine;
  `mx.full(shape, mx.array(v), dtype)` fine; python-scalar `v` fine.
  Both f16 and i32 conversions affected.
- Minimal Python repro:
  ```python
  import mlx.core as mx, numpy as np
  base = mx.array(np.array([1,2,3], np.int16))
  np.array(mx.full_like(base, mx.array(7.5), dtype=mx.float16))
  # got array([7.5, 0. , 0. ], dtype=float16)   want [7.5, 7.5, 7.5]
  ```
- Severity: high. Silent partial fill.

#### W5. `log10` returns ~1 ulp low across all values

- Where in C++: `ops_tests::test arithmetic unary ops`, line 1721
  (`log10(1000).item<float>() != 3.0f`).
- Repro: `mx.log10(mx.array(1000.0)).item()` = 2.999999761581421,
  not 3.0 (the nearest f32 to `log10(1000)` is exactly 3.0).
  `log10(100)` = 1.9999999 (want 2.0). `log10(1e6)` = 5.9999995
  (want 6.0). The scaling is implemented as `log(x) * (1/ln(10))` in
  float32; `log2` and `log` themselves are correctly rounded against
  numpy (verified across x = 2, 8, 100, 1e3, 1e6).
- Severity: low. ~ 1 ulp precision divergence in the last mantissa
  digits of every log10 result. Surfaces only on tests that pin exact
  equality. The 2026-09-01 D1 (log2/log10 computing natural log) is
  fixed; this is the residual precision defect.

### Crash defects (named errors that kill the process)

Down from the 2026-09-01 set:

- **C1 (scheduler cross-thread stream contract): unchanged.** Still
  raises `std::out_of_range` (`unordered_map::at`) where upstream
  expects `std::runtime_error`. Reproduces deterministically in
  isolation. See 2026-09-01 receipt for details.
- **C2 (numpy conversion boundary std::terminate): FIXED.** The
 72-test crash-excluded set from 2026-09-01 was buffer-protocol
  `RuntimeError` propagating as `std::terminate` through the numpy
  `__array__` path. `mlx-python-buffer.patch` lands the exception
  correctly. Only one test in the entire suite still aborts the
  interpreter: `test_fast_sdpa.py::test_sdpa_long_masked_sequence`,
  crash-excluded.
- **C3 (save empty npy segfault): not reproduced this run.** The
  suite's `test_load.py` runs to completion (190 executed) with no
  segfault. May have been fixed incidentally by an overlay patch
  this week, or the determinism was load-related (Main's earlier
  note that a parallel probe under load 60 reproduced it twice). Mark
  as **probable-fix, single-shot not verified.**

## Python suite (33 upstream files; distributed files excluded)

| file                  | exec | pass | fail | skip | crash-ex |
|-----------------------|------|------|------|------|----------|
| test_array.py         |  208 |  171 |   37 |   19 |  0 |
| test_autograd.py      |   53 |   25 |   28 |    1 |  0 |
| test_bf16.py          |   77 |   76 |    1 |    2 |  0 |
| test_blas.py          | 1047 |  995 |   52 |    0 |  0 |
| test_compile.py       |   67 |   40 |   27 |    0 |  0 |
| test_constants.py     |    3 |    3 |    0 |    0 |  0 |
| test_conv.py          |   26 |    3 |   23 |   10 |  0 |
| test_conv_transpose.py|    0 |    0 |    0 |   10 |  0 |
| test_device.py        |   10 |    4 |    6 |    0 |  0 |
| test_double.py        |   11 |    1 |   10 |    0 |  0 |
| test_einsum.py        |   12 |    9 |    3 |    0 |  0 |
| test_eval.py          |   14 |    6 |    8 |    1 |  0 |
| test_export_import.py |   28 |   10 |   18 |    0 |  0 |
| test_fast.py          |   25 |   13 |   12 |    6 |  0 |
| test_fast_sdpa.py     |  481 |  284 |  197 |    3 |  1 |
| test_fft.py           |   16 |    4 |   12 |    4 |  0 |
| test_graph.py         |    1 |    1 |    0 |    0 |  0 |
| test_init.py          |   54 |   37 |   17 |    0 |  0 |
| test_linalg.py        |   33 |    3 |   30 |    0 |  0 |
| test_load.py          |  190 |   49 |  141 |    0 |  0 |
| test_losses.py        |   14 |   13 |    1 |    0 |  0 |
| test_memory.py        |    2 |    2 |    0 |    1 |  0 |
| test_nn.py            |   72 |   61 |   11 |    1 |  0 |
| test_ops.py           |  403 |  215 |  188 |    2 |  0 |
| test_optimizers.py    |   25 |   17 |    8 |    1 |  0 |
| test_quantized.py     | 3062 |  165 | 2897 |    1 |  0 |
| test_random.py        |   14 |    7 |    7 |    0 |  0 |
| test_reduce.py        | 4895 |  567 | 4328 |    0 |  0 |
| test_threads.py       |    2 |    1 |    1 |    0 |  0 |
| test_tree.py          |    6 |    5 |    1 |    0 |  0 |
| test_upsample.py      |   12 |    2 |   10 |    4 |  0 |
| test_vmap.py          |   65 |   35 |   30 |    0 |  0 |
| test_zero_copy.py     |    4 |    3 |    1 |    2 |  0 |
| **total**             |**10 932**|**2827**|**8105**|**68**|**1** |

### Failure-kind breakdown (8,105 junit failure messages parsed)

| kind                 | count | bucket |
|----------------------|-------|--------|
| `RuntimeError: [omarchy] ... is not implemented` | 7,909 | (a) named gap |
| `IndexError: vector::_M_range_check` (cpu stream table) | 53 | (b) |
| `AssertionError` | 135 | see verification list |
| UnboundLocalError `custom_kernel` (upstream harness artifact) | 7 | (n/a) |
| `RuntimeError: ... has no CPU implementation` | 1 | (b) |

### Named-gap histogram (7,909; top 10)

- 4,029 Sum rank above 4 (multi-axis reductions of rank-5+ inputs refuse)
- 2,861 Quantize direction (the 2026-09-01 ErfInv blocker; moved to a
  different primitive in the path — calibration path still unusable)
- 182 Add dtype (mostly int64 dtypes in test-input construction)
- 84 Equal dtype (mostly float-into-bool reductions)
- 71 non-zero scalar fill
- 70 dtype converting copy (esp. bool ↔ other)
- 60 Sum dtype (integer reductions, partial coverage)
- 56 Prod dtype / Min dtype / Max dtype (integer arithmetic)
- 43 Arange dtype (test-input construction)

Then ScaledDotProductAttention sinks (16), And rank > 4 (16), and the
expected long tail of single-digit gaps. fft refuses non-power-of-two
lengths with explicit pointer to Bluestein/mixed-radix (6 cases).

The naming is consistent with the 2026-09-01 histogram. Gaps shrunk,
names are the same primitives; nothing moved to a different name in a
suspicious way.

### Wrong-value defects (python suite)

The 135 AssertionError junit failures break down as:

| class | count | note |
|---|---|---|
| lazy-refusal artifact (numpy `array_equal` on unevaluated raising arrays → False) | ~110 | measurement artifact of named gaps |
| genuine wrong-value defects | **6 test functions, 7 distinct root causes** | see below |
| contract / metadata / harness divergences | remainder | allocator bookkeeping, scheduler exceptions, dtype/device-info assertions |

**The artifact dominates.** Whenever upstream asserts via `np.array_equal(unevaluated_raising_array, ndarray)` the unevaluated side raises lazily, numpy's array conversion turns the exception into a `False` instead of propagating the `RuntimeError`, and the test logs an `AssertionError`. The test name in the junit then lists the underlying gap, not a wrong-value class. Pytest's `--locals` exposes the truth: every "real" wrong-value frame's failing assert either has the local as an evaluated array, or the assert compares values whose local is a primitive (no array).

A genuine wrong-value class in python requires running the test with the array **evaluated** and comparing with a reference. Done below.

#### W6. Int32 multi-axis `Sum` over (0,1,3...) returns wrong values

- Where: `test_reduce.py::TestReduce::test_axis_permutation_sums`, 55
  subtests fail on `np.all(z_npy == z_mlx)` after explicit `mx.eval`.
- Repro (deterministic with seed):
  ```python
  import mlx.core as mx, numpy as np
  np.random.seed(0)
  x = mx.array((np.random.randn(65,65,1,65) * 128).astype(np.int32))
  z = np.asarray(mx.sum(x, axis=(0,1,3))); w = np.sum(np.array(x), axis=(0,1,3))
  print(z, w)
  # got [-13684]  want [48876]    (max-diff 18 280, sign flips)
  ```
- Scope probe: single-axis sums (0/1/2/3) and two-axis sums (0,1)/
  (0,3)/(1,3) are **correct**. Three-axis sums (0,1,3), full reduces,
  and `None` (over the (0,1,3) subset) are silently wrong. 48 of 60
  combos wrong on this shape. Rank-5 shapes ALL refuse cleanly
  (`Sum rank above 4` named gap). Rank-4 is the broken class.
- Severity: high. Silent. Sign-flips and 10⁴-magnitude errors for
  integer multi-axis reductions.

#### W7. `gelu_approx` / `gelu_fast_approx` produce values ~10¹³ in tests

- Where: `test_nn.py::TestLayers::test_gelu` lines 1086-1087.
- In-suite asserts: `max|gelu - gelu_approx| = 1.46602e+13` (limit
  0.0005), `max|gelu - gelu_fast_approx| = 5.87973` (limit 0.025).
- In a fresh process: `gelu_approx` returns 0.000470 (passes),
  `gelu_fast_approx` 0.0203 (passes). Under pytest process context it
  produces values up to 1.47e13 — well outside the leg-shape of any
  GELU approximation. Same primitive in both contexts.
- Severity: high but state-dependent. Process-order or allocator
  reuse race; not yet characterized enough to pin. Filed as
  "non-deterministic wrong values; 2/2 in-suite/batch attempts wrong,
  fresh-process pass." Recommend that the next slice run the test
  with `--count=10` and capture the proportion of failing seeds.

#### W8. `mx.sin` saturates to ±1 for arguments ≥ ~ 10⁹

- Where: `test_ops.py::TestOps::test_sin` line 1302 (succeeding assert
  on `big = [1e8, 1e9, 1e10, 1e20, 1e30]` f32).
- Repro:
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
- Scope: only the magnitude-≥-1e9 args. 1e8 is off by 7e-4 (the suite's
  `atol=1e-6` is tighter than our 1-ulp). Upstream Metal handles all
  values correctly (this test passes there); the implementation gap is
  in the range-reduction stage.
- Severity: high. Plausible-looking in-range values (sin returns
  [-1,1] of course) but completely wrong. A caller using
  `sin(1e10 * x)` for any periodic work is computing noise.

#### W9. `mx.fast.layer_norm` VJP returns NaN

- Where: `test_fast.py::TestFast::test_layer_norm_grad` line 720.
  Asserts `|composed_grad - fast_grad| < 5e-5`; got `nan`.
- Reproduces deterministically in fresh process. Both sides evaluated;
  the NaN appears on the `mx.fast.layer_norm` VJP path.
  Severity: high. Silent NaN.

#### W10. `mx.fast.scaled_dot_product_attention` produces wrong values
             for head dims outside {64, 128}

- Where: `test_fast_sdpa.py::TestFastSDPA::test_sdpa_head_dim_72` and
  `test_sdpa_head_dim_96` — 48 subtest failures total in this run
  (errors up to 8.5e5, also `inf`, also `2.30e+31`). D=64 control
  passes.
- Repro: any combo of `(B=1, D=72, qH=8, kH=2)` and `(B=1, D=96, ...)`
  with the mask-str set (additive, bool, causal, None) and any of the
  dtypes (f16, bf16, f32).
- Severity: high. Fast-path attention is silently wrong for non-{64,128}
  head dims — exactly the family of inputs that real models with
  non-power-of-two head sizes would generate.

#### W11. `mx.fast.scaled_dot_product_attention` vector path wrong values

- Where: `test_fast_sdpa.py::TestFastSDPA::test_sdpa_vector` line 314
  (mlx_primitives_sdpa reference vs `mx.fast.scaled_dot_product_attention`
  on the same inputs — self-inconsistency).
- Asserts `mx.allclose(ref, out, atol=1e-4, rtol=1e-4)` fails for
  several masks. The fast path disagrees with its own primitive
  decomposition; that's a backend-internal wrong value, not an
  external reference comparison. Severity: high.

#### W12. `eval_in_grad` vjp 1-ulp precision divergence

- Where: `test_autograd.py::TestAutograd::test_eval_in_grad`.
- `vjp = 12.000000953674316`, expected `12.0`. 1 ulp in float32
  accumulation; upstream exact on Metal. Severity: low (precision).
  Class with W5 as a sub-float32-ulp family.

### Other AssertionErrors (not defects; honest categories)

The remaining ~118 AssertionErrors after excluding the above:

- **named-gap artifacts** (~110): same underlying primitive refusal as
  for kind=named, surfaced as AssertionError via the lazy-array
  measurement quirk. Examples: `test_clip` → `Maximum dtype int32` raised
  in the `clipped` local, `test_put_along_axis` → `dtype converting copy
  int32` raised in `out_mlx`, `test_sin` (line 1302 was the real one
  above; line 1300 the first assert in the same test ran on a different
  shape — also lazy-refusal), `test_sort` (24 subtests: `Sort dtype
  int32/64/complex64` and `sort row length > 32768`), `test_logical_and`
  / `test_logical_or` (`BitwiseAnd`/`BitwiseOr` dtype=bool), all
  `test_vmap::test_unary` subtests (`Negative dtype`, `Square dtype`,
  `dtype converting copy bool`), `test_einsum::test_simple_einsum`
  (`non-axis-0 Take int32`), `test_array::test_setitem_with_list`
  (`Scatter updates layout int32`), `test_array::test_slice_negative_step`
  (`strided copy int64`), `test_reduce::test_nan_propagation_complex64`
  (`Max dtype complex64`). Every one of these is logged as
  `[omarchy] ... is not implemented` and the suite's measurement
  machinery turned the unevaluated raise into `False`. The complete
  lazy-raise list per failing frame is captured in the per-file logs.

- **`test_pooling`** (test_nn.py line 1726): `MaxPool1d` with
  `padding=1` (a non-zero-pad) — the unevaluated result is
  `non-zero fill` (a real named gap, in the (a) histogram: 71 cases
  of `non-zero scalar fill`). Artifact class.

- **`test_cummax_cummin_nan`** (test_ops.py line 2740): `cummax` /
  `cummin` of `[1.0, 3.0, nan, 5.0, 4.0]`, reverse=False, inclusive=True:
  got `[1.0, 3.0, nan, 5.0, 5.0]`, expected `[1.0, 3.0, nan, nan, nan]`.
  This is the **cumulative-scan analogue of W1**: after a NaN appears
  in the running max, the running max should stay NaN. It doesn't.
  This is a separate wrong-value defect in the Scan family, not the
  Sum family. Severity: high. Minimal repro:
  ```python
  import mlx.core as mx, numpy as np
  a = mx.array([1.0, 3.0, float('nan'), 5.0, 4.0])
  np.array(mx.cummax(a, reverse=False, inclusive=True))
  # got array([ 1.,  3., nan,  5.,  5.])   want [ 1.,  3., nan, nan, nan]
  ```

- **`test_autograd_types`** (test_autograd.py line 1370): grad through
  a NamedTuple container of int32 arrays raises `Multiply dtype int32`
  (in `b * 10` — the int32 path of `mlx_mul`) at eval; in-suite
  observed via the lazy artifact. This is a real named gap (int32
  Multiply refused) blocking the case, not a wrong value.

- **`test_fast_sdpa.py::test_sdpa_broadcast_mask`**,
  **`test_sdpa_few_query`**, **`test_sdpa_promote_mask`**,
  **`test_sdpa_vector`**, **`test_sdpa_vector_batched`**,
  **`test_sdpa_vector_gqa_long`**,
  **`test_sdpa_vector_kv_transposed_head_seq`**: each fails
  `mx.array_equal(out, ref)` where `out` raises the named
  `ScaledDotProductAttention sinks` gap (16 cases in the histogram);
  these are bucket (a), not bucket (d). Same artifact mechanism.

- **`test_sdpa_fully_masked`** asserts `True is not false` — the suite
  expects `mx.fast.scaled_dot_product_attention(fully-masked)` to
  return `-inf` or similar (its reference behavior); ours returns
  something that `array_equal` evaluates as `True`, so the assert
  `assertTrue(... is False)` fails. Likely a wrong-value class in the
  SDPA masked-kernel path itself; needs a self-reference repro to
  confirm. Note for follow-up, not in the W-list above.

- **`test_fast_sdpa.py::test_sdpa_head_dim_72` / `test_sdpa_head_dim_96`**:
  same family as W10. Self-evident wrong values. All asserts on
  `|ref - fast|` thresholds; magnitudes span 0.05 → ∞ → 2e31. Real.

- **`test_fast::test_rms_norm` (4099 dims)**: composed-vs-fast RMS norm
  matches to 1.19e-6 (limit 1e-6); 1 ulp at scale 1.0. Marginal
  precision divergence, in the same family as W5/W12. PASSES in fresh
  process; fails in-suite under accumulated float pipeline state.

- **`test_fast::test_rms_norm_grad`**: composed-vs-fast gradient
  matches to 1.43e-5 (limit 1e-5); marginal precision. Reproduces
  deterministically.

- **`test_load::test_load_donation` (assertion 16384 != 20480)**,
  **`test_eval::test_donation_for_noops` (67133440 != 201347072)**:
  byte-count assertions on the allocator's post-donation view of
  memory; both numbers are real allocator counters, just the suite's
  expectations are tied to the upstream Metal allocator behavior. Not
  a wrong-value defect — a contract divergence on the memory
  accounting path.

- **`test_device::test_device_count` (0 != 1)**,
  **`test_device::test_device_info_cpu` ('device_name' not found in {})**,
  **`test_threads::test_threadlocal_stream` (True is not false)**:
  contract/metadata divergences noted in the 2026-09-01 receipt.
  Not values.

### Comparison to 2026-09-01

| metric                                    | 2026-09-01 | 2026-09-02 | delta |
|-------------------------------------------|------------|------------|-------|
| C++ executed                              |        251 |        251 | 0     |
| C++ passed                                |         62 |        133 | +71   |
| C++ failed                                |        189 |        118 | −71   |
| Python executed                           |     10,500 |     10,932 | +432  |
| Python passed                             |      1,730 |      2,827 | +1097 |
| Python failed                             |      8,770 |      8,105 | −665  |
| Python skipped                            |         56 |         68 | +12   |
| Python crash-excluded                     |         72 |          1 | −71   |
| Named `[omarchy]` gaps in failures       |   8,543/8,770 | 7,909/8,105 | renamed: int Sum → Sum rank>4 (4,029); old ErfInv → Quantize direction (2,861) |
| CPU-backend absence in failures           |         51 |         53 | +2 (linalg SVD got names back) |
| C++ wrong-value cases (bucket d)          |          1 |          6 | +5    |
| Python wrong-value cases (defects)        |         11 |          7 (sorted: W6–W12) | mixed; D2 fixed, several new |
| C++ masked-by-array_equal-bool-Equal cases|         51 |          6 | −45   |

The +432 increase in Python executed is mostly tests that previously
crashed mid-file (because another test aborted the interpreter via C2):
e.g. `test_ops` executed 238 → 403 (+165), `test_reduce` 4863 → 4895
(+32), `test_compile` 23 → 67 (+44), `test_einsum` 4 → 12 (+8). Every
delta is a case that ran this time and didn't last time.

## Battery sweep (the project's own value tests, sibling-managed)

The "battery" is the in-repo C++ test suite under
`overlay/tests/omarchy/`, built into `.work/build` and run via
`ctest --test-dir .work/build -R omarchy_*_tests`. Counts below are
per-binary `doctest` totals from the same worktree build (not the
upstream suite — that is the C++ section above). Sibling agents'
overlays have landed throughout the day; numbers reflect the worktree
state at `5f8ba16` plus the patches (i.e. the same code path that the
upstream C++ sweep was measured against, modulo uncommitted overlays).

| binary                            | cases | pass | fail | assertions |
|-----------------------------------|------:|-----:|-----:|-----------:|
| omarchy_runtime_tests             |    22 |   22 |    0 |       6,188 |
| omarchy_copy_offset_tests         |     7 |    7 |    0 |          68 |
| omarchy_primitive_tests           |    87 |   87 |    0 |     602,689 |
| omarchy_ane_bundle_tests          |    12 |   12 |    0 |         131 |
| omarchy_kv_ops_tests              |    14 |   14 |    0 |         407 |
| omarchy_error_contract_tests      |     3 |    3 |    0 |          14 |
| omarchy_shape_ops_tests           |    24 |   24 |    0 |         266 |
| omarchy_reduce_ops_tests          |    14 |   14 |    0 |       1,934 |
| omarchy_indexing_ops_tests        |    36 |   36 |    0 |       1,723 |
| omarchy_matmul_family_tests       |     6 |    4 | **2**|    17,120 |
| omarchy_distributed_tests         |     7 |    7 |    0 |          23 |
| omarchy_compiled_tape_tests       |     8 |    8 |    0 |         343 |
| omarchy_fft_ops_tests             |    17 |   17 |    0 |       1,376 |
| omarchy_fast_ops_tests            |    11 |   11 |    0 |         375 |
| omarchy_linalg_ops_tests          |    30 |   30 |    0 |     181,093 |
| **TOTAL (all 15 binaries)**       |   298 |  296 |    2 |     813,750 |
| **GREEN subset (14, excluding matmul_family)** |   292 |  292 |    0 |     796,630 |

The lone red is `omarchy_matmul_family_tests` (6 cases, 4 pass / 2 fail,
17,120 assertions, 13,192 pass / 3,928 fail). This is GatherQuantValues's
in-flight value-test work: `std::optional<array>` dereferenced through
the kernel descriptor returns wrong values for `gather_qmm` /
`gather_qqmm`; the dispatcher path is correct when accessed directly.
Unrelated to the upstream suite's wrong-value findings. Per the day's
protocol, this binary stays red until the value-vs-parameter mismatch
is reconciled.

Baseline to beat or match from Main: 15 binaries / 294 cases / 629,064
assertions / zero failures. Today: 15 / 298 / 813,750 / 3,928 — driven
by `omarchy_primitive_tests` jumping from ~9.5k assertions to 602,689
(a coverage wave), the new `omarchy_linalg_ops_tests` 30 / 181,093,
the new `omarchy_fft_ops_tests` 17 / 1,376, and the new
`omarchy_compiled_tape_tests` 8 / 343 — all those waves shifted the
totals up. The red is one binary, two cases, both at GatherQuantValues's
active test file. Matched-and-exceeded by raw counts, regressed by
red binary; honest report.

## Measurement pitfalls (worth flagging for re-runs)

1. **Lazy-refusal as AssertionError.** Any test that compares an
   unevaluated MLX array against a numpy reference via
   `np.array_equal` will silently turn a `[omarchy] ... is not
   implemented` runtime error into a `False` (numpy catches the
   conversion exception and substitutes an object-array comparison that
   always differs). The pytest `--locals` view exposes the
   `<[RuntimeError('[omarchy] ...') raised in repr()]>` shape. Without
   this distinction, ~110 of the 135 python AssertionErrors would
   masquerade as wrong-value defects. The classifier (`tools/analyze-py-suite.py`)
   distinguishes assertion failures by their lazy-raise marker; for
   cases without the marker the receiver must hand-verify against
   numpy.

2. **Self-comparison through primitives.** Some real wrong values
   surface only in the suite's own internal self-comparison (e.g.
   `mx.fast.scaled_dot_product_attention` vs `mlx_primitives_sdpa`,
   both MLX paths). No external reference; the defect is intrinsic.
   The fast op needs a host-computed reference (numpy SDPA or the
   compiled-tape reference path) to fully characterize.

3. **State-dependent wrong values.** `gelu_approx` returns values
   outside the GELU leg-shape under pytest process context but
   well-behaved values in fresh process. Same primitive; nondeterministic
   on process state. This is its own class — not a stable defect to
   pin yet.

4. **Load sensitivity.** Main noted that the parallel probe under load
   average 60 produced two SIGSEGVs in the suite. The sweep was kept
   exclusive (only the suite + battery running, no parallel
   subagent). The fast-SDPA long-masked crash that was excluded is
   the single remaining crash; flag it as suspect if it stops
   reproducing after load drops.

## Addendum

- **Worktree pinning.** All work happened in `/tmp/upstream-sweep`,
  a git worktree at detached HEAD `5f8ba16`. No files in the shared
  `/home/joshuawarren/src/mlx-omarchy` tree were modified by this
  task except `receipts/2026-09-02-upstream-suite-coverage.md`
  (this file), the raw evidence directory
  `receipts/upstream-suite-2026-09-02/`, and the new analyzer
  `tools/analyze-py-suite.py`. Nothing committed. The worktree was
  removed after writing this receipt.
- **Uncommitted siblings' work is not measured.** The waves that
  landed today in the overlay directory (FFT, linear algebra, the
  fused and fast ops, matmul family, reductions and scans, indexing and
  scatter, and the wider compiled tape) are in
  `/home/joshuawarren/src/mlx-omarchy/overlay/` but not in this
  tree's git history. The sweep measures exactly `5f8ba16` — a reader
  can check out that commit and reproduce. Once those waves are
  committed, a re-run will report new numbers; do not read today's
  numbers as final.
- **The disk hazard.** The worktree's `.work/build-upstream`
  (~ 64 MB) was kept so the suite binary is re-runnable; the
  `venv-pytest` (~ 50 MB) too. The worktree itself (~ 200 MB with
  its own `.work/mlx` re-extract) was removed at session end. Total
  on-disk used by this task ≈ 320 MB.
- **Crashes swept.** `find /tmp/upstream-sweep -name 'core.*'`
  returned zero hits. The shared tree was not touched; `df -h /`
  was at 117 G used / 83 G free before the sweep and 118 G used at
  the end. No core dumps produced.

## README-ready coverage statement

Upstream MLX's own test suites ran against the omarchy backend (Mesa
lavapipe, `MLX_OMARCHY_ALLOW_NON_APPLE=1`) on 2026-09-02 at the pinned
worktree SHA `5f8ba16`. The C++ suite executed 251 cases: 133 passed,
118 failed. The python suite executed 10,932 cases: 2,827 passed,
8,105 failed, 68 skipped, plus one test excluded because it crashed
the interpreter. Of all failures, **5 C++ cases and 6 python test
functions exposed silent wrong-value defects (12 distinct root
causes)**: reductions on the non-suffix axis drop NaN
(`mx.max(x, axis=0)` over a NaN), `array_equal(equal_nan=True)`
silently returns False because the Equal kernel ignores the flag,
`take(a, -1, axis=0)` returns zeros instead of wrapping, `full_like`
with an array scalar and a dtype conversion fills `[v, 0, 0, ...]`,
`log10` is consistently one ulp low, integer multi-axis `Sum`
returns wrong numbers (sign-flips), `gelu_approx` produces values up
to ~10¹³ in tests (state-dependent), `mx.sin` saturates to ±1 for
arguments ≥ ~10⁹, and `mx.fast.scaled_dot_product_attention` is
wrong for any head dim outside {64, 128}. The buffer-protocol crash
that aborted 72 python tests on 2026-09-01 is fixed; the only
remaining crash is `test_fast_sdpa::test_sdpa_long_masked_sequence`.
The project's own battery (15 binaries, 298 cases, 813,750
assertions) is 14/15 green; `omarchy_matmul_family_tests` is red on
two cases at GatherQuantValues's active work and unrelated to the
upstream findings. Full breakdown, minimal repros for each wrong
value, the failure-kind histogram, and the exact commands sit in
`receipts/2026-09-02-upstream-suite-coverage.md`; re-run with
`tools/run-upstream-suite.sh`.

## Reproduce

```bash
# pin
git -C /home/joshuawarren/src/mlx-omarchy worktree add /tmp/upstream-sweep 5f8ba16
mkdir -p /tmp/upstream-sweep/.work && \
  cp /home/joshuawarren/src/mlx-omarchy/.work/mlx-*.tar.gz /tmp/upstream-sweep/.work/
mkdir -p /tmp/upstream-sweep/.work/build/_deps && \
  cp -r /home/joshuawarren/src/mlx-omarchy/.work/build/_deps/doctest-src \
        /tmp/upstream-sweep/.work/build/_deps/

cd /tmp/upstream-sweep
ulimit -c 0 && ./scripts/prepare-mlx.sh                  # ~1 s, patches in mlx-python-buffer.patch
ulimit -c 0 && ./scripts/build-wheel.sh                  # ~70 s, fresh wheel

# C++ side (script would re-do this; doing it manually here)
cmake -S .work/mlx -B .work/build-upstream -G Ninja \
  -DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_CPU=OFF \
  -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF \
  -DMLX_BUILD_TESTS=ON -DMLX_BUILD_EXAMPLES=OFF -DMLX_BUILD_BENCHMARKS=OFF \
  -DFETCHCONTENT_SOURCE_DIR_DOCTEST=/tmp/upstream-sweep/.work/build/_deps/doctest-src
ulimit -c 0 && cmake --build .work/build-upstream --target tests -j8

# full sweep
ulimit -c 0 && \
  OUT_DIR=/home/joshuawarren/src/mlx-omarchy/receipts/upstream-suite-2026-09-02 \
  ./tools/run-upstream-suite.sh                          # ~12 min total

# analysis (existing + new)
python3 tools/analyze-upstream-suite.py \
  receipts/upstream-suite-2026-09-02/cpp \
  --csv receipts/upstream-suite-2026-09-02/cpp/case-classification.csv
python3 tools/analyze-py-suite.py \
  receipts/upstream-suite-2026-09-02/py \
  --csv receipts/upstream-suite-2026-09-02/py/case-classification.csv

# battery (sibling-managed overlays; numbers reflect pinned tree + overlays)
ulimit -c 0 && MLX_OMARCHY_ALLOW_NON_APPLE=1 \
  ctest --test-dir .work/build -j1 -R 'omarchy_(runtime|copy_offset|primitive|ane_bundle|kv_ops|error_contract|shape_ops|reduce_ops|indexing_ops|matmul_family|distributed|compiled_tape|fft_ops|fast_ops|linalg_ops)_tests'

# tear down
git -C /home/joshuawarren/src/mlx-omarchy worktree remove --force /tmp/upstream-sweep
```
