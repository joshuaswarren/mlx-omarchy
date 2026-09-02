# Upstream MLX test-suite coverage of the omarchy backend — 2026-09-01

Question: what is the TRUE coverage of the omarchy (Honeykrisp Vulkan) backend
when measured by upstream MLX's own test suites, not our hand-written tests?

Answer in one line: the upstream C++ suite ran 251 test cases — 62 passed,
189 failed — and exactly ONE C++ case failed on wrong values (log2/log10
compute the natural log; a second python-only wrong-value defect, D2, is
below). Everything else failed on a named `[omarchy] ... is not implemented`
error, on CPU-backend absence, on out-of-scope modules, or on a
process-killing crash. Python numbers below.

## How the numbers were produced (repeatable)

- Repo: /home/joshuawarren/src/mlx-omarchy, branch feat/vulkan-primitives,
  HEAD fbdd5ed.
- C++: `.work/build-upstream` configured with
  `-DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_CPU=OFF -DMLX_BUILD_METAL=OFF
  -DMLX_BUILD_CUDA=OFF -DMLX_BUILD_TESTS=ON`. The upstream `tests` target
  built clean (248 ninja steps). doctest v2.4.12. Each of the 20 upstream
  test TUs ran separately via `-sf='*<file>'`, XML reporter, 900 s per-file
  timeout. `MLX_OMARCHY_ALLOW_NON_APPLE=1` on Mesa lavapipe (llvmpipe,
  Vulkan 1.1.230).
- Python: dist/mlx_omarchy-0.32.2.dev20260901+fbdd5ed-cp311-cp311 wheel
  (scripts/build-wheel.sh output, built 2026-09-01 14:40 from the same HEAD;
  the working tree's overlay/compiled.cpp gained the sibling bf16-tape
  refusal at 16:07, which the wheel does NOT contain — py results reflect
  the 14:40 wheel). Fresh venv, wheel + numpy 2.4.6 + pytest 9.1.1, each
  upstream python test file run separately, junit XML per file, 900 s
  per-file timeout. Files that hard-crash the interpreter are re-run with
  the crashing tests excluded (pytest -k) until the rest completes;
  crash-excluded tests are listed per file — they are crashes, not failures.
- Rerun everything: `tools/run-upstream-suite.sh` (both phases; the script
  reconfigures and rebuilds .work/build-upstream if missing), then
  `tools/analyze-upstream-suite.py <out>/cpp --csv
  <out>/cpp/case-classification.csv`. Raw evidence:
  receipts/upstream-suite-2026-09-01/{cpp,py}/ (per-file .log, .xml,
  summary.tsv).

## C++ suite (doctest, 20 upstream test TUs)

| file | executed | passed | failed |
|---|---|---|---|
| allocator_tests | 4 | 4 | 0 |
| arg_reduce_tests | 5 | 3 | 2 |
| array_tests | 9 | 4 | 5 |
| autograd_tests | 25 | 3 | 22 |
| blas_tests | 1 | 0 | 1 |
| compile_tests | 31 | 13 | 18 |
| creations_tests | 3 | 1 | 2 |
| custom_vjp_tests | 2 | 2 | 0 |
| device_tests | 2 | 2 | 0 |
| einsum_tests | 2 | 1 | 1 |
| eval_tests | 4 | 1 | 3 |
| export_import_tests | 8 | 0 | 8 |
| fft_tests | 7 | 1 | 6 |
| linalg_tests | 15 | 0 | 15 |
| load_tests | 7 | 1 | 6 |
| ops_tests | 88 | 16 | 72 |
| random_tests | 12 | 1 | 11 |
| scheduler_tests | 10 | 3 | 7 |
| utils_tests | 4 | 4 | 0 |
| vmap_tests | 12 | 2 | 10 |
| **total** | **251** | **62** | **189** |

scheduler_tests note: `test access stream in other thread` aborts the whole
process (SIGABRT), so the binary died mid-file. The remaining 7 scheduler
cases were recovered with a separate run excluding that case (2 passed, 5
failed). Numbers above include the recovery run.

### Four-way classification of the 189 C++ failures

| bucket | cases | share |
|---|---|---|
| (a) genuine omarchy gap, named `[omarchy]` error | 150 | 79.4% |
| — of which masked by the array_equal machinery | 51 | — |
| (b) CPU-backend-absence artifact | 9 | 4.8% |
| (c) out-of-scope module (fft, linalg, export/import) | 29 | 15.3% |
| (d) wrong values, no named error | 1 | 0.5% |

Per-case evidence: cpp/case-classification.csv (file, case, category, flag,
first error).

(a) Dominant named gaps, by failed-assert count across the suite:

And dtype=bool 335 (array_equal machinery, below), Abs 86, Scatter 42,
dtype converting copy 42 (complex64/bool/int64), non-zero scalar fill 36,
ErfInv 23, Equal dtype=bool 17, NotEqual(bool) 16, non-suffix Sum 15,
matrix layout Take 15, Erf 15, Sign 13, Add dtype (int32 scalar) 13,
Square dtype (int32) 13, non-axis-0 Take 13, Scan reduce 12, negative
stride copy 12, Power 11, RandomBits width 11, FFT 11, Less(bool) 11,
Select 10, reverse Scan 9, dynamic slice offset 9, DivMod 9,
GreaterEqual(bool) 8, LogAddExp 8, SliceUpdate reduce 7, Greater(bool) 6,
Prod 6, Max dtype (uint32) 6, Floor 6, LessEqual(bool) 5, non-suffix Max 5,
Select layout 5, Or dtype 5, Min dtype 5, Log1p 5, ConvertFP8 5, Ceil 4,
non-suffix Scan 4, non-2D Convolution 4, Conjugate 3, Expm1 3,
MaskedScatter 3, Arange dtype (uint32) 3, GatherAxis 2, grouped
Convolution 2, BitwiseBinary (int8) 2, Partition 2, plus single-assert
gaps (Compiled tape op Sin/Greater/Erf, ArgPartition, LogicalNot,
LogicalAnd, ScatterAxis, Remainder, non-suffix ArgMax, Round, Negative,
and others).

**The array_equal mask (important caveat).** Upstream's C++ tests verify
values through `array_equal(a, b).item<bool>()`, which lowers to Equal then
an And-reduce on bool. The omarchy backend implements neither And on bool
nor Equal on bool, so the comparison itself throws
`[omarchy] And dtype is not implemented ... (dtype=bool, shape=[1,1])` even
when the values underneath are correct. 51 failing cases and 335 failed
asserts are this measurement artifact: the op under test may be correct,
but the suite cannot say. This includes blas_tests::test matmul — matmul
VALUES are not verified by the C++ sweep. The un-masking fix is small
(bool And, bool Equal) and would light up a large fraction of the suite.

(b) CPU-absence artifacts — tests that explicitly pass Device::cpu:

- arg_reduce_tests::test arg reduce small, test arg reduce irregular strides
  (ArgReduce on default_stream(Device::cpu), arg_reduce_tests.cpp:35)
- ops_tests::test reduction ops (sum(x, Device::cpu), ops_tests.cpp:1097+)
- random_tests::test random multivariate_normal (cholesky -> linalg -> cpu)
- scheduler_tests::test stream management, test get streams, test
  asynchronous launch, test stream placement (default_stream(Device::cpu),
  scheduler_tests.cpp:22,141,170,200)
- vmap_tests::test vmap SVD (SVD -> linalg -> cpu; also (c) by module)

Error in every case: `vector::_M_range_check: __n (which is 0) >=
this->size() (which is 0)` — an empty stream table indexed for the cpu
device instead of a clean `std::invalid_argument`.

(c) Out-of-scope modules (roadmap excludes distributed, export/import,
fft, linalg): all 8 export_import cases; all 6 failing fft cases
(`[omarchy] FFT is not implemented ...`); all 15 linalg cases (11 reach
cpu through LAPACK — same range_check as (b); norm cases also hit
Square/Abs gaps on gpu).

### (d) Wrong-value defects — individual list

**D1. `log2` and `log10` compute the NATURAL log. No named error; values
are silently wrong.**

- Found in: ops_tests::test arithmetic unary ops (ops_tests.cpp:1709 and
  :1721). doctest expansions: `log2(x).item<float>()` -> 6.93147 where
  10.0 expected; `log10(x).item<float>()` -> 6.90776 where 3.0 expected.
- C++ repro (linked against .work/build-upstream/libmlx.a, llvmpipe,
  MLX_OMARCHY_ALLOW_NON_APPLE=1):
  - `log2(1024)` = 6.931472 (expected 10)
  - `log10(1000)` = 6.907755 (expected 3)
  - `log(1000)` = 6.907755 (correct — Log is right; the log2/log10
    kernels miss the 1/ln(base) scaling)
- Python repro (venv-pytest, wheel above):
  `mx.log2(mx.array(1024.0)).item()` = 6.931471824645996;
  `mx.log10(mx.array(1000.0)).item()` = 6.907755374908447.
- Impact: silent wrong numbers for any log2/log10 user. Not caught by our
  generated compatibility matrix. Root cause: upstream log2/log10 are the
  Log primitive with Log::Base two/ten (mlx/ops.cpp:3370,3382); the
  backend maps OMARCHY_UNARY(Log, LogOperation) (primitives.cpp:1220) and
  the shader does `value = log(lhs)` (shaders/elementwise.comp:137) — the
  base never reaches the kernel.

**D2. `mx.sum` over an expanded (broadcast) axis returns silently wrong
values. No named error for the failing layouts; 11 upstream python subtests
fail on values.**

- Found in: test_reduce.py::test_expand_sums (11 AssertionError subTests).
- Python repro (venv-pytest, seed 7):
  `x = mx.array(np.random.randn(5,1,5,1,5,1).astype(np.float32))`,
  `y = mx.broadcast_to(x, (5,5,5,1,5,1))`, `mx.sum(y, axis=3)` compared to
  `np.sum(broadcast, axis=3)`: 470/625 elements differ, max diff 3.45e-3
  vs atol 1e-4, no NaN. Failing combos include axis (3,), (5,), (3,5).
  Deterministic.
- Note: other expanded layouts ARE refused cleanly — e.g. sum over axis 1
  of a (5,5,5) broadcast raises
  `[omarchy] non-contiguous Sum is not implemented` — so the guard exists
  but does not catch the layouts above, which compute wrong numbers instead.

### Non-value contract divergences (NOT (d); listed for honesty)

Materialized views keep full row-contiguous metadata where upstream keeps
view flags:

- ops_tests::test slice — 6 flag asserts (`!out.flags().row_contiguous`
  expands false; ops_tests.cpp:276,288,289,292,298,299) and 5 data_size
  asserts (upstream slices keep the base buffer's data_size; ours compact —
  ops_tests.cpp:314,327,331,336,346).
- array_tests::test array metadata — same flags/data_size divergence
  (array_tests.cpp:497,498,499,505,507).
- ops_tests::test divmod — `out[0].siblings().empty()` is false where
  upstream divmod shares buffers (ops_tests.cpp:3566,3567).

Values are correct in all of these; view metadata semantics differ.

### Crash defects (named errors that kill the process; not silent)

- C1. scheduler_tests::test access stream in other thread: evaluating a
  stream from a non-owning thread must throw `std::runtime_error` (the
  test catches exactly that); the omarchy scheduler throws
  `std::out_of_range` (`unordered_map::at`) instead. It escapes the catch,
  terminates, and aborts the process (doctest recorded SIGABRT). The same
  `unordered_map::at` family hits scheduler_tests::test thread unsafe
  stream.
- C2. Python binding: a backend exception raised while numpy reads an
  array through the buffer/`__array__` path crosses a non-pybind boundary
  and calls std::terminate — the whole interpreter dies
  (`Fatal Python error: Aborted`). Repro:
  `np.array(mx.abs(mx.array([1.0, -2.0])))` aborts (rc 134), while `.item()`
  paths translate the same `[omarchy] Abs is not implemented` error into a
  clean RuntimeError. This defect truncated most python files mid-run.
- C3. `mx.save('<file>.npy', mx.zeros((0,)))` segfaults (rc 139). Repro'd
  standalone in the venv. Truncates upstream test_load.py.

### Python results (33 upstream files; distributed files excluded by scope)

| file | executed | passed | failed | skipped | crash-excluded |
|---|---|---|---|---|---|
| test_array.py | 200 | 164 | 36 | 18 | 6 |
| test_autograd.py | 53 | 14 | 39 | 1 | 0 |
| test_bf16.py | 35 | 34 | 1 | 2 | 2 |
| test_blas.py | 970 | 955 | 15 | 0 | 8 |
| test_compile.py | 23 | 5 | 18 | 0 | 1 |
| test_constants.py | 3 | 2 | 1 | 0 | 0 |
| test_conv.py | 10 | 1 | 9 | 3 | 1 |
| test_conv_transpose.py | 0 | 0 | 0 | 10 | 0 |
| test_device.py | 10 | 4 | 6 | 0 | 0 |
| test_double.py | 11 | 1 | 10 | 0 | 0 |
| test_einsum.py | 4 | 4 | 0 | 0 | 7 |
| test_eval.py | 14 | 4 | 10 | 1 | 0 |
| test_export_import.py | 28 | 7 | 21 | 0 | 0 |
| test_fast.py | 25 | 3 | 22 | 6 | 0 |
| test_fast_sdpa.py | 486 | 7 | 479 | 3 | 0 |
| test_fft.py | 0 | 0 | 0 | 0 | 1 |
| test_graph.py | 1 | 1 | 0 | 0 | 0 |
| test_init.py | 54 | 37 | 17 | 0 | 0 |
| test_linalg.py | 15 | 0 | 15 | 0 | 3 |
| test_load.py | 185 | 14 | 171 | 0 | 1 |
| test_losses.py | 14 | 2 | 12 | 0 | 0 |
| test_memory.py | 2 | 2 | 0 | 1 | 0 |
| test_nn.py | 66 | 35 | 31 | 1 | 5 |
| test_ops.py | 238 | 85 | 153 | 2 | 34 |
| test_optimizers.py | 25 | 7 | 18 | 1 | 0 |
| test_quantized.py | 3062 | 165 | 2897 | 1 | 0 |
| test_random.py | 14 | 2 | 12 | 0 | 0 |
| test_reduce.py | 4863 | 156 | 4707 | 0 | 3 |
| test_threads.py | 2 | 1 | 1 | 0 | 0 |
| test_tree.py | 6 | 4 | 2 | 0 | 0 |
| test_upsample.py | 12 | 2 | 10 | 4 | 0 |
| test_vmap.py | 65 | 9 | 56 | 0 | 0 |
| test_zero_copy.py | 4 | 3 | 1 | 2 | 0 |
| **total** | **10500** | **1730** | **8770** | **56** | **72** |

- test_ops.py ran under an extended 60-attempt recovery (the in-suite cap
  was 15 at run time and exhausted first); see py/test_ops.recovered.txt.
  34 of its tests crash the interpreter and are excluded from the executed
  count.
- Total outcomes: 10500 executed + 56 skipped + 72 crash-excluded. Every
  crash-excluded test is a C2-family interpreter abort, except one C3
  segfault (test_load empty-save) and one test_fft.py first-test abort
  (module out of scope anyway).

### Python failure kinds

8,615 junit failure messages parsed — 98.2% of the 8,770 counted failures;
the rest are retry-attempt bookkeeping in crash-heavy files.

| kind | count | bucket |
|---|---|---|
| `RuntimeError: [omarchy] ... is not implemented` | 8,543 | (a) |
| `IndexError: vector::_M_range_check` (cpu stream table) | 51 | (b) |
| `AssertionError` | 14 | see breakdown |
| `UnboundLocalError: custom_kernel` (upstream harness artifact) | 6 | n/a |
| `RuntimeError: Abs has no CPU implementation` | 1 | (b) |

The python named-gap histogram is dominated by three items:

- Sum on INTEGER dtypes — 4,011 asserts (dtype=int32 and friends; float
  sums work). This alone is most of test_reduce.py's 4,707 failures.
- ErfInv — 3,085 asserts (dtype=float32). This single primitive blocks the
  whole test_quantized.py calibration path (2,897 failures).
- non-contiguous Sum — 429 asserts (named error, correct refusal).

Then: Abs 225, Add dtype 103, Less 89, Equal dtype 78, Max dtype 73,
Prod dtype 72, Min dtype 72, And dtype 67, non-zero scalar fill 67,
Floor 20, ArgMin 9, dtype converting copy 8, Scatter 7, and smaller
counts. mx.array_equal hits the same bool And/Equal machinery gap as C++,
so python value checks are masked in exactly the same way.

The 14 AssertionErrors:

- 11 are defect **D2** below (test_reduce.py::test_expand_sums subTests —
  wrong values).
- 2 are device-info metadata gaps: test_device_count reports `0 != 1`,
  test_device_info_cpu finds no 'device_name' key.
- 1 is the scheduler cross-thread stream contract (C1 family):
  test_threads.py::test_threadlocal_stream expects RuntimeError inside
  another thread and never sees it.

## README-ready coverage statement

Upstream MLX's own test suites ran against the omarchy backend (Mesa
lavapipe, MLX_OMARCHY_ALLOW_NON_APPLE=1) on 2026-09-01. The C++ suite
executed 251 cases: 62 passed, 189 failed. The python suite executed 10,500
cases: 1,730 passed, 8,770 failed, 56 skipped, plus 72 tests excluded
because they crash the interpreter. Of all failures, exactly one C++ case
and 11 python subtests failed on wrong VALUES; everything else failed on a
named `[omarchy] ... is not implemented` error (integer-dtype Sum, ErfInv,
Abs, Scatter, bool And/Equal being the largest), on the deliberately absent
CPU backend, or in modules the roadmap excludes (fft, linalg,
export/import). Two caveats keep this honest: the suite's own
array_equal comparison cannot run (bool And/Equal gap), so most passing
values are unverified rather than proven, and 72 python tests abort the
interpreter because backend errors cross the numpy conversion boundary as
std::terminate instead of returning a Python exception. Full breakdown,
reproduction commands, and raw logs: receipts/2026-09-01-upstream-suite-coverage.md;
re-run with tools/run-upstream-suite.sh.

## Addendum: known measurement caveats

- Host load averaged 60+ during the runs. A parallel probe on this host
  showed llvmpipe can crash value_and_grad nondeterministically under that
  load, so a SIGSEGV seen in-suite should be treated as probable host
  noise rather than a backend defect. The three crashes reported above
  (C1, C2, C3) are exempt: each reproduced in isolation, deterministically
  (C3: 3/3 runs), away from the suite.
- The wheel predates the working tree's bf16-tape refusal (see How).
- .work/build-upstream remains in place (64 MB) so the tests binary can be
  re-run; tools/run-upstream-suite.sh rebuilds it identically if removed.
