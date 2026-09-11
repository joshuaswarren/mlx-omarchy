# QMM layout parity: fp batched weights, affine bf16 quantize, rank-1 qvm

- schema: mlx-omarchy/qmm-layout-parity/1 (2026-09-11)
- agent: QmmLayoutParity; branch `qmm/qmm-layout-parity` off origin/main
  `bddc061f`; artifacts in this directory measured on the M1 (host
  `<m1-host>`, Apple M1 G13G B1, aarch64 omarchy Linux, Honeykrisp fork
  `26.3.0.devel.hk6f6afc8-1`, stock Mesa via private ICD for stock legs)
- commits: `cddac6bc` + `05dec7f9` carry the implementation; `26e0a249`
  adds the bf16-tile named refusal (see "Honest residual" below)

## Which layouts and why, decided before implementing

Reading the refusal sites and the requal per-case receipts:

1. **258 fp-mode cases (fp_qmv 90, qmv_wide fp 84, fp_qvm 81,
   fp_qmv_large_output 3) are one signature: a 3D (batched) fp-quantized
   weight.** The fp branch of `QuantizedMatmul::eval_gpu` required
   `w.ndim() == 2`; every B=0 (2D w) upstream subtest passed and every
   B!=0 (3D w) subtest refused, which reproduces each test's failure
   count exactly (30x3, 14x6, 27x3, 1x3). Classification: **transform,
   no kernel work** — `binding()` binds from buffer byte 0 and the fp
   shaders take every stream base from push-constant offsets, so one
   dispatch per batch slice of the existing fp kernels serves it
   host-side. Slice pairing is decided by the ops-shaped output size
   (paired x batch vs broadcast shared x). No shader bytes changed, so
   llvmpipe screening is strong evidence for this fix.
2. **83 bf16 cases (qmm_non_transposed 82, gather_qmm_matrix_path 1)
   root-cause to the standing `mx.quantize`/`mx.dequantize` bf16 input
   guard, not to QuantizedMatmul.** quantize.comp/dequant.comp already
   carry `USE_BF16` load/store blocks (built for the fp modes); the
   affine direction simply did not compile them and both guards
   refused. Classification: **mechanical kernel work** — two new
   compilations of existing shader code (`quantize_bf16`,
   `dequant_bf16`), `QuantizeBF16`/`DequantBF16` enum values appended
   per the profile-id-stability convention, and guard widening in the
   ops.cpp quantize-errors patch plus `Quantize::eval_gpu` (patch
   apply verified against the pinned 0.32.2 tree). Numerics: codes
   pack from the unrounded f32 parameters exactly as before; scales
   and biases store bf16 round-to-nearest-even, matching the shader
   contract and the upstream test's self-consistency oracle.
3. **1 case (test_qvm_splitk tail) is rank-1 x** — the requal's
   "rank-3" wording is imprecise; the log shows the 1D-vector tail
   (out shape [2048]) refusing with `QuantizedMatmul rank`.
   Classification: **transform** — derive `m = 1` for 1D x; the flat
   [n] output writes identically to the [1, n] dispatch.

## What changed

- `overlay/mlx/backend/omarchy/primitives.cpp`: fp branch accepts 2D/3D
  weights with per-slice dispatch loops (vec, tile, general routes);
  affine branch derives `m` for rank-1 x; quantize/dequantize guards
  admit bfloat16 and select the new kernels; narrow named refusal for
  the bf16 non-transposed tile zone (below).
- `compute.h` / `compute.cpp` / `CMakeLists.txt`: QuantizeBF16 /
  DequantBF16 kernel registration (append-only profile ids) and the two
  shader compilations.
- `patches/mlx-omarchy-quantize-errors.patch`: bf16 admitted by the
  affine quantize input guard and the affine dequantize scales guard.
- `docs/compatibility-matrix.md`: fast::Quantize dtypes gain bf16*.
- Tests (`overlay/tests/omarchy/test_primitives.cpp`), one per layout
  class:
  - fp batched layouts vs hand-packed e2m1/e4m3/e8m0 oracles: mxfp4,
    mxfp8, nvfp4, transposed and non-transposed, vec/tile/general
    routes, per-slice scales that make a shared-weight bug impossible,
    per-row x values that make a pairing bug impossible.
  - affine bf16 quantize: words bit-exact vs the f32 oracle
    (`host_affine_quantize`), stored params bit-exact bf16 roundings,
    dequantize roundtrip inside an analytic bound, and a bf16 qmm
    against the dequantized reference.
  - rank-1 qvm vs the rank-2 oracle construction.
  - The two stale refusal pins (bf16 quantize input, bf16 dequantize
    scales) were removed; the positive contracts replace them.

## Upstream case-count delta (the headline)

Full upstream Python phase, `tools/run-upstream-suite.sh --py-only`
unedited (DEVICE=gpu, MLX_ENABLE_TF32=0, fork driver), wheel
`0.32.2.dev202609112303+05dec7f9` (pre-refusal commit; provenance
verified match):

| measure | requal 2a9add42 | this branch | delta |
|---|---|---|---|
| Python executed | 11,847 | 11,848 | +1 |
| Python passed | 11,483 | **11,816** | **+333** |
| Python failed | 364 | **32** | **-332** |
| named refusals | 342 | see below | |
| test_quantized.py failures | 343 | 11 | -332 |

Per-test closure inside test_quantized.py (343 -> 11):

| test | requal | now |
|---|---|---|
| test_fp_qmv (layout) | 90 | 0 |
| test_qmv_wide (layout) | 84 | 0 |
| test_qmm_non_transposed | 82 | 10 (see below) |
| test_fp_qvm (layout) | 81 | 0 |
| test_fp_qmv_large_output | 3 | 0 |
| test_qvm_splitk (rank) | 1 | 0 |
| test_gather_qmm_matrix_path (quantize bf16) | 1 | 0 |
| test_gather_qmm_sorted (wrong value) | 1 | 1 (unchanged, sibling cluster) |

The other 21 failures are exactly the requal's non-quantized residual
clusters (custom-kernel harness artifact 5+1, f16 SDPA 4, compile 4,
ops 3, export 2, conv 1, nn 1, optimizers 1) - no new failures outside
test_quantized.py. Raw: `m1-results/upstream-py-summary.tsv`.

## Honest residual: the bf16 non-transposed tile zone

Closing quantize-bf16 exposed a latent accuracy defect that the old
guard had been hiding: 10 upstream `test_qmm_non_transposed` bf16
subtests (M in {8,33,65}, K in {64,128}, bits in {4,8}, plus the
(33000,128,64) K=128 sweep row) now compute but fail the upstream
1.5e-3 tolerance with max_abs_diff 0.00195312 = exactly 2^-9 - one
bf16 output ULP at the relevant magnitude. The tile route's
accumulation order rounds a few elements to the adjacent bf16 code
relative to the dense bf16 matmul reference; the m==1 vec route, K=256
and K>=512 agree within tolerance. At wheel 05dec7f9 those cases are
wrong answers, which this task's contract calls a failure, not
progress. Commit `26e0a249` therefore refuses exactly that zone by
name: `QuantizedMatmul bf16 non-transposed tile` for bf16,
transpose=false, m>1, K<=128 - never a wrong answer. The refusal is
host-side dispatch logic and was verified on llvmpipe for all twelve
failing signatures (each raises the named error) plus K=256/K=512
controls (still compute, max diff 9.8e-4 / 4.9e-4); the hardware
confirmation is below. Root-causing the tile accumulation order is the
follow-up that reopens these 10 cases.

Final accounting at branch HEAD (`68e4f10d`): 342 named refusals ->
**10 named** (the bf16-tile zone: bits 4/8, K in {64,128}, M in
{8,33,65} plus the (33000,128,64) row) + 1 unchanged wrong value
(gather_qmm_sorted, sibling's cluster); 332 of the 342 closed, 332
total failures closed (364 -> 32). Upstream bits-2 subtests in the
same shapes pass on hardware and stay served: an intermediate refusal
draft without the bits-2 exemption briefly named them on hardware,
and `68e4f10d` exempts bits 2 (llvmpipe-verified: bits-2 M=8 K=64
computes with max diff 0.0).

## M1 verification

Wheel `0.32.2.dev202609112303+05dec7f9` built on the M1 from this
branch, outside the lock; sha256 in `wheel-sha256.txt`. One flock
window carried digest gates and the C++ suites; the upstream phase ran
in the same window (23:08-23:12Z); digest-only re-grab 23:20-23:22Z
after a venv fix (first attempt skipped all legs on a missing mlx_lm).

### Digest gates - all twelve canonical pins UNCHANGED

Pins from receipts/2026-09-11-main-parity-12-matrix; measured values in
`m1-results/digests-{fork,stock}.json`:

| driver | model | leg | pin | measured |
|---|---|---|---|---|
| fork | qwen25-0.5b-4bit | short | 7fd25a869ff21678 | UNCHANGED |
| fork | qwen25-0.5b-4bit | long | 4cc08910089477fd | UNCHANGED |
| fork | qwen25-0.5b-4bit | longctx | 7da83f06ec9f001d | UNCHANGED |
| fork | qwen25-0.5b-bf16 | short | f26175202f3dabe9 | UNCHANGED |
| fork | qwen25-0.5b-bf16 | long | 8690dc83246b39f8 | UNCHANGED |
| fork | qwen25-0.5b-bf16 | longctx | ff502900d2a179a5 | UNCHANGED |
| stock | qwen25-0.5b-4bit | short | 7fd25a869ff21678 | UNCHANGED |
| stock | qwen25-0.5b-4bit | long | 4cc08910089477fd | UNCHANGED |
| stock | qwen25-0.5b-4bit | longctx | 7da83f06ec9f001d | UNCHANGED |
| stock | qwen25-0.5b-bf16 | short | 7fc0f968789b1882 | UNCHANGED |
| stock | qwen25-0.5b-bf16 | long | 46108ad71157cb4d | UNCHANGED |
| stock | qwen25-0.5b-bf16 | longctx | ff502900d2a179a5 | UNCHANGED |

### Omarchy C++ suites on the M1 (fork driver)

- `omarchy_primitive_tests`: 103/103 cases, 0 failed (includes the
  three new TEST_CASEs and the two flipped bf16 pins).
- `omarchy_matmul_family_tests`: exit 0.
- Raw: `m1-results/primitive-tests.log`, `m1-results/matmul-family-tests.log`.

### llvmpipe screening (dev box, before the window)

- The three new TEST_CASEs: 1548 assertions green.
- Full `omarchy_primitive_tests` on this branch: 102/103 cases,
  2,700,946/2,700,947 assertions; the single failure (`quantized
  matmul binds affine streams at storage offsets`) reproduces
  identically on UNMODIFIED bddc061f (163/164 asserts, same bound) -
  a pre-existing software-driver deviation outside this change; the
  M1 runs that case green (103/103 above).
### Hardware confirmation of the named refusal

A post-refusal wheel (`26e0a249`, sha256
c7f15054dc2cd375cd620bbcf7d9497f37dcc61dd3d52a7dffc7dad4690660d0)
reran `test_quantized.py` on the M1 inside one short window (23:38Z):
the K<=128 bf16 tile subtests raise
`RuntimeError: [omarchy] QuantizedMatmul bf16 non-transposed tile ...`
(named refusal) instead of assertion failures - see
`m1-results/test_quantized-final.log`. That wheel refused the bits-2
subtests too (over-broad); `68e4f10d` exempts bits 2, verified on
llvmpipe (bits-2 computes, max diff 0.0; bits 4/8 still refuse; K=256
and K=512 controls compute at 9.8e-4 / 4.9e-4). The refusal is
host-side dispatch logic, so that llvmpipe run plus the hardware run
of its immediate predecessor pins the final behavior; the final-commit
hardware rerun of test_quantized.py is queued behind the landing
queue's windows as a formality.
