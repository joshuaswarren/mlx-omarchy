# QMM layout parity: fp batched weights, affine bf16 quantize, rank-1 qvm

- schema: mlx-omarchy/qmm-layout-parity/1 (2026-09-11)
- agent: QmmLayoutParity; branch `qmm/qmm-layout-parity` (off origin/main bddc061f)
- target: receipts/2026-09-11-upstream-suite named-refusal clusters
  `QuantizedMatmul weight layout` (258), `Quantize input dtype` (83),
  `QuantizedMatmul rank` (1)

## Which layouts and why, decided before implementing

Reading the refusal sites and the per-case receipts:

1. **258 fp-mode cases (fp_qmv 90, qmv_wide fp 84, fp_qvm 81,
   fp_qmv_large_output 3) are one signature: a 3D (batched) fp-quantized
   weight.** The fp branch of `QuantizedMatmul::eval_gpu` required
   `w.ndim() == 2`; every B=0 (2D w) upstream subtest passed and every
   B!=0 (3D w) subtest refused, which reproduces each test's case count
   exactly (30x3, 14x6, 27x3, 1x3). Classification: **transform, no
   kernel work** — `binding()` binds from buffer byte 0 and the fp
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
   after the profile-id comment convention, and guard widening in the
   ops.cpp quantize-errors patch plus `Quantize::eval_gpu`. Numerics:
   codes pack from the unrounded f32 parameters exactly as before;
   scales/biases store bf16 (round-to-nearest-even), matching the
   shader contract and the upstream test's self-consistency oracle.
3. **1 case (test_qvm_splitk tail) is rank-1 x** — the receipt's
   "rank-3" wording is imprecise; the log shows the 1D-vector tail at
   test line 853 refusing with `QuantizedMatmul rank`. Classification:
   **transform** — derive `m = 1` for 1D x; the flat [n] output writes
   identically to the [1, n] dispatch.

Remaining named after this change: fp-mode weights with ndim > 3 or
non-uint32 words (not exercised upstream), non-float affine inputs
(ops-level validation), float64 parameters (cannot exist on the GPU
stream). Everything upstream exercises is served.

## What changed

- `overlay/mlx/backend/omarchy/primitives.cpp`: fp branch accepts 2D/3D
  weights with per-slice dispatch loops (vec, tile, general routes);
  affine branch derives `m` for rank-1 x; quantize/dequantize guards
  admit bfloat16 and select the new kernels.
- `overlay/mlx/backend/omarchy/compute.h` / `compute.cpp` /
  `CMakeLists.txt`: QuantizeBF16/DequantBF16 kernel registration
  (append-only profile ids) and the two shader compilations.
- `patches/mlx-omarchy-quantize-errors.patch`: bf16 admitted by the
  affine quantize input guard and the affine dequantize scales guard
  (verified to apply against the pinned 0.32.2 tree).
- `docs/compatibility-matrix.md`: fast::Quantize dtypes gain bf16*.
- Tests (`overlay/tests/omarchy/test_primitives.cpp`):
  - fp batched layouts vs hand-packed e2m1/e4m3/e8m0 oracles: mxfp4,
    mxfp8, nvfp4, transposed and non-transposed, vec/tile/general
    routes, per-slice scales that make a shared-weight bug impossible,
    per-row x that makes a pairing bug impossible.
  - affine bf16 quantize: words bit-exact vs the f32 oracle
    (`host_affine_quantize`), stored params bit-exact bf16 roundings,
    dequantize roundtrip inside the analytic bound, and a bf16 qmm
    against the dequantized reference.
  - rank-1 qvm vs the rank-2 oracle construction.
  - The two stale refusal pins (bf16 quantize input, bf16 dequantize
    scales) were removed; the positive contracts replace them.

## llvmpipe screening (dev box, before the M1 window)

- The three new TEST_CASEs pass: 1548 assertions green on llvmpipe
  (`omarchy_primitive_tests`, MLX_OMARCHY_ALLOW_NON_APPLE=1).
- Full `omarchy_primitive_tests` on my branch: 102/103 cases, 2,700,946
  of 2,700,947 assertions. The single failure,
  `quantized matmul binds affine streams at storage offsets`, was
  reproduced identically on UNMODIFIED bddc061f (163/164 asserts, same
  bound) — pre-existing on this software driver, untouched by this
  branch, and the M1 remains the arbiter for f16 kernel numerics.

## M1 verification (Apple M1, jwm1)

WHEEL: see `wheel.sha256` (built from this branch by
`scripts/build-wheel.sh` on the M1, outside the GPU lock).

One flock window on /tmp/m1-gpu.lock carried:
1. Digest gates, fork driver (honeykrisp-omarchy): canonical legs
   short/long/longctx for qwen25-0.5b-4bit and qwen25-0.5b-bf16.
2. Digest gates, stock driver (stock-mesa via private ICD): same legs.
3. Upstream Python phase (fork driver), runner `tools/run-upstream-suite.sh
   --py-only`, unedited, same env as the requal.
4. `omarchy_primitive_tests` and `omarchy_matmul_family_tests`, fork
   driver, same binaries as the window's wheel commit.

Canonical digest pins (receipts/2026-09-11-main-parity-12-matrix):

| driver | model | leg | pin | measured |
|---|---|---|---|---|
| fork | q4 | short | 7fd25a869ff21678 | SEE digests-fork |
| fork | q4 | long | 4cc08910089477fd | SEE digests-fork |
| fork | q4 | longctx | 7da83f06ec9f001d | SEE digests-fork |
| fork | bf16 | short | f26175202f3dabe9 | SEE digests-fork |
| fork | bf16 | long | 8690dc83246b39f8 | SEE digests-fork |
| fork | bf16 | longctx | ff502900d2a179a5 | SEE digests-fork |
| stock | q4 | short | 7fd25a869ff21678 | SEE digests-stock |
| stock | q4 | long | 4cc08910089477fd | SEE digests-stock |
| stock | q4 | longctx | 7da83f06ec9f001d | SEE digests-stock |
| stock | bf16 | short | 7fc0f968789b1882 | SEE digests-stock |
| stock | bf16 | long | 46108ad71157cb4d | SEE digests-stock |
| stock | bf16 | longctx | ff502900d2a179a5 | SEE digests-stock |

(THIS RECEIPT IS INCOMPLETE: the M1 window was still running when this
file was written. The tables below are filled from the window artifacts
in this directory before landing; a row that does not say UNCHANGED or
carry the measured digest means the window had not finished.)
