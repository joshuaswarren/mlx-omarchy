# Activation quantization for qqmm fp modes — receipt

Date: 2026-09-06
Branch: fix/activation-quant
Commit: 7b778d29fb3ef50f3a5b0380ef96802c4c77af89 (base d2a79b0b)
Host: EPYC 7443P PVE node, Mesa llvmpipe (Vulkan software), python 3.11

## Scope delivered

True activation quantization for GatherQQMM and QQMatmul non-affine
modes (mxfp4 / nvfp4 / mxfp8):

- The activation is fake-quantized on the GPU: dequant(quant(x))
  through the existing QuantizeFp / DequantFp kernels (quantize.comp /
  dequant.comp, bit-mirroring the pinned Metal fp4.h / fp8.h). The
  dispatch blocks were factored out of fast::Quantize::eval_gpu into
  dispatch_fp_quantize / dispatch_fp_dequantize so Quantize and the
  qqmm paths share one parameterization; the affine Quantize branch is
  unchanged.
- Float weights are packed by the same kernels (upstream qqmm
  contract: quantized or float w).
- nvfp4 global scales: global_scale_x is absorbed by the activation
  fake-quantization pair; global_scale_w (one float32 scalar for the
  tensor, upstream contract) is applied per output through three new
  GatherQmmNbFpHgs{F32,F16,BF16} shader variants (binding 4, multiply
  by gs_w / 2688), matching the pinned Metal _hgs kernels. Round-up
  e8m0 (group 32) and e4m3 RNE (group 16) scale semantics and e2m1 /
  e4m3 element rounding are exactly the existing kernels'.
- The product contracts quantized-x codes against quantized-w codes
  via dispatch_gather_qmm (fp_mode=true); no raw-float x against
  dequantized w anywhere. Affine paths and refusals unchanged;
  non-affine group sizes outside {16, 32} refuse by name.

## Files

- overlay/mlx/backend/omarchy/primitives.cpp (GatherQQMM::eval_gpu,
  QQMatmul::eval_gpu, dispatch_fp_* helpers, dispatch_gather_qmm out
  global scale)
- overlay/mlx/backend/omarchy/shaders/gather_qmm.comp
  (USE_OUT_GLOBAL_SCALE block)
- overlay/mlx/backend/omarchy/CMakeLists.txt (3 HGS variants)
- overlay/mlx/backend/omarchy/compute.h / compute.cpp (3 kernel enums)
- overlay/tests/omarchy/test_matmul_family.cpp (host bit-reference
  suite + updated refusal tags)

## Verification (commands and raw outcomes, this session)

1. C++ targeted + full family, build .work/aq-build:

   MLX_OMARCHY_ALLOW_NON_APPLE=1 LP_NUM_THREADS=4 \
     omarchy_matmul_family_tests -tc='*qqmm*,qq matmul*'
   -> test cases: 2 | 2 passed | 0 failed; assertions: 1340 | 1340

   MLX_OMARCHY_ALLOW_NON_APPLE=1 LP_NUM_THREADS=4 \
     omarchy_matmul_family_tests   (full binary)
   -> test cases: 9 | 9 passed | 0 failed
   -> assertions: 18295 | 18295 passed | 0 failed (EXIT=0)

   New case "qqmm fp modes fake-quantize the activation": 181
   assertions (threshold-boundary identity products for exact
   quant/dequant reference values, mxfp8/mxfp4/nvfp4+gs qqmm and
   gather_qqmm against double-precision host bit-references,
   float-weight packing, group-size refusal tag).

2. Wheel + provenance:

   dist/mlx_omarchy-0.32.2.dev202609061024+7b778d29-cp311-cp311-linux_x86_64.whl
   sha256 de04a5ff0b90dc09135f42b0159ec4ef116b2c5a5a7dd1ff6d2617a2e72d88d9
   scripts/mlx_provenance.py --expect-wheel: "verified": "match",
   dist_version == mx_version == 0.32.2.dev202609061024+7b778d29

3. Targeted upstream python tests (venv-pytest, DEVICE=gpu,
   MLX_ENABLE_TF32=0):

   test_quantized.py::TestQuantized::test_qqmm     PASSED
   test_quantized.py::TestQuantized::test_qqmv     PASSED
   test_quantized.py::TestQuantized::test_gather_qqmm PASSED
   -> 3 passed, 130 subtests passed in 11.40s
   (These are exactly the 130 previously-refusing subtests:
   80 GatherQQMM x quantization + 32 test_qqmm + 18 test_qqmv.)

4. Full test_quantized.py raw outcome (my wheel):

   374 failed, 37 passed, 2991 subtests passed in 1188.29s
   (py4 baseline same file: 391 failed, 36 passed, 2636 subtests
   passed; target subfail rows 80+32+18 = 130 -> 0.)

5. A/B against pure base d2a79b0b (separate worktree, separate wheel
   0.32.2.dev202609061103+d2a79b0b):

   test_qmm_non_transposed + test_qmv_wide on base: 189 failed subtests
   on my line (7b778d29):                      105 + 84 = 189
   -> identical failure set; my commit introduces no python-suite
   regression. BatchedQmmRecovery independently corroborated the same
   rows failing on their before-control. These rows (fp16
   qmm_non_transposed / mxfp8 qmv_wide / fp_qmv / fp_qvm weight
   layout) are the batched-fp-qmm territory (fix/batched-fp-qmm-v2).

## Remaining known failures (not this task's scope)

- test_qmm_non_transposed (fp16), test_qmv_wide (mxfp8),
  test_fp_qmv / test_fp_qvm / test_fp_qmv_large_output (QuantizedMatmul
  fp weight layout), test_gather_qmm_sorted/matrix_path rows: inherited
  at d2a79b0b, owned by the batched-fp-qmm worker.
- test_qvm_splitk ("QuantizedMatmul rank") remains refused.

## Notes

- llvmpipe-only validation per repo contract; M1 untouched (no M1
  access this session, per assignment).
- Full-file runs were contended (two suites concurrently); the
  classification counts above are per-run raw outcomes, receipts at
  /tmp/aq-py-full.log, /tmp/aq-run-full.log, /tmp/aq-ab-py.log.
