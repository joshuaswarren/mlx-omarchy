# Affine QuantizedMatmul: direct scale/bias bindings, packing copies removed

Date: 2026-09-06 (recovery session AffineQmmRecovered, continuing
AffineQmmCopyElimination)
Base: c8618f53 (v0.3.6) - candidate commit: cad495ec (local, not pushed)
Device: llvmpipe (LLVM 15.0.6) software Vulkan, x86 development box.
Provenance: `scripts/mlx_provenance.py` = verified "match" for both wheels
(baseline RECORD d8bfcf62..., candidate re-checked at install time).

## Change

`QuantizedMatmul::eval_gpu` affine path staged `[scales | biases]` into one
combined buffer with two `vkCmdCopyBuffer` calls per eval, then dispatched
four bindings. It now binds the two streams directly:

- bindings 2/3 are the scale and bias buffers, output moves to binding 4;
  five bindings, within the existing six-slot descriptor budget
  (`kComputeBindingBudget = 6`, compute.h). FP_MODE keeps bindings 2/3
  unchanged (scale word view at 2, output at 3).
- shaders `qmm.comp`, `qmm_vec.comp`, `qmm_tile.comp` index each stream from
  its own storage offset. The dead push-constant slots the packed layout
  reserved carry the two base items: `aux_offset` = scale base item,
  `aux_size` = bias base item. `shape[3]` (bias-region offset, read as
  `scale_total`) and the shader-side `scale_total` are deleted.
- the `vkCmdCopyBuffer` 4-byte-alignment refusal
  ("QuantizedMatmul scales byte offset") is retired as obsolete:
  item-indexed reads need no word alignment. `checked_item_offset` still
  refuses sub-item offsets. Odd 16-bit views are now first-class (covered
  by a new test and the Python probe below).
- per-eval work removed: one combined allocation + flags + two copies +
  three add_temporary calls. Dispatch count per eval is unchanged.
- untouched by construction: non-affine FP_MODE path (same source, separate
  `#ifdef FP_MODE` arms, separate eval branch with its own four bindings),
  gather qmm (`shaders/gather_qmm.comp`, separate source + its own packing
  copies owned by ActivationQuantParity), dequantize/quantize kernels.
- all 24 shader variants (three storage dtypes across scalar, tiled,
  tree-vector and subgroup-vector kernels, each in affine and FP_MODE) compile clean under `glslangValidator -V -Os
  --target-env vulkan1.3`; SPIR-V disassembly of the affine f16 vec variant
  shows bindings scales=2, biases=3, output=4; the FP_MODE variant keeps
  scale_bias=2, output=3.

## Numerical parity (before vs after, same device, same inputs)

C++ battery (fixed seeds, host double-precision references):

| suite | baseline c8618f53 | candidate cad495ec |
|---|---|---|
| omarchy_primitive_tests | 94/94 cases, 2,580,602 assertions | 95/95 cases, 2,580,766 assertions (+1 new test, +164 assertions) |
| omarchy_matmul_family_tests | 8/8, 18,114 | 8/8, 18,114 (identical) |
| omarchy_runtime_tests | (not run at baseline) | 34/34 on quiet reruns; one timing flake during a -j10 build (llvmpipe watchdog teardown), clean twice after |

Parent-corrected Python A/B probe (`affine_qmm_probe.py`, PYTHONHASHSEED=0,
identical inputs, output bytes hashed after f32 cast): operands are evaluated
before sampling counters. The original worker probe ignored its `batched`
argument; its batch claim was invalid. The corrected probe records input,
weight and output shapes and tests three genuine two-batch cases. Results
are in `probe-before-parent.json` and `probe-after-parent.json`.

| case (path) | copies before | copies after | dispatch before/after | output parity |
|---|---|---|---|---|
| vec_f32_m1_T | 2 | 0 | 1 / 1 | bitwise-identical |
| tile_f32_m7_T | 2 | 0 | 1 / 1 | bitwise-identical |
| scalar_f32_m7_T | 2 | 0 | 1 / 1 | bitwise-identical |
| vec_f16_m1_T | 2 | 0 | 1 / 1 | bitwise-identical |
| tile_f16_m7_NT | 2 | 0 | 1 / 1 | bitwise-identical |
| tile_f32_batched | 2 | 0 | 1 / 1 | bitwise-identical |
| bits6_m7_T | 2 | 0 | 1 / 1 | bitwise-identical |
| vec_f16_batched | 2 | 0 | 1 / 1 | bitwise-identical |
| scalar_f32_batched_NT | 2 | 0 | 1 / 1 | bitwise-identical |
| nonaffine_mxfp4 (untouched) | 0 | 0 | - | n/a |

Exactly the two packing copies per affine eval are gone; every dispatch
count is unchanged; outputs are byte-identical across builds.

## Storage-offset regression (new test)

"quantized matmul binds affine streams at storage offsets"
(overlay/tests/omarchy/test_primitives.cpp): f16 scale/bias views at storage
offsets 2 and 6 bytes (distinct odd item bases 1 and 3), decode m=1 and tile m=7, checked against the host reference.

- baseline wheel: refuses with the exact named error
  "[omarchy] QuantizedMatmul scales byte offset is not implemented ..."
  (probe-before-parent.json, "odd_offset.refused").
- candidate wheel: computes; view vs compact-storage reference are
  byte-identical (probe-after-parent.json, odd_offset.match = true).

## Trace evidence

Backend trace counters via the exported C ABI
`mlx_omarchy_trace_snapshot` (same mechanism as
scripts/fragmentation_probe.py): `vk_buffer_copies` delta per affine eval
drops 2 -> 0 across executed vec, tile, and scalar paths and f16/f32;
`vk_compute_dispatches` delta unchanged; non-affine delta 0 -> 0.

## Subgroup variant note

llvmpipe does not dispatch the subgroup variant (caps gate:
"Skipping: subgroup size != 32 or no ARITHMETIC"), so the subgroup binary is
not directly exercised here. Its shader source compiles in all 6 variants,
the affine addressing edit is byte-identical to the tree variant, and the
SPIR-V binding layout was disassembly-verified.

## Timing claim status

No timing claims are made in this receipt. Local llvmpipe wall-clock is
software-development evidence only and is NOT M1 speed evidence. The bounded
M1 A/B command for the parent is `m1-ab-command-affine_qmm_ab.py`:

    python3 -m venv .venv-ab && .venv-ab/bin/pip install --quiet numpy
    MLX_OMARCHY_QMM_TILE=1 .venv-ab/bin/pip install --force-reinstall --no-deps <WHEEL>
    MLX_OMARCHY_QMM_TILE=1 .venv-ab/bin/python m1-ab-command-affine_qmm_ab.py <baseline|candidate>

Run once per wheel (baseline c8618f53 wheel, then candidate cad495ec wheel),
same M1, same shell, machine otherwise idle; the script prints the wheel
stamp per side (assert the stamps differ before comparing: 0.32.2.dev...+c8618f53
vs 0.32.2.dev...+cad495ec), median-of-N ms for decode gemv f16 m=1 n=k=4096
(50 iters) and prefill tile f16 m=1023 (15 iters). This command has not been run. The compiler session owns the M1
exclusively; native qualification requires explicit hardware handback.

## Wheels

- baseline: mlx_omarchy-0.32.2.dev202609061632+c8618f53-cp311-cp311-linux_x86_64.whl,
  sha256 89f2ba3a67c5b61140db0eb8f5e9f202da2e992c350ad0db1d7d26b9f6e63b0f
  (dist file later replaced by the candidate build; hash + stamp preserved
  in baseline-wheel-sha256.txt and baseline-wheel.log)
- candidate: mlx_omarchy-0.32.2.dev202609061702+cad495ec-cp311-cp311-linux_x86_64.whl,
  sha256 8c172441c58afc18dc67030df4e4a9416172ecfdc1c02d7870166a0df8e0b3e3

## Session defect record (origin confirmed, not a code issue)

ORIGIN-NOTE-multiline-literal-defect.txt: the eval harness rewrites
negation-prefixed physical lines inside multiline Python string literals,
which initially corrupted one inserted test line (first observed at
overlay/tests/omarchy/test_primitives.cpp:5011 as a __omp_shell fragment).
Repaired by line-index write; final sources contain zero defect markers
(verified). Recorded so the parent knows the edit lineage is clean.
