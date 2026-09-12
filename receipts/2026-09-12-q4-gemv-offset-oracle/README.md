# Q4 GEMV affine storage-offset case — diagnosis and test rework — 2026-09-12

Closes the investigated mechanism for `omarchy_primitive_tests`
"quantized matmul binds affine streams at storage offsets" (m=1 failure in
`receipts/2026-09-12-parity-status/battery/omarchy_primitive_tests.log` and
Item 2 of `receipts/2026-09-12-composed-regressions/README.md`). The
1f6a7bf8 offset-composition hypothesis is refuted: the aux binding is
correct on every route; the old case's decode leg compared the native-qmv
arithmetic route against a double-precision oracle at an epsilon that
arithmetic cannot meet on cancelling outputs. The case is reworked to its
real contract — storage offset invariance, checked bitwise — with absolute
values still pinned by the host-oracle cases.

M1 hardware confirmation (window of 2026-09-12, `drivers.txt`,
`focused.log`, `battery/`): tree `49325af7` + this fix as the patch
`1df2d9fec6efc7bb…`, driver mesa-honeykrisp-omarchy
26.3.0.devel.hk6f6afc8-1, kernel 7.1.6-1-1-ARCH, Apple M1 (aarch64),
glslc 2026.3. The reworked case passes (1 case, 166 assertions), and the
full standing battery passes 30/30 (25 suites plus 5 capability-sim
profiles); `omarchy_primitive_tests` runs 103/103 cases, 2,700,949
assertions, zero failures — the suite that was 29/30 in
`receipts/2026-09-12-parity-status` is green.

Dev-box supporting evidence (x86, AMD EPYC, software Vulkan via lavapipe,
`MLX_OMARCHY_ALLOW_NON_APPLE=1`, binaries from `.work/build` after
`scripts/prepare-mlx.sh`) is in sections 1-4. It is supporting evidence
only; the M1 run above is the hardware claim of record.

## 1. The old failure reproduces on the dev box, values identical

The committed case (checkpoint `198cb7f4`, test text byte-identical to the
M1 runs) was rebuilt and run on lavapipe:

```
test_primitives.cpp:5514: ERROR: CHECK( device_values[index] ==
    doctest::Approx(expected[index]).epsilon(4e-3) ) is NOT correct!
  values: CHECK( 0.416748 == Approx( 0.409468 ) )
  logged: m := 1
```

Same single element, same values as the M1 battery log. The divergence is
IEEE arithmetic, not driver behavior.

## 2. Exact-arithmetic replica: correct binding reproduces the value bit for bit

A standalone C++ replica (seed 13 generator, `host_affine_quantize` and
`host_quantized_matmul` transcribed verbatim from the test file, plus a
transcription of the `qmm_vec.comp` `QMM_VEC_Q4_WORD` arithmetic: f16 x
quad chain sums `((h0+h1)+h2)+h3`, per-quad fma nibble ladders, fma block
finish, stride-1-first pairwise 32-lane slot reduction, f16 store) ran the
m=1 decode for all 20 columns under each binding hypothesis:

| hypothesis | columns outside 4e-3 | max abs delta |
|---|---|---|
| correct aux bases (1/3 applied once) | **1 of 20** (column 11) | 0.0073 |
| scale base dropped | 20 of 20 | 127.1 |
| bias base dropped | 18 of 20 | 76.5 |
| both bases dropped | 19 of 20 | 127.0 |
| scale base double-counted | 20 of 20 | 115.0 |
| bias base double-counted | 20 of 20 | 76.4 |
| scale/bias bases swapped | 19 of 20 | 88.1 |

Only the correct binding produces the documented signature. With it, the
replica emits column 11 as `0.416748` (f16) against the oracle's
`0.409468` — the exact documented pair — and every other column inside
4e-3. An indexing error on this data moves a column by O(1); a 0.0073
delta on one column cannot be one.

Mechanism: the native qmv route adds each x quad in f16 (`load_vector`,
T = half) and the block epilogue multiplies that sum by the affine bias.
Eight f16 roundings wobble the bias term by ~1e-2 absolute; where the
output nearly cancels (column 11 expected 0.409) that wobble exceeds a
4e-3 relative epsilon, and where |expected| >= 1.3 it does not. The
double-precision oracle transcribes the dequant math, not the route
arithmetic — the test file itself says it is "not an independent M1
execution oracle". `receipts/2026-09-09-q4-gemv-order` pins this route
arithmetic as bit-identical to native Metal qmv by design; changing it to
satisfy the old epsilon would break that contract.

## 3. Shared-caller audit (no backend change needed)

- `QuantizedMatmul::eval_gpu` single-weight dispatch: `binding()` binds
  each whole buffer at offset 0; `aux_offset`/`aux_size` carry the stream
  bases in items; `qmm.comp`, `qmm_tile.comp`, and `qmm_vec.comp` (all
  three variants incl. `Q4_WORD`) add them consistently. The m=7 tile leg
  of the same case passes with the same offset views.
- `dispatch_quantized_gemv_group` (multi-weight decode): refuses
  non-whole-dense operands via `whole_dense`, so its "every stream starts
  at element 0" shader contract holds.
- FP8/FP4 (`FP_MODE`) vec paths: byte-stream semantics, separate shaders,
  own suites green.
- `Quantize`/`Dequantize`: allocate fresh group parameters (offset 0).

## 4. Test rework (the landed change)

`overlay/tests/omarchy/test_primitives.cpp` only. The case now runs the
same dispatch twice per m in {1, 7}: once with the scale/bias streams as
slices of padded f16 storages at item bases 1 and 3, once with fresh
zero-offset arrays holding the same logical values, and requires the two
outputs to match bit for bit. Any mis-composed stream base moves every
column by O(1), which bitwise equality refuses; the f16 arithmetic route
no longer needs an epsilon at all. Absolute correctness stays pinned by
"quantized matmul matches dequant and host references" and "runs f16 and
bf16 activations".

Dev-box verification:

- Reworked case passes: 1 case, 166 assertions, 0 failed (lavapipe).
- Mutation check: with `params.aux_offset` forced to 0 in the affine
  dispatch (one-line local mutation, reverted after), the reworked case
  fails loudly (`-80.75 == 19.7344`, `74.3125 == -18.1094`, ...), so the
  case detects the defect class it exists for.
- Negative control: the pre-rework case fails on this box with the exact
  documented values (section 1).

## 5. Status: closed

- Dev-box verification: complete (sections 1-4).
- M1 hardware confirmation: complete (header). Standing battery 30/30;
  the previously failing case passes on the M1 under the installed
  baseline driver.
- `docs/known-defects.md` affine entry rewritten from the refuted
  composition hypothesis to the oracle-miscalibration diagnosis in this
  change; Item 2 of `receipts/2026-09-12-composed-regressions/README.md`
  marked superseded by this receipt.
