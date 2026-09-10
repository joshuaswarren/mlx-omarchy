# qmm coopmat offset-alignment determinism hardening

Branch: `wave/QmmAlignDeterminism` (commit recorded in verdict.json).
Parent: origin/main `250b9014`.

## Gate

The qmm coopmat gate (`overlay/mlx/backend/omarchy/primitives.cpp`,
`QuantizedMatmul::eval_gpu`, the rb/f16/q4/g64 branch) dispatched
`QmmPrefillCoopmatF16` only when `lhs_offset` and `output_offset` were
even; any odd-offset row-contiguous activation view silently rerouted
`matrix_m > 1` q4/group-64 f16 quantized matmul to the register-blocked
tile kernel (`QmmTileRbF16`, or `QmmTileRbPreciseF16` at m >= 1024).
The coopmat shader (`shaders/qmm_coopmat.comp`) reads x as 32-bit word
pairs (`x_slice = lhs_offset / 2`), which is why the parity condition
exists.

## What the investigation found (premise correction)

The assigned premise - that this fallback is not digest-neutral, and
that the f873dc2b load-window flip (receipts/2026-09-10-qmm-splitk-parity,
`fragmentation_hazard_found`) comes from it - does not reproduce:

- Forcing the unaligned condition deterministically (same x values as a
  whole buffer vs a row-contiguous view at f16 element offset 1) gives
  bit-identical outputs on the pre-fix backend and pre-fix installed
  wheel at m=64/262/1053 over the Qwen prefill shapes, including the
  m >= 1024 Precise-variant route.
- `mx.metal.device_info()` before model load does not flip the pinned
  long-decode-128 digest on a quiet machine (2 reps each arm).
- The committed fork-split receipt already held all six pins on the
  pure-tile route (`MLX_OMARCHY_NO_COOPMAT=1` full matrix).

The tile fallback is bit-identical to coopmat on the installed fork.
The f873dc2b flip under concurrent GPU load is real but unattributed;
verdict.json records what was excluded and what a follow-up needs
(contention-window repro with kernel profiling).

## The change (landed as assigned hardening)

Coopmat eligibility is resolved before operand normalization; an odd
f16-element-offset row-contiguous x view is materialized through the
existing `ensure_dense`/`contiguous_copy_gpu` engine into a fresh
aligned buffer before dispatch. The output is always a fresh offset-0
allocation. The route depends only on capability and env - allocation
luck is removed from the equation - and a post-staging alignment
violation refuses by name (`coopmat operand alignment`) instead of
silently rerouting.

Fast path: the eligibility conditions were already evaluated (env reads,
caps lookup) plus one modulo; no dispatch, copy, or allocation changes.
The staging copy fires only on the rare unaligned view. Stock Mesa
(`cooperative_matrix_f32_8=0`) never has `coopmat_reachable` true and
takes the identical route as before.

## Evidence

See verdict.json and `m1/`. Summary:

- Regression test "qmm coopmat output is bit-identical across x offset
  alignment" (forces the unaligned view directly): green on llvmpipe and
  the M1 fork, on fix backend and pre-fix backend alike - it pins the
  offset-independence contract, which today holds on both routes.
- Six canonical digests on a quiet M1 fork with the fix wheel: 3/3 runs
  DIGESTS_OK (with retry-on-mismatch); pre-fix wheel baseline 2/2 runs
  DIGESTS_OK. Prefill medians: fix q4 333.3-337.1 / 959.7-974.0 /
  1109.6-1119.0 tok/s vs pre-fix 333.3 / 959.7 / 1109.6 - within noise.
- Stock Mesa matrix with the fix wheel: DIGESTS_OK (stock pins incl.
  bf16_long c5be9207833d2a26).
- omarchy_matmul_family_tests + omarchy_runtime_tests: green on
  llvmpipe (20/20, 20,810,294 assertions; 41/41, 22,691) and on the M1
  fork and stock (20/20 and 41/41 on both drivers).
- Staging cost probe: odd-view arm vs aligned arm across the 8 Qwen
  shapes: delta -0.097 to +0.074 ms (noise); outputs bit-identical.

## Reproduce

```
python3 repro_offset_parity.py
```
