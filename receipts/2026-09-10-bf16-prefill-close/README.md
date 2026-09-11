# Dense BF16 prefill coopmat: restructure to the qmm staging form

Branch: `wave/Bf16PrefillClose` (commits 79e161c and this receipt).
Parent: origin/main `1242cb32`.

## The gap

The fork's BF16 prefill ran at 0.54 / 0.34 / 0.23 of the committed
native M1 baseline (232.6 / 1007.7 / 1655.7 tok/s at the 30 / 262 /
1053 legs; `receipts/native-baseline-2026-09-06`), while Q4 prefill was
already at 1.13 / 0.80 / 0.60. Per-layer projections are ~89% of BF16
prefill FLOPs, so the dense kernel was the binding term.

## What dispatched at the leg shapes

Static dispatch analysis (`dispatch_matmul`, primitives.cpp) plus the
window A2 shader bench and attribution probes:

- m = 262 / 1053 (fork): the seven projection matmuls dispatch
  `MatmulBF16Coopmat` (the 32-lane staged kernel, 8-wide k steps,
  scalar bf16 element loads, 16 f32 accumulators per lane). Attention
  runs the f32 composition (scores on `MatmulF32` via alpha != 1,
  probs @ v on `MatmulF32Coopmat`), rope/softmax in f32, four casts per
  layer. Stock dispatches `MatmulBF16` (16x16 scalar tile) everywhere.
- m = 30: `params.matrix_m >= 32` routes EVERYTHING to the scalar tile
  on both drivers; the coopmat kernel never runs.

The old profile in `receipts/2026-09-08-recover-dense-bf16` (stale
pre-fused-chain, but structurally indicative) plus the fixed
attribution probe arms show the projection kernel dominating leg time
at 262 / 1053 (linear token scaling: 0.683 s vs 2.769 s for 4.02x
tokens), so the coopmat kernel's instruction mix was the target.

## The change (minimal, bit-identical)

`matmul_coopmat_bf16.comp` restructured to the form that won the Q4
coopmat bakeoff (0dbb1d1d: two subgroups per 32x32 tile, 16-wide k
steps; NOT the losing variants - 4 subgroups, 32-wide staging,
double buffering):

- 64-lane workgroup, each subgroup owns a 16x32 half (8 accumulators
  per lane instead of 16).
- One barrier per 16 k instead of per 8.
- Word-pair loads: one 32-bit load covers two bf16 elements (exact
  16-bit left-shift widening) instead of two scalar element loads.
  Scalar element loads remain for the two orientations without
  k-adjacency (column-major lhs, row-major rhs).
- The k % 16 == 8 tail is one final 8-wide step (the gate only
  requires k % 8 == 0).

Per output the accumulator sequence is the same ascending-k chain of
identical 8x8x8 coopMatMulAdd ops on identical f32-widened operands, so
stored bits are identical by construction, and proven identical by the
window A2 bench: 36/36 shape x fill cells with `cand_vs_base_mismatch=0`
(small-integer exact fills and fractional bf16 fills, all four model
projection shapes at m in {30, 262, 1053}; the m=30 cell is a
bench-level dispatch, not a production route). The host gate adds a
shared-memory check (4 KiB, the qmm constant) mirroring the qmm route.

## Measured kernel effect (window A2, fork driver, wall-clock medians)

| m | shape (k x n) | base us | cand us | speedup | cand TF/s |
|---|---|---:|---:|---:|---:|
| 262 | 896x896 | 1313.0 | 969.0 | 1.36x | 0.43 |
| 262 | 896x128 | 641.2 | 284.5 | 2.25x | 0.24 |
| 262 | 896x4864 | 5732.8 | 4221.6 | 1.36x | 0.54 |
| 262 | 4864x896 | 7112.3 | 5017.2 | 1.42x | 0.46 |
| 1053 | 896x896 | 4031.9 | 2838.4 | 1.42x | 0.60 |
| 1053 | 896x128 | 717.3 | 574.1 | 1.25x | 0.33 |
| 1053 | 896x4864 | 21058.2 | 15174.9 | 1.39x | 0.55 |
| 1053 | 4864x896 | 22904.9 | 15613.5 | 1.47x | 0.59 |

Per-layer projection chain at m=1053: 74.5 ms -> 52.8 ms (1.41x).
Stock tile numbers for the same shapes are 2.1-3.2x slower still.

## llvmpipe screening

This machine's llvmpipe (and the M1's) does not expose
VK_KHR_cooperative_matrix at subgroup 32 (probe: subgroupSize 8,
coopmat false), so the coopmat kernels cannot dispatch there by
construction. The local llvmpipe leg
(`tools/bf16-prefill-bench/out-llvmpipe-local.ndjson`) proves the
harness, the tile fallback, and the dispatch plumbing stay correct
(m=30 all four shapes, integer + fractional fills, dead-row guard,
float64-reference rows green), and the family suite's llvmpipe legs
prove the same for the backend route. The coopmat kernel's
every-element proof is a hardware leg: the family suite's
"matches host on every row" case runs the full shape x orientation
sweep on the M1 fork driver (window B phase 5, both wheels).

## Regression tests

`tests/omarchy/test_matmul_family.cpp`: the existing dense-bf16
every-row case already sweeps the model projection shapes and all four
orientations against the float64 host reference on coopmat devices.
Added rows pin the restructured kernel's k-tail path (k % 16 == 8 and
k < 16), which the production shapes (k = 896 / 4864) never hit:
`tail-m32-k24-n34`, `tail-m33-k40-n18`, `tail-m48-k8-n16`.

## Negative result recorded: the m = 30 coopmat route

The bench measured the restructured coopmat kernel at m = 30 against
the scalar tile that production routes to. Mixed per shape (gate_up
2.28x faster, kv 2.5x SLOWER - the tile wins tiny-n), net kernel-time
~1.34x per layer, but: (a) the 30-token leg is host/dispatch-floor
paced (DecodeAttributionFinish/HostPathOverhead findings; whole-leg
gain would be a fraction of the kernel gain), and (b) rerouting m=30
changes the fork BF16 short digest, requiring the full
parity-id-policy amendment procedure for a sub-15% expected leg gain.
Not landed. The binding constraint on the 30-token leg is the
host/dispatch floor, not the dense kernel.

## Paired M1 verification (window B)

{MATRIX_RESULTS}

## Digest gates

{DIGEST_RESULTS}
