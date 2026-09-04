# Quantized GEMV for decode rows: M1 A/B, +37% 4-bit decode

2026-09-04, jwm1 (Apple M1 G13G B1, Honeykrisp 1.4.354). Before = main
eeff51b. After = subgroup-qmm-mat 1e57f4f, based on that main. Both wheels
built on the M1 from detached checkouts (sha256 fe5cb6e9 vs 7eb78ebd),
fresh venv per side, provenance verified=match on every run. Qwen2.5-0.5B,
pinned 64 tokens, EOS suppressed, decode over the last 63; bf16 eager.

## What the change is

Main had no single-row path for `QuantizedMatmul`: every decode step ran
each quantized matmul through the general 16x16 tiled kernel. The branch
adds `qmm_vec.comp`, a GEMV dispatched when `matrix_m == 1`, with a
subgroup-reduction variant selected when the device reports
`subgroupSize == 32` and the ARITHMETIC feature bit (the M1 does; llvmpipe
does not and takes the shared-memory tree).

## Equivalence, three ways

1. `omarchy_matmul_family_tests` on the Apple GPU: 7/7 cases, 17,336
   assertions, including the new subgroup case with a bound derived from
   reduction depth and accumulator type (not fitted).
2. Tree kernel (main wheel) vs subgroup kernel (branch wheel), same inputs
   through `mx.quantized_matmul`, 144 decode shapes (bits 4/8, group 32/64,
   f32/f16/bf16, n in 1,7,9,37,64,256, k in 192,896): bf16 bit-identical on
   48/48; f16 within 1 ulp on 2/48, identical on 46; f32 within 1.5e-6 of
   the row maximum (summation order). Scripts: jwm1:/tmp/qmm_dump.py,
   /tmp/qmm_compare.py, /tmp/qmm_summary.py; dumps /tmp/qmm-{before,after}.npz.
3. Greedy generation, 128 tokens, 4-bit model: token ids identical across
   the two wheels.

## Decode, tok/s, five runs

| dtype | before | median | after | median | delta |
|---|---|---|---|---|---|
| 4-bit | 12.85 12.91 12.73 12.77 12.83 | 12.83 | 17.01 17.47 18.06 17.68 17.61 | 17.61 | +37% |
| bf16 | 8.74 8.77 8.75 8.77 8.76 | 8.76 | 8.75 8.73 8.73 8.70 8.77 | 8.73 | noise |

bf16 is untouched by design: the path is quantized matmul only.

## Attribution: the GEMV, not the subgroup

The branch's microbenchmark (`tools/subgroup-bench`, two kernels differing
only in the reduction body, GPU-timestamped, on this device) reports
`ratio_gpu` of 36,667/35,667 ns at 65,536 groups and 13,333/13,625 ns at
4,096 groups: subgroupAdd(float) and the five-round shared-memory tree
cost the same. So (a) the claim in the old shader comments that
Honeykrisp software-lowers float subgroup arithmetic is refuted - it is
not slower - and (b) the 37% is not the reduction. It is dispatching a
GEMV for a one-row matmul instead of a 16x16 tile.

That GEMV was built on 2026-09-02 (`receipts/2026-09-02-gemv-decode/`)
and stood down as "the wall is 91% host". Every timing in that receipt
was taken with 1 of 8 CPU cores online, the condition the project later
found had contaminated its early host-side numbers. The kill was wrong.
The contributor guide's killed-strategies row is corrected in the same
commit as this receipt.

## Open

- The tree variant of `qmm_vec` runs on llvmpipe only (the M1 always
  takes the subgroup path). Its correctness is battery-covered; its speed
  on any real non-subgroup device is unmeasured and there is no such
  target today.
- 37% wall from a kernel change against a ~19.5 ms/token GPU-busy figure
  (receipts/2026-09-02-gpu-profile-decode.md) means the tiled kernel was
  costing host time too - most likely the per-dispatch fixed cost of a
  16x16 workgroup grid for a 1-row problem. Not measured; the next
  per-kernel GPU-time breakdown on the M1 should attribute it.

Logs: jwm1:~/benchq/logs/subgroup/ and ~/benchq/logs/subgroup-ab-session.log
(second session, from "start 11:21:48"); microbench /tmp/subgroup-bench.log.
