# Dense BF16 decode GEMV: compensated kernel NOT landable — receipt-only negative

Date: 2026-09-10
Agent: Bf16DecodeGemvClose
Branch: wave/Bf16DecodeGemvClose (kernel, selection, regression test, harness
scripts all on the branch; NOT merged)
M1 host: jwm1-linux (Apple M1 G13G B1, honeykrisp fork + stock Mesa)

## Question

Can a real dense f16/bf16 M=1 GEMV (subgroup reduction, wide loads) replace
the sequential `matmul.comp` path for decode q/k/v/o/down (M=1, N<4096) and
close the 5x BF16 decode gap, subject to the correctness bar: every affected
value at least as close to RNE(f64) as the current sequential kernel?

## What was tried before (state it forward)

`wave/Bf16DecodeGemv` 6f1612c2 (rejected at ba696cc5): a native-order GEMV
(deliberately reproducing Metal's reduced-precision accumulation), 3-4x
kernel bandwidth, 2.2-2.9x paired fork decode. Rejected because its bits
tracked Metal, and `receipts/2026-09-10-bf16-rootcause` proves Metal itself
deviates from RNE(f64) on decode q/k/v (up to 20 ULP / 14328 raw under bias
cancellation). The oracle is RNE(f64), not native bits.

## What this branch did

- `MatmulVecBF16Decode` (shaders/matmul_vec.comp, DECODE_GEMV variant): one
  subgroup per four output columns, uvec2 wide loads, per-lane double-single
  accumulation (Knuth two-sum), compensated shuffle-down reduction; selected
  for BF16 M=1 dense with 4-alignment and K%(4*subgroupSize)==0 guards;
  N>=4096 keeps the old MatmulVecBF16; everything else keeps the tile.
- Regression test (test_matmul_family.cpp): bit-exact vs RNE(f64) oracle on
  the Qwen2.5-0.5B decode shapes, plus a deep-cancellation case (partials
  ~40, exact sums ~1e-6) mirroring the decode v_proj bias-cancellation
  regime found by the root-cause receipt.
- Offline IEEE simulation (numpy + standalone C++ replicating the shader
  op-for-op on the exact test data): compensated kernel matches RNE(f64) on
  60/60 adversarial seeds where the sequential order misrounds 59/60.

## Blocker (the negative)

On real drivers the chained compensation does not deliver exactness:

- M1 fork AND stock (subgroup 32, hardware): random/model-shape data is
  bit-exact, but the deep-cancellation case deviates deterministically and
  IDENTICALLY on both drivers (fork 6/8 elements off, stock 7/8; e.g. col 5
  got 0x3698 vs oracle 0x3634, 100 bf16 ULP; col 7 0xb86f vs 0xb874).
  Driver agreement with each other, disagreement with IEEE simulation of the
  same op sequence, means real-driver shader execution does not preserve the
  two-sum error-recovery identities (`bb = total - sum` exactness,
  NoContraction semantics) that the compensation is built on.
- lavapipe (llvmpipe, subgroup 8): same failure class (27 bf16 ULP at
  k=896); K=32 (single add per lane) is exact, chained adds are not.
- The current sequential tiled kernel, by contrast, IS RNE(f64)-exact on
  these projections (root-cause receipt, GPU-confirmed). Any replacement
  that misses by tens of ULP under cancellation regresses the bar, and the
  rejected prior candidate already demonstrated that such flips move the
  BF16 262-token digest.

Therefore the correctness bar ("at least as close as the current kernel") is
not met on-device, and the kernel is not landable. The test on the branch
asserts bit equality on the hardware profile (subgroup 32) and the
documented accumulation-order bound on software drivers; on hardware it
currently FAILS, which is the honest gate doing its job.

## Timing evidence

Not qualified: timing without correctness is not evidence. The prior
candidate's paired numbers (11.70->32.04 / 11.25->28.37 / 10.03->22.64 tok/s
fork decode) remain the best estimate of the achievable speedup (~0.57 /
0.51 / 0.42 of native at the 30/262/1053 legs) if exactness is ever solved.

## Path forward (recorded for the next attempt)

1. The blocker is FP-semantic, not arithmetic: find a compensated form that
   survives driver compilation (e.g. integer/fixed-point mantissa
   accumulation over the bf16 product grid, or op orderings with no
   exactness assumptions), then re-run this branch's oracle harness.
2. fixed_bf16.py/oracle_f64.py captures (candidate/baseline, fork/stock) are
   scripted here; `oracle_f64.py --baseline` asserts the per-element
   closeness bar mechanically.
3. Baseline wheel: ~/src/mlx-main-b6d662a8/dist/...b6d662a...whl
   (sha256 98821134f4bcf306ab1d64d6885ddf3d3e8e932311283bbd11f97aecf6a8e439,
   the canonical 12-matrix wheel). Candidate wheel built at ef1ec2b:
   sha256 2aa00c3c75bb0c2bc852da6536aa8b71133cd6eceb6964a9e4a106c16888095e.

## Provenance

- Branch wave/Bf16DecodeGemvClose (d784cc8c, ef1ec2bd, 4e6d1bde); never
  merged.
- M1 decode test logs: fork + stock, 2026-09-10 (this directory's
  decode-*.log pulled from ~/src/mlx-gemvclose/receipt-gemvclose/).
- llvmpipe: full omarchy_matmul_family_tests 20/20 cases, 20,785,721
  assertions passed (x64, lavapipe) — existing suites unregressed.
