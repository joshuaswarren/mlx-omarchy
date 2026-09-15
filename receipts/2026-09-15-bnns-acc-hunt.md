# BNNS accumulator hunt vs captured native gates (2026-09-15)

Verdict: **no candidate form found** — best single form 57.20 % exact on the
75,257 captured singleton gate lanes; the >95 % gate for a live decode is
**not met**, so no candidate was implemented, jwm1 was never contacted, the
decoder stays unlanded, and `63c1d3cf` is untouched.

## Data and protocol

- Native gate values: `/tmp/gate-preact-art/native_gates.npz` (probe v7,
  macstudio CPU_ONLY fused `ios18.lstm`). Each lane's native sigmoid/tanh
  output is pinned uniquely by the three-multiplier intersection
  (`sizes == 1`); 75,257 of 76,800 lanes are singleton and finite
  (L0_i 9567, L0_f 8417, L0_o 9280, L0_g 9599, L1n_i 9599, L1n_f 9599,
  L1n_o 9596, L1n_g 9600). A form z scores 1 per lane where
  `LUT16[fp16(z)] == captured`.
- Inputs: `traces.npz` x-rows, h-rows; weights `concat_0..5` (impl2).
  **Correction to modehunt.py**: the v7 capture feeds native
  `next_hidden[0]` (`tXXXX_nh[0]`) as the L1 sequence row
  (`run_gate_preact7.py` `load_cases`), but modehunt scored L1n against the
  embedding row `x0`. Every L1n host score in `modehunt.json`
  (f64_exact 48–477, gpu 5724 etc. aside) is an artifact of that wrong row.
  With the correct row, host f64-exact on L1n jumps from ≤477 to
  4402–5487 per gate — L1n and L0 sit at the same plateau.
- Baseline replication: this harness reproduces `modehunt.json` L0 numbers
  exactly (f64_exact 5975/3541/4461/7816, gpu_dump 6057/3652/4581/7838),
  confirming the protocol before the sweep.

## Rates (exact match / 75,257 singleton lanes, host numpy, no device)

| form | L0_i | L0_f | L0_o | L0_g | L1n_i | L1n_f | L1n_o | L1n_g | total | % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f64 exact sum, bias after | 5975 | 3541 | 4461 | 7816 | 5487 | 4402 | 4857 | 5248 | 41787 | 55.52 |
| f32 sequential/pairwise (all orders, blocks 4–320, packed k 1–64) | 3056* | 207* | 1635* | 5773* | – | – | – | – | 41789 | 55.53 |
| vulkan 3-op (`r16(r16(sx16+sh16)+b16)`) | 6058 | 3652 | 4581 | 7837 | 5725 | 4702 | 5140 | 5348 | 43042 | 57.19 |
| jwm1 gpu_dump (CR fp32 dots) | 6057 | 3652 | 4581 | 7838 | 5724 | 4703 | 5139 | 5349 | 43043 | 57.19 |
| **best single: fp16-rounded half-sum partials, fp16 combine** | 6058 | 3652 | 4581 | 7837 | 5724 | 4702 | 5142 | 5349 | **43045** | **57.20** |
| sum of per-group bests (8 forms, overfit bound) | | | | | | | | | 43199 | 57.42 |
| BNNS RE contract blk8_H1 (BnnsLstmRe) | 5807 | 3345 | 4313 | 7742 | 4861 | 3880 | 4175 | 4840 | 38963 | 51.77 |
| BNNS RE contract best (blk32_H1) | 5963 | 3526 | 4453 | 7802 | 5427 | 4327 | 4681 | 5166 | 41345 | 54.94 |

*f32-class numbers in that row are modehunt's L0 figures (all f32 orderings
agree within 6 lanes; with the corrected L1n rows the f32 class lands at
41789 total, inside the same plateau).

Swept and rejected (≈150 forms, all ≤ 57.20 %): f64 once/sequential;
f32 sequential, numpy-pairwise, recursive trees (base 8/16); k-blocks
B ∈ {4,8,16,32,64,128,160,320} × {f32-seq, f32-pairwise, f64} combines;
packed k-interleaves s ∈ {1,2,4,8,16,32,64}; descending k; split-K x/h with
every bias placement (after, between, seeded, f16/f32/f64, per-half fold);
fp16 carry every N ∈ {4,8,16,32,64,128,320}; products rounded to fp16, bf16,
tf32(10-bit) × {f64, f32 pairwise, f32 seq, blocks}; fp16-rounded block
partials B ∈ {2..1280} × {fp16-seq, fp16-pairwise, f32, f64} combines;
bias-seeded accumulators; two-pass max-scaling (numerically identical to
one-pass — exponent scaling is exact, no overflow/subnormals in range) and
power-of-2 scale/shift (rounding-invariant) recorded as no-ops.

## BnnsLstmRe's disassembled contract scores below the plateau

The macstudio RE (macOS 26.6.2, BNNSGraphContextExecute_v2, nodes
`matmul_ih`/`matmul_hh` → `lstm_elementwise`) recovered: B = 8 for N=2560,
per-term `acc = rnd16(acc + x·w)` (fp16 FMLA, ascending k), per-block
`y = rnd16(y + acc)`, ih then hh folded into one buffer, bias outside the
GEMV. Scored here via exact fp64-then-round16 emulation: **blk8 51.77 %**
(H1), H3 50.54 %, bias-seeded 46.36 %; B=16 53.59 %, B=32 54.94 % — all
below even f64-exact. The recovered kernel therefore does not explain the
captured gates, and the direction is informative: it is *further* from the
capture than correctly-rounded arithmetic.

## The residual is non-accumulation

- Long-tailed misses: exact-z lands inside the native unary preimage on
  57–81 % per group; miss distances beyond that are 2–14 arg ulps at q90 but
  reach 31–96 arg ulps at q99 (up to 6.5 % relative in z). No accumulator
  rounding at fp16/fp32 scale produces that.
- Unary-composition transform rejected: scoring `σ = r16(0.5+0.5·tanh16(z/2))`
  (and the tanh dual) collapses L0_g from 7837 to 1151 — the fused unaries
  are the CR σ16/tanh16 tables, as the v7 bias-only anchor already pinned.
- The v7 anchor (zero row, z = bias exactly) matched `LUT16[bias]` on every
  checked lane, so the pinning works where z is known; the delta enters only
  where z comes from a real GEMM, yet no GEMM form moves toward it.

Agreed with BnnsLstmRe: the remaining delta is most likely in the capture
inversion (LUT-preimage artifacts when backing gates out of σ16 under the
multiplier product) or a capture-host vs macstudio libBNNS skew — not in any
accumulation order. Their end-to-end seeded-probe check (fused op vs full
emulated chain incl. unaries, on macstudio) is the experiment that
separates those two.

## Status

- Acceptance "land only if >95 % exact and misses ±1 ulp": not met on both
  counts — misses are long-tailed, best rate 57.20 %.
- Nothing landed; no live decode; `/tmp/m1-gpu.lock` untouched; jwm1 and
  jw16 never contacted. `63c1d3cf` not merged.
- Scoring lane list (singleton mask) shared with BnnsLstmRe for their
  preact-level scoring once direct native preacts (their 76733/76800
  capture) are available; arg-exact scoring on those lanes supersedes this
  interval-based protocol.

## Artifacts

`receipts/2026-09-15-bnns-acc-hunt/`:
`acc_hunt.py` (main sweep; reproduces modehunt baselines exactly),
`bnns_acc_hunt.json` (≈80 forms × 8 groups + offset histograms),
`fp16blk_hunt.json` (fp16 block-partial family),
`longshot_hunt.json` (bias seeds, mixed partials),
`bnns_contract_hunt.json` (BnnsLstmRe contract, B/H variants),
`composition_test.json` (σ/tanh transform rejection),
`residual_structure.json` (per-group preimage distances).

resolved_model: this session (`zai/glm-5.3-flash`).
