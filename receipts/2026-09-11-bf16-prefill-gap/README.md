# BF16 prefill 1K-context gap closed — attribution + landed gains (2026-09-11)

Status: LANDED on branch `wave/Bf16PrefillGap` (59e35b01 + 1f5c23d6,
rebased onto 5aab281f). Paired fork numbers, all six canonical digests
held, 12/12 primitive-level bit-identity. Nothing digest-moving; no
policy argument required.

## Headline (paired, same window, fork driver hk6f6afc8)

Prefill tok/s, bench_matrix canonical legs, base wheel 63c9a8d8 vs
candidate wheel 7a0b3f60:

| leg | base | cand | delta | fraction of native (base -> cand) |
|---|---|---|---|---|
| BF16 30/32 | 131.6 | 132.2 | +0.4% | 0.566 -> 0.568 |
| BF16 262/128 | 445.6 | 582.2 | **+30.7%** | 0.442 -> 0.578 |
| BF16 1053/32 | 457.2 | **677.2** | **+48.1%** | **0.276 -> 0.410** |
| Q4 short | 333.3 | 333.3 | 0 | unchanged |
| Q4 long | 952.7 | 966.8 | +1.5% | unchanged digest |
| Q4 1K | 1114.3 | 1116.6 | +0.2% | unchanged digest |

BF16 decode legs move <= +-0.3% (the causal change also removes 24
mask-add dispatches per decode token; within noise here).

Digest gates: every canonical pin HOLDS on the candidate wheel, 3 legs
x 6 cells: Q4 `7fd25a869ff21678` / `4cc08910089477fd` /
`7da83f06ec9f001d`; BF16 `f26175202f3dabe9` / `8690dc83246b39f8` /
`ff502900d2a179a5` (the native-matched rule-2 1K digest).

Bit-identity probe (primitive level, candidate vs base wheels,
fixed seeds, raw-bytes sha256): 12/12 PASS — 5 sdpa causal cases
(q_len=k_len=257 and 1052; offset>0; ragged k_len<q_len which takes the
kept additive path; decode shape q_len=1), 4 dense bf16 matmul cases
(aligned pair-path, gate_up shape, offset-2 and gap-894 scalar-path
controls), 3 dense f32 matmul cases (k%8=4 tail, k%8=0 control, k=7
tail).

## Per-class attribution of the 1K prefill token (the deliverable table)

Instrument: diag wheel `dev202609112320+diag.63c9a8d`
(MLX_OMARCHY_GPU_PROFILING), `scripts/profile_generate.py` prefill
phase markers, `receipts-work/analysis/attribute.py` (submission-window
phase contract). Dispatch count is 2244 at EVERY leg — the graph is
identical; all scaling is inside kernels. GPU busy: 244.3 ms (m=30),
580.9 ms (m=147 prompt), 2324.1 ms (m=1052).

Prefill-phase per-class GPU time and dispatch counts:

| class | m=30 | m=147 | m=1052 | dispatches | share at 1K |
|---|---|---|---|---|---|
| MatmulBF16Coopmat | — | 324.3 | 1114.1 | 92 | 47.9% |
| ElementwiseF32 | 3.9 | 30.7 | 379.9 | 94 | 16.3% |
| MatmulBF16 (16x16 tile) | 152.2 | 94.7 | 286.1 | 215 | 12.3% |
| MatmulF32 (16x16 tile) | 5.9 | 25.7 | 258.8 | 119 | 11.1% |
| MatmulF32Coopmat | — | 8.6 | 96.2 | 23 | 4.1% |
| SoftmaxF32 | — | — | 50.6 | 71 | 2.2% |
| MatmulVecBF16 | 41.1 | 40.3 | 40.5 | 194 | 1.7% |
| CopyGeneralBF16 | 8.4 | 13.4 | 31.1 | 356 | 1.3% |
| SwigluBF16 + casts + rms + rope + binary | ~21 | ~45 | ~103 | ~1114 | ~4.4% |

Instrument caveat (confirmed by Q4FusedProjections' same-night probe):
the diag wheel's isolation barrier inflates per-dispatch device time —
at the short leg it inflates mean dispatch cost to 57.6 us against an
8.99 ms/tok release wall. The absolute tick values in this table are
therefore profiled-run figures; the class ranking and the O(m^2)
identification were cross-checked against release-wheel wall clocks
(attribution_components probes) and are unaffected. All headline
prefill tok/s numbers in this receipt are release-path walls from the
unmodified release wheels, never profiled runs.

Note (leg definition): the component probe's `whole_prefill` computes
full-vocab logits and reads ~19% slower than the canonical leg, which
materializes lm_head only for the final row (census: last-row
MatmulVecBF16, 10.5 ms). The 458 ms "lm_head" in
receipts/2026-09-11-bf16-prefill-attention's component table is that
probe artifact, not canonical-leg cost.

### The superlinear terms (why the fraction fell 0.57 -> 0.44 -> 0.28)

1. **The causal mask add was the superlinear class.** The f32
   composition materialized a q_len x k_len f32 mask on the host
   (0 / -1e30) and added it to the scores: one ElementwiseF32
   dispatch per layer over the 14x1052x1052 f32 score tensor
   (62 MB out + 62 MB in + 4.3 MB mask per layer) — 379.9 ms at 1K
   (16.3%), 30.7 ms at 147, 3.9 ms at 30: O(m^2), ~8-22 GB/s.
2. **The pv matmul fell off the coopmat gate.** pv has k = k_len =
   1052, and k % 8 != 0 failed `coopmat_base`, dropping
   probs @ v onto the 16x16 f32 tile: 236.9 ms at 0.09 TFLOP/s
   (per-dispatch mean 10.3 ms, 23 dispatches). Constant per layer,
   but its share grows as native scales while the fork does not.
3. **The dense BF16 tile kernels ran at 0.53-0.57 TFLOP/s** at
   gate_up/down shapes (per-tick: 16.2 ms gate_up, 17.2 ms down at
   m=1052) versus the Q4 sibling's 0.9-1.03 TFLOP/s on the same
   hardware, and q/k/v (bias-bearing matmuls, excluded from coopmat by
   `use_c`) rode the 16x16 tile at 0.19-0.24 TFLOP/s (199.8 ms for q
   alone). Both tile kernels stream operands at ~35-40 GB/s effective;
   the BF16 kernel moves ~2.5x the operand bytes of Q4 (weights are
   bf16, not q4-packed) — bytes-bound staging, not arithmetic (the
   8x8x8 MMA ceiling is 1.44 TFLOP/s per receipts/2026-09-11-qmm-coop-datapath).

## The three landed mechanisms (all bit-preserving)

1. **Causal-mode softmax replaces the materialized additive mask**
   (primitives.cpp, f32 composition). The softmax kernel's causal mode
   (`flags` bit 0, the same SoftmaxF32 shader the bf16-fast arm uses)
   clamps each row to keys <= offset + position and stores exact zeros
   past them: exp(masked - max) underflows to +0.0, +0.0 never wins
   the max, and inserting or removing exact zeros cannot move an f32
   bit in the unchanged reduction tree, so probs are bit-identical to
   the additive path. k_len < q_len keeps the additive path (rows with
   zero admissible keys normalize to 1/k_len under the additive mask
   but store an all-zero row in causal mode — bit-equality only holds
   from offset >= 0). Deletes the 380 ms mask add AND the per-call
   q_len x k_len host mask fill.
2. **f32 coopmat k-tail** (shaders/matmul_coopmat.comp + gate). The
   shader masks `k < matrix_k` at staging (zero-padded shared patch;
   appended products are exactly +0.0 in the accumulator chain), so
   the k % 8 == 0 gate requirement is dropped for f32 (bf16 keeps it;
   its tail handles k % 16 == 8 only). pv moves from the 16x16 tile at
   0.09 TFLOP/s to the coopmat route: 236.9 -> ~5 ms class.
3. **bf16 coopmat uvec2 pair staging** (shaders/matmul_coopmat_bf16.comp,
   flags bit 8 set host-side when offsets/gaps/strides are 4-byte
   aligned). The two word-adjacent orientations stage two bf16 pairs
   per load from aliased uvec2 buffer views and unpack in the same
   (row, du) coordinates; staged words are identical to the scalar
   path. Pure transport; measured as part of the combined gain below.

Per-leg wall deltas exceed the sum of the three class costs
(-748 ms measured vs ~-750 ms estimated GPU-side) — consistent within
the noise of one paired window.

## Honest bounds

- One paired window, single reps per cell (bench_matrix default);
  the 1K delta (+48.1%) is ~30x the rep spread of the committed
  matrix, but the receipt carries no multi-day repetition.
- The probe covers the production orientations; exotic flag
  combinations (both operands column-major, tail steps with bit 8)
  route to the unchanged scalar path by construction and by gate.
- The bf16 tile kernel remains bytes-bound at ~0.56 TFLOP/s pre-fix;
  after the pair staging it is still under the Q4 sibling's rate. The
  64x64 tile (halved operand bytes per output) is the next lever and
  was not attempted tonight.
- q/k/v bias routing (use_c excludes coopmat) was identified and NOT
  changed: fusing the bias would change the rounding sequence; a
  bit-exact split needs a dedicated look.
- Window contamination: none. The window ran alone (flock held,
  driver pin verified in-window); one earlier attempt of the same
  window aborted in-phase B on a probe bug before any timing ran.

## Provenance

- Host: jwm1-linux (Apple M1 G13G B1), driver package verified
  in-window: mesa-honeykrisp-omarchy 26.3.0.devel.hk6f6afc8-1.
- Base wheel: mlx_omarchy-0.32.2.dev202609112357+63c9a8d8
  (sha256 2a32a2611ce8fee2672d2bd0955f2d4c6ee43b4efa33ddd2b7ce2e020cbb28e4).
- Candidate wheel: mlx_omarchy-0.32.2.dev202609120013+7a0b3f60
  (sha256 4e4905a8da04fa5a9aabf66d63cfe2bd9dc4dd09e45449c0dc22a79c0d4671bd);
  built from 7a0b3f60 whose overlay equals code commit 1f5c23d6
  (the commit above it is receipt artifacts only).
- Census wheel: dev202609112320+diag.63c9a8d (profiling build).
- Every GPU second under one top-level flock /tmp/m1-gpu.lock;
  wall-anchored timings only (device clocks undercount ~2.07x).
- Artifacts: m1-logs/ (census NDJSON + markers, per-leg analysis
  files, probe NDJSON, matrix.json per cell), window_a.sh /
  window_b.sh / probe_bitexact.py / prefill_components3.py /
  attribute.py.
- Branch: wave/Bf16PrefillGap, pushed; Main integrates.
