# BNNS fused-LSTM accumulator: contract closed, landing handed off (2026-09-15)

Verdict: **contract closed.** The fused `ios18.lstm` gate arithmetic is the
BNNS GEMV contract recovered from the macstudio libBNNS disassembly
(sibling receipt 2026-09-15-bnns-lstm-re) and independently validated here
against the probe-v7 captured native gates: **99.98 % exact
(75243/75257 singleton lanes)**, and all 14 misses are the known ambiguous
σ16 table arguments (±1 output ulp preimage ties), not contract errors.
Live free decode on jwm1 reproduces the native trace token-exactly
(`"first_divergence": null`, 104/104 emissions; emission 101 = native 7892).
Landing of the decoder is owned by LandHostLeaves taking BnnsLstmRe's
verified implementation; this receipt is the accumulator-side evidence.

## The contract

1. **Gate preacts**: ONE fp16 GEMV over `concat[x; h0]` with repacked
   `[wi | wh]` — not two matmuls (separate ih/hh scored 426/640 on the
   macstudio probe vs 640/640 for concat). k runs in blocks of B = 128 for
   N = 2560 (the RE's first B=8 reading was wrong; the 16384 divisor is over
   a 128-lane tile = 32768 B budget / 256 B per k-step). Each block: fp16-FMA
   chain from `+0.0`, ascending k, single rounding per term (`FMLA .8h`);
   after each block `y = rnd16(y + acc)` (`FADD .8h`), first block stores.
   Bias: one fp16 add after the fold. No AMX, no fp32 accumulation.
2. **Unaries**: dense sigmoid/tanh tables indexed by fp16 bit pattern,
   built from the macstudio direct fused-op reads (`sigma_full.npz` /
   `tanh_full.npz`; fills: σ(±0)=0.5, saturation at ±30, 17 ambiguous args
   CR, tanh clamps ±(1−2⁻¹⁰), tanh(−0)→+0).
3. **Cell**: `c1 = rnd16(rnd16(sig_f*c0) + sig_i*tanh16(g))` — forget
   product rounded (FMUL), then FMA-fused with the unrounded input product
   (single-rounding 453/640, pairwise 543/640, this 640/640). Answers the
   unary-mac receipt's open micro-question.
4. **Hidden**: `h = rnd16(sig_o * tanh16(c1))` (rounded tanh; raw 360/526).

BnnsLstmRe's macstudio seeded-probe check reproduces the fused op
bit-exactly under this contract (640/640 next_cell AND next_hidden on the
full dense model, plus a 0/1/2-product ladder) — libBNNS is deterministic
and fully modeled.

## Hunt narrative (why the disassembly was needed)

Host numpy search over ≈150 accumulation forms against the v7 singleton
lanes plateaued at **57.20 %** (fp16-rounded half-sum partials + fp16
combine 43045/75257; vulkan 3-op class 43042-43043; f64 exact 41787; every
f32 ordering, packed interleaves, split-K, product roundings to
fp16/bf16/tf32, fp16 carries and block partials, bias seeds, two-pass and
power-of-2 scale variants all within 55.3-57.2 %). Misses were long-tailed
(2-14 arg ulps at q90, up to ~96 arg ulps / 6.5 % relative at q99) — a
non-accumulation signature. The σ≈0.5+0.5·tanh(z/2) composition transform
collapsed L0_g to 1151/9599, pinning the unaries as the CR tables.

**Correction to modehunt.py**: the v7 capture feeds native `next_hidden[0]`
(`tXXXX_nh[0]`) as the L1 sequence row (`run_gate_preact7.py load_cases`),
but modehunt scored L1n against the embedding row `x0`. Every L1n host score
in `modehunt.json` is an artifact of that wrong row; corrected, L1n
f64-exact sits at 4402-5487 per gate, the same plateau as L0.

## Scoring the closed contract

Scored exactly like the hunt: form z → fp16 → dense LUT → compare to the
captured native unary output on singleton lanes.

| group | ok | contract_B128 exact | misses |
| --- | --- | --- | --- |
| L0_i | 9567 | 9565 | 2 |
| L0_f | 8417 | 8412 | 5 |
| L0_o | 9280 | 9277 | 3 |
| L0_g | 9599 | 9599 | 0 |
| L1n_i | 9599 | 9598 | 1 |
| L1n_f | 9599 | 9597 | 2 |
| L1n_o | 9596 | 9595 | 1 |
| L1n_g | 9600 | 9600 | 0 |
| **total** | **75257** | **75243** | **14** |

Sensitivity: B=64 55.89 %, B=256 52.61 % — B=128 is load-bearing.

**The 14 misses are the ambiguous σ16 table arguments.** Every missed lane's
contract preact sits on one of the σ16 LUT's ambiguous arguments —
z ∈ {−10.3984 (5 lanes), −0.5103/−0.5112 (4), −1.4658 (2), 1.0977 (2),
plus one more at −0.5103} (`miss_tie_check.json`) — exactly the argument set
of `ambiguous_sigma_remapped.json` (−10.4, −0.5103, −0.511, −1.466, 1.095-1.1).
These are the σ16 table entries whose dense macstudio reads came back NaN;
the ±1-ulp disagreement is which rounding fills the table entry, not the
GEMV. On directly-pinned lanes the contract has no misses.

## Live decode (jwm1-linux, independent reproduction)

`flock -x -w 120 /tmp/m1-gpu.lock` (never stolen),
`PYTHONPATH=/tmp/vulkan-tdt-142 control-venv/bin/python` (mlx
`0.32.2.dev202609131130+e4d079c`, `Apple M1 (G13G B1)` Honeykrisp).

Two independent contract implementations were run through the free decode:

1. This receipt's implementation (`bnns_decode_result.json`): wall 3.43 s,
   `emission_matches 104/104`, `"first_divergence": null` — emission 101
   emits the native token **7892** (frame 373, duration 0). Every previous
   gate form diverged at emission 101 to 8029.
2. The landed-module re-validation (`bnns_landed_check.json`, worktree
   overlay files): `104/104`, `"first_divergence": null`; single transitions
   on all 15 native-input traces `next_cell 19170/19200`,
   `next_hidden 19163/19200` bit-exact (residue = the same ambiguous-σ16
   ties). BnnsLstmRe's shipped variant additionally reports decision-142
   logit gap 0.0.

## Landing ownership

Per instruction: the decoder landing itself is **not** pushed from this
session. LandHostLeaves takes BnnsLstmRe's verified decoder implementation
(shipped code `overlay/tools/coreml/vulkan_decoder.py`, receipt
`2026-09-15-bnns-lstm-re.md`). This receipt carries the accumulator-side
evidence and the harness corrections. `63c1d3cf` remains unmerged; jw16
remains owned by RopePairNondonate; nothing else in the shared checkout was
touched.

## Artifacts

`receipts/2026-09-15-bnns-acc-hunt/`:
`acc_hunt.py` + `bnns_acc_hunt.json` (≈80-form sweep, modehunt replication),
`fp16blk_hunt.json`, `longshot_hunt.json` (extended families),
`bnns_contract_hunt.json` (B=8/16/32 contract variants),
`contract_closed_hunt.json` (closed B=128 contract sweep, 99.98 %),
`contract_misses.json` + `miss_tie_check.json` (14 ambiguous-σ16 lanes),
`composition_test.json` (σ/tanh transform rejection),
`residual_structure.json` (preimage distances),
`bnns_decode_test.py` + `bnns_decode_result.json` (first live decode),
`bnns_landed_check.py` + `bnns_landed_check.json` (landed-module validation).

resolved_model: this session (`zai/glm-5.3-flash`).
