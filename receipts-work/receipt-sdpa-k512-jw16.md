# receipt: sdpa k=512 decode arm — REJECTED-premise (jw16 / t6001, 2026-09-24)

Assignment: extend the composition-exact bf16 SDPA decode arm to the contract
key depth k=512, prove it faster than the composed fallback, gate it, merge to
main. Verdict: **REJECTED-premise** — acceptance 1's microbench refuted the
premise before any gate cycle. Full numbers below; artifacts in
`/var/tmp/sdpa-k512-out/` (micro_cand_bfbf7581.json, micro_ctl_06add05a.json,
micro_crossover_bfbf7581.json, micro windows under gpu_window.sh).

## What the microbench proved

Bitwise identity (arm vs composition) holds at every k tested, both
directions, on both stacks — raw uint16 bit hashes equal and f32 views equal:
`ident_arm_first = ident_comp_first = f32_equal_both = true` for every row.
Width-256 arm output hashes: k=32 `7bd5d650f5c83750`, 448 `ba9bd5d83e48a9cc`,
512 `913e0776cba7899d`, 544 `967e159972a8e9e6` (identical across both stacks).

Timings (us/launch, family_bench amortized methodology: 64 enqueues per timed
flush, median of 30 trials, 3 warmups, `mx.clear_cache` per trial; shapes
B=1 T=1, q (1,8,1,256), k/v (1,2,k,256), bf16, scale 256^-0.5, no mask; arm =
default env, composition = `MLX_OMARCHY_SDPA_DECODE_NATIVE=0`, which eval_gpu
reads per call):

Window 1 — candidate stack `/var/tmp/normfold-venv` (wheel
0.32.3.dev202609242003+bfbf7581; code-identical to branch tip 8b5610e6, the
two commits differ by a receipts file only):

| k   | arm    | comp   | faster     |
|-----|--------|--------|------------|
| 448 | 452.09 | 303.68 | comp 1.49x |
| 512 | 496.91 | 348.03 | comp 1.43x |
| 544 | 533.64 | 347.00 | comp 1.54x |
| 32  | 138.96 | 276.98 | arm 1.99x  |

Window 1 — ctl stack `/var/tmp/v072-venv-fused` (installed wheel
0.32.3.dev202609241305+06add05a):

| k   | arm    | comp   | faster     |
|-----|--------|--------|------------|
| 448 | 450.22 | 304.98 | comp 1.48x |
| 512 | 506.58 | 350.38 | comp 1.44x |
| 544 | 534.18 | 366.43 | comp 1.46x |
| 32  | 135.45 | 267.78 | arm 1.98x  |

Window 2 — crossover sweep, candidate stack (same protocol):

| k   | arm    | comp   | note                       |
|-----|--------|--------|----------------------------|
| 32  | 102.41 | 260.07 | arm 2.5x                   |
| 64  | 178.96 | 256.76 | arm                        |
| 96  | 180.98 | 281.04 | arm                        |
| 128 | 173.72 | 282.44 | arm                        |
| 160 | 199.60 | 288.77 | arm                        |
| 192 | 235.02 | 281.90 | arm                        |
| 256 | 285.14 | 292.27 | ~tie                       |
| 448 | 456.41 | 326.78 | comp                       |
| 512 | 502.95 | 339.78 | comp                       |
| 544 | 525.58 | 360.55 | comp                       |

Arm cost model: ~60-70 us fixed (one dispatch) + ~0.86 us/key
(173.72@128 -> 502.95@512). Composition: ~270 us fixed (ten dispatches)
+ ~0.15 us/key. Crossover sits near 256-300 keys on today's sweeps; the
author's 300-rep crossover (16df8b6b) tied at 128. Both agree the contract
regime k=448..544 is composition-favored and k<=192 arm-favored.

## Premise resolution

The family row `sdpa[k=512] = 504.11` (families-linux-t6001.json) measured the
**arm**, not the composition: the shipped window engages the width-256 arm at
`12 <= k_len <= 2048` (the `k_len <= uint32_t{2048}` bound at
primitives.cpp:12056), so the contract decode k=512..544 already ran the arm
on every stack built from main. The true composition at k=512 is ~340-350 us.
The assignment's premise — "arm disengaged at contract; extend and prove
faster" — was therefore doubly refuted: nothing needed extending, and the
route main already takes is the slower one.

## Acceptance 1 verdict

Arm at k=512 across three runs (two stacks, two windows): 496.91 / 502.95 /
506.58 us. No clear win against even the stale 504 figure, and a 43-48% LOSS
against the measured composition (348.03 / 350.38 / 339.78). REJECTED-premise;
no gate cycle, no family-row move, no merge to main.

## Why key-split p1/p2 cannot be composition-exact

The composition's observable arithmetic pins two order-sensitive reductions.
PV output[d] = one strict ascending-key f32 chain per element (`acc += a*b`
over ascending 16-wide zero-padded tiles, matmul.comp order), and the
softmax denominator is per-lane strided sequential exp-sum plus the 256-lane
binary tree (softmax_suffix.comp order). Splitting keys across workgroups or
blocks yields partials combined as (sum_A + sum_B) or (acc_A + acc_B), which
rounds differently from the sequential chain in general — bit-identity dies.
The score dots are per-key independent and max-over-keys is exact under any
split, but those two reductions cannot be reordered, so the acceptance's
"if the composition's softmax/matmul order can be preserved" resolves to NO
for every multi-pass key split, including the p1/p2 combine-kernel sketch.

## Design for what WOULD win

Keep the exact arithmetic; attack the latency. The serial walk is
DRAM-latency-bound: each key-step's V row load feeds a dependent FMA and only
8 workgroups (one per head) are resident, so nothing hides the ~600ns
memory latency — hence ~0.86 us/key. Design: one workgroup per head, byte-
identical FMA chains, softmax trees, and stores, with a software-pipelined
prefetch that stages K/V rows for keys k+1..k+P (P=2..4, compile-time -D)
into registers/shared while the chain consumes key k. Prefetching changes no
arithmetic, so the route stays bit-identical to the composition at every k by
construction, verified with this receipt's both-directions harness. Target:
per-key cost drops to FMA issue + shared read (~10-30 cycles), giving
arm(512) ~ 65 us fixed + well under 100 us of chain vs ~340 us composition —
a credible 2x+ win in the contract regime, worth a follow-up ticket with the
full gate battery.

## Corrective on this branch (not merged)

`overlay/mlx/backend/omarchy/primitives.cpp`: the width-256 k window cap is
restored to the documented 128 (`k_len <= (head_dim == 256 ? 128 : 2048)`),
making the code match the comment 16df8b6b shipped with, and the comment now
cites this receipt. Digest-safety: route choice is perf-only and both routes
store identical words at every k (measured above), so the token stream — the
only thing the digests hash — cannot move; the gate battery was reserved for
the winning-arm path that never materialized. Main still carries the <=2048
bound and routes the contract regime to the slower arm; landing the cap on
main needs its own update push (parent's call).

## Provenance

- Host: jw16 (jw16mbp1-linux, aarch64, M1 Pro / t6001), GPU discipline held:
  llm-inference stopped, /tmp/m1-gpu.lock taken, service restored after each
  window (HEALTH_OK=1 + live completion probe, twice).
- Branch: agent/t6001-norm-swarm at 8b5610e6 + the corrective commit.
- Stacks: candidate wheel bfbf7581 (normfold-venv clone), ctl wheel 06add05a
  (v072-venv-fused, installed 2026-09-24 13:05 UTC, source checkout since
  wiped with /dev/shm/sdpa256).
- GitHub main tip observed from relay host: 9fb8b675 ("merge: land the SDPA
  hd256 decode arm plus its 24-commit GPU/runtime fleet lineage").

---

# ADDENDUM: full gate battery on the cap-fix wheel (2026-09-24, parent-directed)

Candidate = wheel 0.32.3.dev202609242059+2b8bdd7f
(sha256 0416a786bc8a3c8f64b1dd2d4a36a72e5bf8e95e22e3dceac387a62bfb7d7c51,
built from branch tip 2b8bdd7f), candidate venv /var/tmp/sdpa-k512-venv
(clone of v072 + forced reinstall), fold OFF via documented
MLX_OMARCHY_FUSED_GEMV_NORM=0 — numerically the merge content, since the
norm-fold lineage is absent from origin/main.

## 1. Logits gates x3 (gold = ctl, 90 points: 10 prompts x [prefill + 8 greedy steps])

- leg 1: 0 flips, max top1 diff 0.125, PASS
- leg 2: 0 flips, max top1 diff 0.125, PASS
- leg 3: 0 flips, max top1 diff 0.125, PASS
- (attribution, fold ON: intermittent flip — 1/90 in one leg, 0/90 in another)

The 0.125 top-1 delta is a deterministic base difference between the 06add05a
and 2b8bdd7f trees, identical in every leg, argmax-invariant everywhere.

## 2. Digests (ordered_records_sha256)

- 3-pass cand (fold off) x3: bc519c03c4ef... EXACT (contract bc519c03) x3/3
- 10-pass ctl: dbf704971617... EXACT (contract dbf70497)
- 10-pass cand (fold off) x2: dbf704971617... EXACT x2/2
- 3-pass ctl (discriminator leg): bc519c03c4ef... — stack-equivalence confirmed
- 3-pass fold-ON control: cceba7527e06 — token moved; the fold, not the cap,
  is the only token-moving difference between the lineages

## 3. Ten interleaved paired 3-pass reps (ctl vs cand), decode_tok_rate medians

per-rep deltas (tok/s, cand - ctl): +0.17 +0.29 -0.01 -0.05 +0.26 +0.15
+0.02 +0.00 +0.00 +0.03; mean +0.086, sd 0.122, CI95 (t9=2.262)
[-0.001, +0.173] — NOT entirely positive: end-to-end decode is
route-insensitive (~77 tok/s both stacks; ~13 ms/token, host-enqueue-bound at
~280 dispatches/token), so the isolated 150 us/layer op saving does not
propagate to token rate. Prefill-side parity: ttft 80.5 vs 80.5 tok/s,
pure_prefill 819.4 vs 821.8 tok/s.

## 4. Family row (same-window pairing)

- sdpa[k=512]: ctl 505.65 -> cand 335.93 us/launch (-33.6%; original baseline
  504.11) — the composition now runs the contract decode regime
- sdpa[k=32]: 143.74 -> 147.83 (arm still engaged <=128, regime unchanged)

## Disposition

Gates PASS, digests PASS everywhere, family row moved -33.6%, paired CI not
entirely positive (workload property, no regression: point estimate +0.086).
Merge decision escalated to parent per the GO rule; cherry-pick
250d556b = 066baea3 onto origin/main 9fb8b675 staged and verified
(fast-forward, diff = cap hunk only, 7+/2- in primitives.cpp).

GPU discipline: five windows total (micro, sweep, gates x2, digests, perf,
discriminator); llm-inference restored after each (HEALTH_OK=1 + completion
probe every time).

## FINAL DISPOSITION (2026-09-24)

Main approved MERGE as a corrective. Pushed and verified:
9fb8b675..250d556b (fast-forward update), origin/main =
250d556bf35e78a53d751e2f2dddd78b919654c7 (cherry-pick of 066baea3, diff =
cap hunk only, 7+/2- in primitives.cpp). Recorded exactly as: corrective,
end-to-end neutral (CI [-0.001, +0.173], host-enqueue-bound). No end-to-end
speedup is claimed anywhere. Service active, health ok, live completion
probe returned content: ok.
