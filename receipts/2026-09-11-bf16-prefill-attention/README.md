# BF16 prefill attention — window 2 complete, default flip REVERTED (2026-09-11)

Agent: `Bf16AlphaFixAndVerdict` (window 2 execution + verdict); interim
attribution/digest A/B by `Bf16PrefillAttention` (window 1, retained
below). Branch: `bf16-prefill-attn`.

## Decision

The default flip of `MLX_OMARCHY_SDPA_BF16_FAST` is **reverted** on this
branch (keep the alpha coopmat fix + its regression test; the same fix
lands independently on `bf16-prefill-attn`'s sibling branch
`bf16-alpha-fix`, commit `c6c44674`, cut from origin/main `ee8d26fb`).
The flip fails on four independent counts:

1. **Rule 2 (native-matching legs) breached.** BF16 1K context matched
   native `ff502900d2a179a5`; under the flip it produces
   `5fd2fe812bf2a4ce` (fork) and `0b81e255dc4f68d5` (stock), 3/3 reps
   consistent on each. Rule 2 holds at full strength for this leg.
2. **Rule 1 breached.** BF16 262 moves to third values -
   `9bb3388dbce0b4c8` (fork), `f2accdd262f9707d` (stock) - neither the
   pinned Linux values nor native. Rule 1 allows a diverged leg to move
   to native's digest only.
3. **Re-pinned stock values moved without an owner amendment.** The
   2026-09-10 amendment pins BF16 short/long per driver. The flip moves
   stock short `7fc0f968789b1882` -> `f26175202f3dabe9` and stock long
   `46108ad71157cb4d` -> `f2accdd262f9707d`. (Fork short is unchanged
   at `f26175202f3dabe9`.)
4. **The oracle never favoured the candidate - and could not yet.**
   Status below; per the decision rule the burden is on the flip.

Measured gains exist and are real (see the matrix); they are banked in
this receipt only.

## Paired prefill matrix (window 2, warmup + 3 reps x {fork, stock} x {base, cand})

Prefill tok/s, 3-rep medians, `bench_matrix` canonical legs, both
wheels' provenance verified in-run, fork driver =
mesa-honeykrisp-omarchy 26.3.0.devel.hk6f6afc8-1 (pinned in-window):

| leg | base (fork) | cand (fork) | Δ | base (stock) | cand (stock) | Δ |
|---|---|---|---|---|---|---|
| BF16 30/32 | 131.6 | 135.7 | +3.2% | 132.7 | 136.4 | +2.7% |
| BF16 262/128 | 447.1 | 467.9 | +4.6% | 228.4 | 234.6 | +2.7% |
| BF16 1053/32 | 455.5 | 556.3 | **+22.1%** | 224.0 | 238.6 | +6.5% |

Q4 legs are gate-invariant (f16 sdpa; the flag never engages) and all
their digests held on every cell. Base-wheel digests: **all six
canonical Q4 + all BF16 pins held on both drivers, 3/3 reps**
(`q4_base_held`, `bf16_pins_held_base` in `verdict.json`). The base
wheel is a fresh build from a5b8c4ab
(`mlx_omarchy-0.32.2.dev202609111647+a5b8c4a`, sha256 `0a588e63...`,
provenance `match`); the original `dist/` wheel was wiped by the cand
build (`scripts/build-wheel.sh` does `rm -rf dist`), and window 1's
gate A/B ran on the earlier build of the same commit - its digests
match, which is itself cross-build digest stability evidence.

Full cell data: `verdict.json` (`paired_prefill`,
`digest_gates`), `matrix/r{1,2,3}-{fork,stock}-{base,cand}/`.

## Exact digests the flip would have produced (receipt-only record)

| leg | driver | today (pin) | under flip | native | verdict |
|---|---|---|---|---|---|
| BF16 30/32 | fork | `f26175202f3dabe9` | `f26175202f3dabe9` | - | unchanged |
| BF16 30/32 | stock | `7fc0f968789b1882` | `f26175202f3dabe9` | - | re-pinned value moved |
| BF16 262/128 | fork | `8690dc83246b39f8` | `9bb3388dbce0b4c8` | `407b7624ed1b3b29` | third value |
| BF16 262/128 | stock | `46108ad71157cb4d` | `f2accdd262f9707d` | `407b7624ed1b3b29` | third value |
| BF16 1053/32 | fork | `ff502900d2a179a5` | `5fd2fe812bf2a4ce` | `ff502900d2a179a5` | **leaves native** |
| BF16 1053/32 | stock | `ff502900d2a179a5` | `0b81e255dc4f68d5` | `ff502900d2a179a5` | **leaves native** |

## Oracle status (honest)

The oracle is the only instrument that could have rescued a rule-2
leg, and it never produced its decision numbers. Three separate defects
found and fixed while executing it:

1. `--seed`/`--out` argparse bugs (window 1's rc=1; fixed pre-window-2).
2. `model.config` -> `model.args` (mlx_lm 0.31.3; commit `47294881`).
3. `sdpa64` passed 2-D head slices into a 3-D einsum helper
   (`863a6869` + `1dd7031d`).

Window 2b (rerun on both wheels, `m1-logs/oracle-{base,cand}.{ndjson,log}`):
parts 1 (token streams both compositions) and 3-GPU (synthetic attention
outputs both gates, inputs saved to `attn-inputs-m{262,1053}.npy`)
complete; part 2 (numpy f64 truth) then produced a **systematically
broken truth stream** - `stream_match` 0/147 positions for BOTH
compositions (`both_wrong: 147`), which indicts the numpy forward
(RoPE/lm_head convention class of bug), not the GPU sides. Part 3-CPU
(the per-route attention ULP comparison against f64 truth - the actual
"equal or closer to truth" measurement on the affected operations) has
therefore never completed. The candidate-wheel run
(`--default-is-fast`, so its f32comp rows really are the f32
composition) is scripted and ready; the truth-forward fix is the one
remaining blocker. Until that lands, the oracle cannot favour the flip,
and the digest violations above disqualify it regardless.

## Component attribution (window 2, fixed probe, 9 reps, medians)

Wall-clock, `attribution_components.py` (probe scalar-multiply fix from
window 1's partial run), base and cand wheels, `--default-is-fast` on
the cand side so "gate off" is genuinely the f32 composition there:

| component | base wheel | cand wheel |
|---|---|---|
| whole 30/32 off | 198054 us (151.5 tok/s) | 198834 us (150.9) |
| whole 262/128 off | 378036 us (388.9) | 380287 us (386.6) |
| whole 1053/32 off | 2750714 us (382.8) | 2752585 us (382.5) |
| whole 262/128 on | 367667 us (399.8) | 366613 us (401.0) |
| whole 1053/32 on | 2534199 us (415.5) | 2534520 us (415.5) |
| sdpa m262 off/on | 3016 / 2843 us | 2460 / 1760 us |
| sdpa m1053 off/on | 18927 / 24620 us | 18971 / 24660 us |
| lm_head m=262 | 123871 us, 0.576 TFLOP/s | 124046 us, 0.575 |
| lm_head m=1053 | 458658 us, 0.625 TFLOP/s | 457072 us, 0.627 |

Notes: (a) whole-prefill off-rows agree between wheels within 0.1% -
the alpha fix leaves the f32 composition untouched, as designed.
(b) `lm_head` (286 GFLOP at m=1053) is measured for the first time: at
0.58-0.63 TFLOP/s it is ~33% of the 262 leg and ~17% of the 1K leg -
larger than the entire attention block and the single biggest
unexplored prefill lever (it rides MatmulBF16Coopmat at alpha==1).
(c) the synthetic `sdpa_whole` m=1053 "on" regression (18.9 -> 24.6 ms)
does not reproduce in the whole-model path (whole 1053 on improves
+8.5%); probe shapes (dense 14 heads, no GQA regroup, no cache) diverge
from the model path and the probe number is recorded, not trusted.

## Abort-gate finding: 8 f16 sdpa throws on the M1 (pre-existing)

The cand tree's `omarchy_fast_ops_tests` on the M1 throws in 8 sdpa
cases (`m1-logs/attn-fork-fastops-cand.log`):
`[omarchy] ScaledDotProductAttention dtype is not implemented
(dtype=float16, rank-5 mask/GQA shapes)`. The same suite passes 34/34
on llvmpipe. The f16 path is byte-identical across main, the alpha fix,
and this candidate (the env flip only reaches bf16), and the suite has
never been in the M1 battery (prior receipts ran fam + runtime only) -
so this is a pre-existing main defect surfaced by extending coverage,
not a regression of either strand. The alpha-tree control run is
queued in the S1 window (`/tmp/fast-alpha` full-suite log) to prove it
empirically. Owner decision required; open, not fixed here.

(The window script's phase-A gate missed this on first pass: it grepped
`FAILED`, and doctest prints `FAILURE!`. The gate now also fails on
rc!=0 - fixed in this receipt's `m1_window2.sh` for future windows.)

## What a rule-2 amendment would have to claim

If the owner wants to revisit the flip, an amendment in the
2026-09-10 style would have to claim, with evidence:

1. That the BF16 1K leg's native match carries no parity content, i.e.
   macOS native and this fork agree on `ff502900d2a179a5` by
   coincidence of rounding, not by shared arithmetic - and that the
   f64 oracle (once its truth-forward is fixed) shows the bf16
   score-storage bits equal or closer to truth than the f32
   composition on m=262 and m=1053 (attn_ulp rows).
2. That the 262-leg third values are acceptable as NEW per-driver
   pins (this is a stronger claim than the 09-10 amendment, which
   only moved already-diverged legs to f64-nearest arithmetic).
3. That the stock-driver digest moves (short + long + 1K) are
   re-pinned deliberately.
4. Why the +22.1%/+6.5% 1K and +2.7..4.6% short/long prefill gains
   justify four pin movements when the un-flipped
   bit-preserving-restructure path (bf16 loads, exact in-shader
   upcast, alpha at A-stage, f32 score/prob storage; bounded ~2-3% of
   the leg by the committed-receipt arithmetic) and the newly-measured
   lm_head lever (~17-33% of leg at 0.6 TFLOP/s) remain unexplored.

## Provenance

- cand wheel: `mlx_omarchy-0.32.2.dev202609111606+b1f9ba6` (overlay =
  6fa75adf code), sha256 `6c1a2f5c...`, `.venv-attn-cand`, provenance
  `match`.
- base wheel: `mlx_omarchy-0.32.2.dev202609111647+a5b8c4a`, sha256
  `0a588e63...`, `.venv-attn-base`, provenance `match`.
- suite binaries `/tmp/{fam,fast,rt}-attn`: rebuilt post-brace-fix,
  sha256 in `m1-logs/suite-binaries-attn.sha256`.
- Driver pin checked in-window on every lock acquisition (window2.log,
  window2b.log headers).
- GPU windows: window 2 11:58-12:23, window 2b (oracle) 12:29+; single
  top-level `/tmp/m1-gpu.lock` flock, 7200s wait cap, no nesting, no
  foreign processes touched.

## Remaining work

1. Fix the oracle's numpy f64 truth forward (RoPE convention /
   tie-embeddings are the suspects); rerun both wheels; fill the
   attn_ulp decision table (`make_verdict.py` already renders it).
2. Execute the S1 window for `bf16-alpha-fix`
   (`/tmp/m1_prep_s1.sh` then `/tmp/m1_window_s1.sh` on jwm1): alpha
   suites on the M1, trap fail-proof, f64 probe, and the fix wheel's
   digest gates (all 12 canonical cells must hold).
3. The 8 f16 sdpa throws - root-cause and fix or file for the owner.

---

# Appendix: window 1 interim receipt (Bf16PrefillAttention, unchanged)

The interim verdict text from before window 2 is retained in git
history (commit `6ed02096`); its attribution-probe rows crashed on the
scalar-multiply bug and its digest A/B single runs are superseded by
the 3-rep paired matrix above. The gate A/B table it reported:

| leg (fork, single run, base wheel) | gate off | gate on | Δ prefill | digest |
|---|---|---|---|---|
| Q4 short / long / 1K | pins hold | pins hold | ~0 | unchanged |
| BF16 short (30/32) | `f26175202f3dabe9`, 132.2 tok/s | same digest | +3.2% | unchanged |
| BF16 long (262/128) | `8690dc83246b39f8`, 448.6 | `9bb3388dbce0b4c8` | +4.7% | third value |
| BF16 1K (1053/32) | `ff502900d2a179a5`, 449.2 | `5fd2fe812bf2a4ce` | +11.9% | leaves native |
