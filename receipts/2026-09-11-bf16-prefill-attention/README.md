# BF16 prefill attention — component attribution and f32-composition re-qualification (2026-09-11)

Agent: `Bf16PrefillAttention`. Branch: `bf16-prefill-attn` (a5b8c4ab + staged
candidate). **Status: attribution + digest A/B complete; decision data in;
paired matrix and oracle rerun queued behind the shared GPU (window 2).**

## Interim verdict (pending window-2 confirmation)

The unconditional default-flip of `MLX_OMARCHY_SDPA_BF16_FAST` is **not
landable under the current parity contract**, despite a real prefill gain:

| leg (fork, single run, base wheel) | gate off | gate on | Δ prefill | digest |
|---|---|---|---|---|
| Q4 short / long / 1K | pins hold | pins hold | ~0 | unchanged (4-bit models run f16 sdpa; gate never engages) |
| BF16 short (30/32) | `f26175202f3dabe9`, 132.2 tok/s | same digest | +3.2% | unchanged |
| BF16 long (262/128) | `8690dc83246b39f8`, 448.6 | `9bb3388dbce0b4c8` | +4.7% | **third value** (neither Linux nor native `407b7624ed1b3b29`) |
| BF16 1K (1053/32) | `ff502900d2a179a5`, 449.2 | `5fd2fe812bf2a4ce` | +11.9% | **leaves native** (rule-2 leg) |

- BF16 1K is native-matching today; `docs/parity-id-policy.md` rule 2 holds
  at full strength for it. The 2026-09-10 amendment precedent covers only
  legs already diverged from native.
- BF16 262 moved to a third value; rule 1's carve-out allows only a move to
  native's digest. The f64 oracle (rerun queued) is expected to show the
  bf16-score-storage path farther from truth than the f32 composition,
  which would also fail the amendment's "equal or closer" standard.

## Component attribution, 1053-token leg (base wheel, gate off, fork driver)

From `m1-logs/attribution-base.ndjson` (wall-clock medians; probe crashed
after 9 rows on a scalar-multiply bug — fixed, rerun queued in window 2):

- whole prefill: 2,760,173 µs (381.5 tok/s under probe conditions;
  bench_matrix measured 449.2 tok/s on the same wheel/driver)
- whole prefill gate-on: 2,532,682 µs (415.8 tok/s) = **+9.0%** from the
  attention composition alone
- `sdpa_whole` at m=262 shapes: 2,528.8 → 2,279.0 µs (gate on) = 1.11x

Probe-condition numbers are internally consistent (same conditions both
gates) but run ~15% below bench_matrix conditions; use the matrix numbers
for fractions-of-native and the probe for component shares.

Committed-receipt arithmetic for the same leg (52.8 ms/layer projection
chain, receipts/2026-09-11-bf16-prefill) puts projections+MLP at ~54% of
the leg and attention f32 compute+traffic at ~15-20%; the gate-on delta
(+11.9% at 1053 on the matrix run) is consistent with attention being the
second dominator behind the GEMM rate. `lm_head` (286 GFLOP at m=1053,
never previously measured) is quantified by the window-2 rerun.

## Staged source changes (branch `bf16-prefill-attn`, commit 6fa75adf)

1. `shaders/matmul_coopmat_bf16.comp`: alpha applied to the f32
   accumulator at the drain (same point as the 16x16 tile; alpha==1
   multiplies exactly, so existing projection traffic stays bit-identical).
   Before this fix the kernel declared alpha and never used it, while the
   dispatch gate did not check alpha — a silent wrong-arithmetic trap for
   any alpha!=1 bf16 coopmat dispatch.
2. `primitives.cpp` `dispatch_matmul`: coopmat alpha==1 gate relaxed for
   `MatmulBF16` only (`matmul_coopmat.comp` still never reads alpha).
3. `primitives.cpp` sdpa: `MLX_OMARCHY_SDPA_BF16_FAST` default flipped to
   on (env `=0` opts out) — **staged, not landed** (see verdict).
4. `overlay/tests/omarchy/test_fast_ops.cpp`: new regression
   "sdpa bf16 fast scores scale through MatmulBF16Coopmat" (coopmat-gated
   shape, alpha=0.25; a missing scale fails by orders of magnitude); GQA
   mask test relabeled for the new default (fast default + `=0` opt-out).
5. Docs: `docs/compatibility.md` (bf16 coopmat alpha semantics + sdpa
   default), `docs/install-omarchy.md` (flag status).

## Remaining (window 2, scripts in this directory)

1. `m1_build_cand.sh` — RUNNING on jwm1 (taskset 0-1, niced, per peer
   agreement): cand wheel + `.venv-attn-cand` + suite binaries
   (fam/fast_ops/runtime).
2. `m1_window2.sh` (flock, capped): suite abort gates, paired matrix
   warmup + 3 reps x {fork,stock} x {base,cand}, base attribution rerun
   (probe fixed), oracle rerun (`--seed` fix), cand attribution.
3. `make_verdict.py` assembles `verdict.json`; decision rules:
   - If the oracle shows bf16-fast closer to f64 truth on 262 AND the 1K
     move gets owner-level rule-2 treatment: reconsider default flip.
   - Otherwise: revert the default-flip commit, keep the shader alpha fix
     + regression test (correctness fix, zero bit change for alpha==1
     traffic), land the receipt-only negative with the attribution table.
   - Bit-preserving restructure (bf16 loads, exact in-shader upcast,
     alpha at A-stage, f32 score/prob storage) is measured-bounded at
     ~2-3% of the leg and is documented as a follow-up, not built.

## Files

- `attribution_components.py` — component-isolated wall-clock probe
- `oracle_f64.py` — f64 oracle: token streams, numpy-f64 full-model truth,
  attention-block ULP study
- `m1_window1.sh` / `m1_window2.sh` / `m1_build_cand.sh` — window runners
- `m1-logs/` — attribution NDJSON (partial, pre-fix), oracle NDJSON
  (part 1), build logs
- `matrix/attn-gate-{off,on}-fork/` — digest A/B single runs
- `make_verdict.py` — verdict assembly
