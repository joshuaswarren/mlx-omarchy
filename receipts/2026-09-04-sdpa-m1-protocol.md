# M1 protocol: SDPA f16-scores rework (consolidated, one run)

Owner: BenchQueueM1 on jwm1 (100.84.184.102). Requested by SdpaRework.
v2, rebased: the BEFORE is now integrated main WITH the fused fast::RoPE
(commits bbda339..5ac59e8). The earlier 1,953-dispatch baseline was
pre-RoPE and is retired; both legs below were re-measured on the
integrated tree on llvmpipe.

## Legs (both on origin; build each with scripts/build-wheel.sh
--diagnostics; on jwm1 never set MLX_OMARCHY_ALLOW_NON_APPLE)

- BEFORE (reference): branch `sdpa-ref-integrated` (tip `1294e69`) =
  integrated main + census instruments, three SDPA fixes reverted.
  llvmpipe reference: 753 dispatches/token, casts 96, CopyGeneralF16
  96 (4/layer), MatmulF32 48, SoftmaxF32 24, median inter-token
  959.10 ms, greedy "The capital of France is Paris."
- AFTER (under test): branch `sdpa-f16-scores` (tip `9bff78e`).
  llvmpipe: 633 dispatches/token, casts 0, CopyGeneralF16 96 (4/layer,
  unchanged), MatmulF16 48, SoftmaxF16 24, median inter-token 959.45
  ms, greedy token-identical.

Per-fix commits on `sdpa-f16-scores` (PROVISIONAL until this leg):

- `05d8ce2` instruments (sdpa_equivalence.py, sdpa_counts.py)
- `9e9c7e7` fix 1: f16 scores, f32 accumulation, alpha scale, no
  output downcast
- `cf96f12` fix 2: stride-aware matmul operands
- `5133e54` fix 3: GQA regroup through broadcast views
- `9bff78e` fix(trace): completes the rebased census counter set
  (build breaker on the earlier push; include it)
- findings: `receipts/2026-09-04-sdpa-f16-scores-rework.md` - read
  before interpreting copy numbers; the census copy attribution was
  wrong and the 4-6-deleted-copies prediction is refuted (recorded
  copies 4/layer unchanged by these fixes; the RoPE leg, not this
  rework, removed the other 8/layer).

Model: mlx-community/Qwen2.5-0.5B-Instruct-4bit snapshot
`a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3`, mlx-lm==0.31.3, prompt
"What is the capital of France?" (chat template), 32 max tokens, temp
0, seed 0.

Provenance line mandatory on every number (fleet rule 2026-09-04): run
`python scripts/mlx_provenance.py` beside each measurement and paste
verified=match + wheel version into the report.

## Runs (per leg: BEFORE and AFTER)

1. Census decode (command in the receipt): dispatches/token,
   CopyGeneralF16/token and /layer, casts/token, MatmulF16/F32,
   SoftmaxF16/F32, median inter-token ms.
2. Decode tok/s (same run) and one prefill-dominated run per leg
   (--raw-prompt, >=256-token prompt).
3. bf16 leg on the AFTER side only (mlx-community
   Qwen2.5-0.5B-Instruct-bf16): must route to the unchanged f32
   composition with no dispatch regression.
4. Greedy token identity: BEFORE and AFTER outputs must match token
   for token. Any NaN/garbage is an automatic reject-and-report (f16
   scores cap at 65504).
5. On-device correctness: MLX_OMARCHY_ALLOW_NON_APPLE unset,
   `python scripts/sdpa_equivalence.py` must print ALL PASS.
6. Copy watch: report CopyGeneralF16/token both sides. If AFTER shows
   fewer copies than BEFORE on the M1, say so explicitly - llvmpipe
   could not see that deletion (dispatch-neutral there).

## Observable for fixes 2+3 (per Main's rule: no metric, no ship)

Fix 1 owns the entire llvmpipe delta: -120 dispatches/token (casts 96
+ scale multiply 24), -16% against the 753 integrated baseline. Fixes
2+3 were dispatch-neutral on llvmpipe and delete the matmul's internal
operand materialization pass, which the census instrument cannot see.
The ONLY observable that can justify them on hardware: CopyGeneralF16
(or any dispatch class) lower on AFTER than BEFORE, or a decode
tok/s gain attributable beyond fix 1. If M1 shows AFTER copies ==
BEFORE copies (expected 4/layer both sides) and no tok/s gain, fixes
2+3 ship nothing measurable and get DROPPED - keep fix 1 alone.

## Keep rule

- KEEP fix 1 (all three if the 2+3 observable fires): greedy
  token-identical, equivalence ALL PASS, dispatches/token <= BEFORE,
  tok/s within 3% of BEFORE or better.
- DROP fixes 2+3 if the AFTER copy/token equals BEFORE and no tok/s
  gain appears (expected outcome per llvmpipe).
- REVERT everything on NaN/garbage or a tok/s regression >3%.
- Both sides' numbers ship with the decision either way.
