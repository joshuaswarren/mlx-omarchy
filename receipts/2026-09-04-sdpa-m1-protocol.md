# M1 protocol: SDPA f16-scores rework (consolidated, one run)

Owner: BenchQueueM1 on jwm1 (100.84.184.102). Requested by SdpaRework.

## Under test

Branch `sdpa-f16-scores` on origin (joshuaswarren/mlx-omarchy), stacked
on `ew-fused-elementwise` (`77c0b95` = origin/main + the 5
census/instrumentation commits - keep that base, the census
instruments live there).

Commits, in order (all marked PROVISIONAL until this leg returns):

- `641b96a` instruments (sdpa_equivalence.py, sdpa_counts.py)
- `f231704` fix 1: f16 scores, f32 accumulation, alpha scale, no
  output downcast
- `59de953` fix 2: stride-aware matmul operands (shader gaps, free
  batch strides, zero materializations)
- `7770d4c` fix 3: GQA regroup through broadcast views
- `3113543` receipt (findings; read before interpreting numbers)

Reference (revert side of the A/B): the tree WITHOUT the three fixes =
`641b96a` (instruments only, code identical to `77c0b95`).

Build each side with `scripts/build-wheel.sh --diagnostics`
(MLX_OMARCHY_GPU_PROFILING=ON is required for the counts). On jwm1
never set MLX_OMARCHY_ALLOW_NON_APPLE. Provenance is mandatory per the
new rule: run `python3 scripts/mlx_provenance.py` beside every number
and paste the verified=match line + wheel version into the report. A
number without the line is not evidence.

## Runs (per side: reference and final; plus fix-1-only if cheap)

1. Census decode: pinned model snapshot
   `a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3`
   (mlx-community/Qwen2.5-0.5B-Instruct-4bit), mlx-lm==0.31.3,
   prompt "What is the capital of France?" (chat template), 32 max
   tokens, temp 0, seed 0:
   `MLX_OMARCHY_GPU_PROFILE=<p.jsonl> python scripts/profile_generate.py
   --model <snapshot> --prompt "What is the capital of France?"
   --max-tokens 32 --temp 0 --seed 0 --markers <m.jsonl>`
   then `python scripts/sdpa_counts.py <p.jsonl> --compute-h
   .work/mlx/mlx/backend/omarchy/compute.h --markers <m.jsonl>`
   Report: dispatches/token, CopyGeneralF16/token (and /layer),
   casts/token, MatmulF16/F32, SoftmaxF16/F32, median inter-token ms.
2. Decode tok/s (same run's median_inter_token_ms) and prefill tok/s:
   rerun with --raw-prompt and a long prompt (>=256 tokens) once per
   side for a prefill-dominated measurement.
3. bf16 leg (dtype coverage): same census with
   mlx-community/Qwen2.5-0.5B-Instruct-bf16 if present on the box -
   expected NO dispatch change vs reference (bf16 takes the unchanged
   f32 composition); confirms the dtype gate.
4. Greedy token identity: the France run's output text must be
   identical on reference and final ("The capital of France is
   Paris." on llvmpipe; report whatever M1 produces, both sides).
5. Correctness on device: `MLX_OMARCHY_ALLOW_NON_APPLE` unset,
   `python scripts/sdpa_equivalence.py` - must print ALL PASS.
6. Watch for score overflow: any NaN/garbage in greedy output on the
   4-bit model is an automatic reject-and-report (f16 scores cap at
   65504).

## llvmpipe reference numbers (dev box, verified=match provenance in the
receipt)

- dispatches/token: 1,953 -> 1,833
- casts/token: 240 -> 144
- CopyGeneralF16/token: 288 -> 288 (unchanged; attribution finding in
  the receipt - the census copies were misattributed, do not expect a
  copy drop)
- greedy: token-identical; equivalence 17/17

## Keep rule (proposed)

- KEEP all three fixes if M1 shows: greedy token-identical (or
  coherent and explained), equivalence ALL PASS, dispatches/token not
  higher than reference, and decode tok/s not slower than reference
  beyond run noise (<3%).
- KEEP fix 1 alone, REVERT fixes 2+3, if M1 shows the stride/GQA
  changes costing tok/s or correctness while fix 1 alone carries the
  win (fix 1 is the whole measured llvmpipe delta: -120
  dispatches/token, all casts).
- REVERT everything if M1 shows any NaN/garbage or a tok/s regression
  >3% on fix 1 alone.
- Report both sides' numbers either way; a revert ships with the
  numbers and the reasoning, not a silent drop.
