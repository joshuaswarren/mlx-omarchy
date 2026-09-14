# 2026-09-14 RMSNorm Add+Norm fold 5-repeat A/B on jw16 — NO-LAND

## Verdict

**Named no-land.** The ctx1053 median decode tok/s did not rise (139.87 vs
141.08, −0.86%), so the landing rule fails. Stronger: the fold provably never
executes in the Qwen2.5 decode path — per-token `vk_compute_dispatches` is
identical with `MLX_OMARCHY_FUSED_RMSNORM` on and off, even with the GEMV
epilogue claimer disabled — so there is no kernel effect to land. Nothing was
merged; `rmsnorm-epilogue` (a3e9f486) remains an unmerged branch.

## Setup

- Host: `jw16mbp1-linux` (aarch64, 7.1.6-1-1-ARCH). GPU serialized under one
  `flock` hold on `/tmp/m1-gpu.lock` (inode 12 before and after; never
  unlinked). Nested `flock -n` correctly refused.
- Model: `mlx-community/Qwen2.5-0.5B-Instruct-4bit`
  (snapshot `a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3`).
- Protocol: `ab_decode.py` interleaved 5 rounds x 2 legs (short 30/32,
  ctx1024 1053/32) x 2 arms, arm order alternated per round, every leg a
  fresh `bench_decode.py` process (`MLX_DISABLE_COMPILE=1`,
  `HF_HUB_OFFLINE=1`, `--tokens 32 --temp 0.0 --seed 0 --warmup-tokens 4`),
  per-arm wheel provenance asserted (`verified=match` on all runs).
- base arm: wheel `mlx_omarchy-0.32.2.dev202609141639+b79a4b68`
  sha256 `beeeaee99834a24db788ffa8ea809689667343327a271bafd8576a2779c719b7`
  (branch point commit b79a4b68), venv `venv-base`.
- cand arm: wheel `mlx_omarchy-0.32.2.dev202609142212+a3e9f486`
  sha256 `db91990f4ae0ecf4b507aa186afdaf6fc2d0b8265a91b2b2dc8d37c7b0ab0469`
  (`rmsnorm-epilogue` tip a3e9f486 = 4853e18e fold + a3e9f486 typeid fix,
  worktree `/var/tmp/rmsnorm-epilogue`, clean), venv `venv-rms`.
- Fold gates default ON (`getenv==nullptr || env_flag`); no gate env was set
  in the A/B arms, so the fold was enabled in the cand arm.

## 5-repeat decode tok/s medians (all 20 legs digest-pinned)

| leg | base | rmsnorm fold | delta |
|---|---|---|---|
| short (30/32) | 167.7603 | 170.3265 | +1.53% |
| ctx1024 (1053/32) | **141.0811** | **139.8689** | **−0.86%** |

All runs (base first except odd rounds):
- short base: 168.964, 168.9137, 166.1569, 165.981, 167.7603
- short rms: 166.2723, 170.613, 169.121, 170.6886, 170.3265
- ctx1024 base: 141.0811, 144.1121, 135.046, 139.5959, 145.4184
- ctx1024 rms: 126.9572, 143.4962, 139.8689, 142.1127, 136.8747

Pins held on every run: short `7fd25a869ff21678` (10/10), ctx1024
`7da83f06ec9f001d` (10/10). prompt_tokens 30/1053 asserted each leg.

## Why the medians are noise, not the fold

Per-token counters (`dispatch_count.py`, median over decode tokens, same
wheel + model):

| config | vk_compute_dispatches | gpu_primitive_dispatches | vk_submissions |
|---|---|---|---|
| base wheel, default gates | 249 | 858 | 2 |
| rms wheel, default gates | 249 | 858 | 2 |
| rms wheel, `FUSED_GEMV=0` | 465 | 858 | 3 |
| rms wheel, `FUSED_GEMV=0 FUSED_RMSNORM=0` | 465 | 858 | 3 |

`FUSED_GEMV=0` moves the counter (249→465), so the counter is sensitive to
fusion state. Toggling `MLX_OMARCHY_FUSED_RMSNORM` moves nothing in either
gate configuration: the Add+RMSNorm pair is planned zero times per token.
The commit message targets "two of the 249 per-token decode dispatches per
layer boundary" (24 layers → ~225 expected if firing); it stays at 249.
The fold's typeid fix (a3e9f486, `typeid(RMSNorm)` →
`typeid(mlx::core::fast::RMSNorm)`) made the node type match, but the pair is
still never claimed in real decode — plausibly because the residual Add is
already claimed by the GEMV epilogue in the default config, and with
`FUSED_GEMV=0` it is still not claimed (mechanism inference; the counter
result is measured, not inferred).

With an inert kernel, the two wheels differ only by the inert
`ane-compiler.lock` pin commit (7f8786b0); the ±1.5% tok/s spread is
build-to-build noise, consistent with the earlier one-leg numbers
(+1.8% short, −0.7% ctx1053).

## Decision

Land rule: ctx1053 median must rise → 141.0811 → 139.8689 is a decline →
**no-land**. Additionally a fold that never executes must not land as a
"kernel" at all. `63c1d3cf` was not merged (hash is unreachable in every
reachable repo on both hosts — nothing to merge; noted as instructed).
No repo state was changed: `mlx-omarchy` `main` untouched (3db3cb9a), branch
`rmsnorm-epilogue` left at a3e9f486, RoPE fold lane untouched.

## Artifacts

- jw16: `/var/tmp/DecodeEpilogueFold/jw16-out-rmsnorm/` — `ab.txt`, `ab.json`
  (medians + per-run + provenance), `dispatch-{base,rms}.json`,
  `dispatch-probe-{gemv-off,all-off}.json`, `lock.txt`, `started.txt`,
  `finished.txt`.
- Runner: `/var/tmp/DecodeEpilogueFold/run-rmsnorm-ab.sh`,
  `probe-gates.sh`; harness `ab_decode.py`, `dispatch_count.py` (unchanged).
- Wheels: paths + sha256 above.
