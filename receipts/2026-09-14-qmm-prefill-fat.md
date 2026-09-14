# QmmPrefillCoopmatF16 fat shapes: a 16-column twin, measured and rejected

Date: 2026-09-14

## Verdict

**A 16-column tile twin of `QmmPrefillCoopmatF16` was built, measured on
both hosts, and is not landed.** It is digest-preserving everywhere it
fires, but it is not a fat-shape win: at m=1053 the fat cells are
unchanged (the gate keeps them on the shipped 32-column tile), and at
every prompt length where the twin does fire with meaningful wall time,
it is reproducibly slower end-to-end. The occupancy-16-row change
(45b71465) already took the starved-grid win; this session closes the
next hypothesis and leaves the fat-shape gap (rel 0.87-0.95,
`receipts/2026-09-14-jw16-max-gpu-attribution/`) attributed to the
driver's serialized staging, where the 2026-09-14 arms receipt already
showed shader-source changes cannot reach it.

The candidate stays on branch `agent/qmm-prefill-fat-shapes`
(02b0a7f4) with the gate constants set to the measured-no-win state
described below; `main` is untouched.

## What was built

`shaders/qmm_coopmat.comp` gained a `TILE_COLS` build parameter (16 in
addition to the shipped 32): 16-column output tiles per 64-lane
workgroup, halving the staged weight tile's column width and doubling
the dispatch grid. Each output's ascending-k f32 chain and its 8-wide
`coopMatMulAdd` updates are untouched, so generated ids cannot move.
The twin ships as `QmmPrefillCoopmatN16F16` (appended after
`QmmPrefillCoopmatM16F16`, keeping profile kernel ids stable) and is
selected by `coopmat_tile_cols()` in `primitives.cpp` behind the same
workgroups-per-core shape the 16-row pick uses.

Selection gate (final form, all conjuncts from measurements below):
known-wide part only (`cores >= 24`; the 8-core M1 loses everywhere the
twin fires), `matrix_m <= 512` (halved x-tile staging reuse loses on
long-prompt working sets), and under 64 workgroups per core.
`MLX_OMARCHY_QMM_COOPMAT_N16_WG_PER_CORE` overrides or disables (0) the
per-core ceiling.

## Identity

- Candidate: `agent/qmm-prefill-fat-shapes` at
  `02b0a7f418f5e8239ade60960e394c50095d406c` (base `4188ba95` =
  origin/main).
- Wheels, one per host, built by `/var/tmp/qmm-fat-*/build-fat-wheel.sh`
  from that commit, `-DMLX_OMARCHY_GPU_PROFILING=OFF`:
  - jw16 `mlx_omarchy-0.32.2.dev202609142107+fat.02b0a7f4…whl`
    SHA-256 `56f0a011645d4f1194a18dbb6473f0ae56f1516f69909128cf8d19cd3288034a`
  - jwm1 `mlx_omarchy-0.32.2.dev202609142107+fat.02b0a7f4…whl`
    SHA-256 `6623fae6895d005ea3d62cc5c9e867a3948cc486f8423b34e23be79382fb2353`
- Same-wheel A/B: `base` arm = `MLX_OMARCHY_QMM_COOPMAT_N16_WG_PER_CORE=0`
  (shipped pick exactly); `n16` arm = compiled-in gate. One binary, two
  environments, so no build delta can leak into the comparison.
- Hosts: `jw16mbp1-linux` (M1 Max T6001, 32 GPU cores per the census),
  `jwm1-linux` (M1 T8103, 8 cores), both on
  `mesa-honeykrisp-omarchy 26.3.0.devel.hk6f6afc8-1`, kernel
  `7.1.6-1-1-ARCH`.
- Model: `Qwen2.5-0.5B-Instruct-4bit` (jwm1 local dir; jw16 HF snapshot
  `a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3`), affine 4-bit group 64.
- Prompts: `scripts/bench_matrix.json` `short` (30 chat-template
  tokens) and `ctx1024` (1053); a mid-length leg built from the same
  numbered template cut to 5 entries (452 chat-template tokens).
- Model: `anthropic/claude-opus-5`; routing fallback: false.

## Protocol

- Kernel-isolated: `qmm_coop_bench_probe.py` at the eight real prefill
  cells, 4 warmups + 30 timed reps, median. Sequential A/B first, then
  three interleaved rounds inside one `/tmp/m1-gpu.lock` hold
  (`ab-interleaved.sh`) after the sequential runs proved noisy.
- End-to-end: `scripts/bench_decode.py --tokens 32 --temp 0 --seed 0
  --warmup-tokens 4 --wheel <wheel>`, fresh subprocess per leg,
  `MLX_DISABLE_COMPILE=1 HF_HUB_OFFLINE=1`, under
  `flock -w 60 /tmp/m1-gpu.lock`; three interleaved rounds at ctx262
  and one round per arm at short/ctx1024.
- Locks: never stolen, never unlinked. No ANE command, no reboot, no
  driver change, `dist/`/`.work/` of the checkouts untouched (private
  `/var/tmp/qmm-fat-*` roots and venvs).

## Results

### Digests (the hard gate)

Pinned digests match on every leg, both hosts, both arms:

- 1053-token: `7da83f06ec9f001d` (12 legs, all rounds)
- probe f16 output digests: all eight cells equal base vs n16 on every
  round on both hosts (`digest equality: True`; raw rows in `jw16/`
  and `jwm1/`)

### jw16 (M1 Max) - the candidate is not a fat-shape win

Sequential session 1 suggested mid-m wins (+3.5/+4.0/+24.5% at m=262),
but they did not reproduce: session 2 flipped signs at three cells, and
the interleaved probe rounds are dominated by DVFS noise - an arm that
*never fires* (m=1053 excluded cells) still shows up to +-13%
probe-level, so probe deltas below that are not evidence.

End-to-end is consistent and is what decides:

| leg | base tok/s | n16 tok/s | n16 delta | rounds |
| --- | ---: | ---: | ---: | ---: |
| ctx1024 (1053 tok, twin never fires) | 3886.6 | 3878.9 | -0.2% | 1 |
| ctx262 (452 tok, twin fires) | 3302.0 / 3264.5 / 3325.6 | 2760.4 / 2779.9 / 2752.2 | **-16.2 / -14.8 / -17.3%** | 3 |
| short (30 tok, twin fires) | 444.6 | 458.1 | +3.0% (0.07 ms absolute) | 1 |

The -16% at 452 tokens is three-for-three reproducible. The only win is
a +0.07 ms absolute effect on a 2 ms prefill.

### jwm1 (M1) - no-regression PASS

The gate never fires on the 8-core part, and the legs confirm identity
within noise:

| leg | base tok/s | n16 tok/s |
| --- | ---: | ---: |
| ctx1024 | 1111.96 | 1110.66 |
| ctx262 | 1045.72 | 1051.97 |
| short | 378.86 | 380.07 |

All six legs `7da83f06ec9f001d`; ctx1024 1111.96 against the pinned
1111.58 (`receipts/2026-09-14-jwm1-gpu-parity-refresh.md`).

## Why this is a rejection, not a near miss

The win region the gate was built to capture collapsed under
repetition: m=262 cell wins were single-session artifacts, and the one
reproducible effect inside the region is a large mid-length prefill
loss - the doubled grid costs more in x-tile restaging than it buys in
occupancy as soon as the prompt carries real length. On the fat cells
themselves (m=1053, n=896/4864) the twin is neutral-to-negative, so no
gate setting can make this candidate a fat-shape win. Combined with the
2026-09-14 arms receipt (six shader arms neutral-or-slower at the
dominant cell) and the 16-row twin taking the starved-grid win, the
remaining fat-shape headroom sits in the driver's serialized staging
waits, below what shader-source or dispatch-geometry changes can reach.

## Artifacts

`jw16/` and `jwm1/` hold the raw e2e logs, interleaved probe rows and
the e2e round log; `SHA256SUMS` covers all of them. On the hosts:
`/var/tmp/qmm-fat-jw16/` and `/var/tmp/qmm-fat-jwm1/` (build logs,
wheels, venvs, worktree at the measured commit).
