# 2026-09-15 SwiGLU-only GEMV store epilogue confirmed on jw16 — pins 20/20, 225→201 dispatches, ctx1053 +1.8% median

Task: confirm origin/main `a21b3c81` (the SwiGLU store epilogue landed on
jwm1, receipts/2026-09-15-swiglu-epilogue.md) on a second host, jw16,
before stacking the RMSNorm GEMV prologue retest on top of it.

## Verdict

**Confirmed. Every gate holds on jw16, matching jwm1.**

1. Dispatches drop — 225 → 201 `vk_compute_dispatches` per decode token
   (−24 = exactly the per-layer standalone swiglu), `gpu_primitive_
   dispatches` stays 858; `MLX_OMARCHY_FUSED_GEMV_SWIGLU=0` on the same
   cand wheel restores 225, so the drop is the fold.
2. Both pins hold — short `7fd25a869ff21678` and ctx1024
   `7da83f06ec9f001d` in every arm and every round of the 5-round
   interleaved A/B (20/20 leg digests), plus the dedicated ctx1024
   digest arm on base, cand, and cand-with-fold-off (first ids
   `13060,498,369`, identical on all three arms).
3. ctx1053 median does not regress — it rises:

| leg | base median | cand median | Δ |
|---|---|---|---|
| ctx1024 (1053/32) | 141.39 tok/s | 143.91 tok/s | **+1.8%** |
| short (30/32) | 179.41 tok/s | 190.26 tok/s | +6.1% |

Per-run ctx1024 base: 140.87, 143.68, 144.09, 135.63, 141.39; cand:
132.35, 143.91, 135.13, 147.36, 150.81. The cand spread is wider than
jwm1's but every gate is on the median, and the short leg's five cand
runs (190.17–190.71) beat every base run (173.32–179.69) outright.

## Identity

- Base wheel `mlx_omarchy-0.32.2.dev202609152131+1deb70f1` sha256
  `f7271ba507b9696aba8227ff301fa664d39eda01b1a9cc88f13f88ff9bc1701a`
  (origin/main before the fold); cand wheel
  `mlx_omarchy-0.32.2.dev202609152134+a21b3c81` sha256
  `1540b11d2828b810c70ea7fb45d1faf5fa57e82c8babb4c420b63593c120c655`
  (origin/main). Worktrees `/var/tmp/SwigluRmsJw16/{base,cand}`
  (detached at the two commits), built with the jw16 canonical flags
  (receipts/2026-09-14-jw16-max-gpu-attribution: openblas include dirs
  + local FetchContent seed), stamp check `+1deb70f1` / `+a21b3c81`.
- Host: jw16mbp1-linux (M1, Honeykrisp), Python 3.14, mlx-lm 0.31.3,
  `MLX_DISABLE_COMPILE=1`, `HF_HUB_OFFLINE=1`, model snapshot
  `a5339a41…` (pinned), `bench_decode.py --tokens 32 --temp 0.0
  --seed 0 --warmup-tokens 4`, fresh process per leg.
- Every GPU step ran under ONE `flock` hold on `/tmp/m1-gpu.lock`
  (inode 12 before and after; nested `flock -n` refused; never
  unlinked). `63c1d3cf` not merged; no RoPE in qmm_vec.
- Protocol error caught and corrected mid-receipt: the setup step
  initially copied the 2026-09-14 lane's `digest_ctx1024.py` over this
  lane's, so the first digest arm ran the stale 09-14 wheels and gate.
  The arm was re-run with the correct wheels (`MLX_OMARCHY_FUSED_GEMV_
  SWIGLU=0` for the off arm) before any number was quoted; the A/B and
  dispatch counts were never affected.

## Artifacts

- `receipts/2026-09-15-swiglu-epilogue-jw16/`: `dispatch-{base,cand,
  cand-off}.json`, `digest.txt`, `ab.txt`, `ab.json`, `lock.txt`,
  `host.txt`, `started.txt`, `finished.txt`, `wheel-sha256.txt`.
- jw16: `/var/tmp/SwigluRmsJw16/` (worktrees, wheels, venvs,
  `run-confirm.log`, `build.log`).
