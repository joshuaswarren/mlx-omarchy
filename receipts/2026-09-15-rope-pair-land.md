# 2026-09-15 Rope-pair dispatch + targeted linear chain-settle land on origin/main (1014a76e) on jw16 — LAND
## Verdict
**LAND.** The winning decode-trio content (layout 519d7336 + non-detaching
nested eval d6b87ee4 + targeted linear chain-settle bd3e5726) is extracted
as a minimal single-commit branch off origin/main `1014a76e`:
`agent/rope-pair-linear-land` at `2f20d485`, 10 files, +1191/−5. The
branch wheel's default configuration (no env) measures **225
vk_compute_dispatches/token** vs the origin/main wheel's 249, both digest
pins are exact on all 20 battery legs, and the ctx1053 (bench leg
"ctx1024") median rises +6.0% over the origin/main wheel (the reference
batteries on the older `b41e2b74` base measured +0.7%/+2.7%). Carries no
ring (ctx −2%, rejected twice), no detaching nested eval (8f1ed713 was
superseded by d6b87ee4's depth guard), no megashader change, `qmm_vec.comp`
untouched, `63c1d3cf` not merged.
## What was built (three-way merge, base = merge-root b41e2b74)
- `overlay/mlx/backend/omarchy/primitives.cpp` — `dispatch_rope_pair` with
  per-side layout resolution, the pre-settle that records only the
  unsettled producer chain (producer-first `gpu::eval` + status marking,
  `settle()` kept as the non-linear fallback), trio plan/refusal tracing.
  One merge conflict: origin/main's `dispatch_dense_gemv_group` and the
  chain's `dispatch_rope_pair` were both added at the same anchor; resolved
  by keeping both functions (git consumed main's trailing
  `return true; }` into the common suffix — re-inserted).
- `overlay/mlx/backend/omarchy/fused_chain.cpp/.h` — rope-pair plan scan,
  `MLX_OMARCHY_FUSED_TRIO` gate (default ON), trio trace.
- `overlay/mlx/backend/omarchy/eval.cpp` — trio trace + skip guard.
- `overlay/mlx/backend/omarchy/compute.cpp/.h`, `CMakeLists.txt`,
  `shaders/fast_trio.comp` — the three trio shaders (norm / rope-pair /
  swiglu, exactly-one-of), kernel ids appended.
- `patches/mlx-rope-settle-tape.patch` + `scripts/prepare-mlx.sh` — the
  non-detaching nested-eval guard (`eval_nest_depth` in upstream mlx
  `transforms.cpp`; a nested pass never detaches nodes an outer tape still
  owns). This is d6b87ee4's depth guard, not 8f1ed713's detaching
  experiment.
- Dropped from the merge: `receipts/2026-09-13-jw16-m1max-gpu-performance.json`
  (unrelated m1max perf data that rode along on the chain).
- Gate defaults: `MLX_OMARCHY_FUSED_TRIO` ON, `MLX_OMARCHY_KV_DIRECT` ON
  (the `KV_DIRECT=0` merged fallback measures 249 on the branch wheel —
  the base behaviour is intact behind the gate). No new gates; the default
  build IS the 225-dispatch winning configuration.
## Measurements (jw16, MLX_DISABLE_COMPILE=1, HF_HUB_OFFLINE=1,
## Qwen2.5-0.5B-Instruct-4bit snapshot a5339a4131f135d0…, 5-round
## interleaved A/B, battery `jw16-out-rp-land`)
| leg | origin/main 1014a76e (249 vk) | branch 2f20d485 (225 vk) | delta |
|---|---|---|---|
| short median | 167.97 | 178.58 | +6.3% |
| ctx1053 median | 135.95 | 144.07 | +6.0% |
(per-round values in `ab.json`; base rounds short 167.11–170.15, ctx
126.01–140.62; arm rounds short 177.55–179.11, ctx 130.91–146.56.)
Dispatch probes (8 tokens): branch wheel default 225 ×3/3 exact;
`MLX_OMARCHY_KV_DIRECT=0` → 249 (merged fallback intact); base wheel 249.
Pins exact on all 20 battery legs (`7fd25a869ff21678` short,
`7da83f06ec9f001d` ctx1024); ab_decode asserts per leg and stops on any
mismatch — none. One flock hold on `/tmp/m1-gpu.lock` for the whole
battery, inode 12 before and after, never unlinked, nested `flock -n`
refused. One pre-battery smoke probe of the branch wheel ran lock-free
while the lock was free and no other holder existed — recorded here for
completeness. No SEGV anywhere: every A/B leg, probe, and the kvdir0
fallback completed rc=0.
The origin/main base wheel measures ctx1053 median 135.95 — below the
`b41e2b74` base range from the reference batteries (137.49/140.28) — so
main's own drift (dense GEMV group, qmm coopmat m16) costs ctx; the
branch's +6.0% is measured against main as it stands, which is the
comparison that matters for landing.
## Decision
`agent/rope-pair-linear-land` at `2f20d485` is a single-parent commit on
`origin/main` (`1014a76e`) carrying exactly the 10-file feature diff, this
receipt included. Ready to merge; the diff applies cleanly to
`1014a76e` by construction.
## Artifacts
- Branch wheel `mlx_omarchy-0.32.2.dev202609151832+2f20d485-cp314-cp314-linux_aarch64.whl`
  sha256 `eac805ba1608d8175e08ae88c68db026e7a7d062e1027dd228ffc3354a835ace`
  (`/var/tmp/rope-pair-land/dist/`, installed in `venv-trio`-style
  private venv `/var/tmp/DecodeTrioRun/venv-rp-arm`).
- Base wheel `mlx_omarchy-0.32.2.dev202609151835+1014a76e-cp314-cp314-linux_aarch64.whl`
  sha256 `08678917247e8265ebe1ca243416d5630860b7b31eea284647c7fac8ca3fb0bb`
  (`/var/tmp/main-base/dist/`, venv `/var/tmp/DecodeTrioRun/venv-rp-base`).
- Battery: `/var/tmp/DecodeTrioRun/jw16-out-rp-land/` (dispatch probes
  ×3 default + kvdir0 + base probe, provenance lines, 5-round interleaved
  `ab.json`, wheel sha256s, lock records, started/finished stamps).
  Runner: `/var/tmp/decode-trio/run-rpland-battery.sh`; build logs
  `/tmp/rp-land-build.log`, `/tmp/rp-base-build.log`; battery log
  `/tmp/rpland-battery.log`.
- Harness reused unchanged: `/var/tmp/DecodeEpilogueFold/ab_decode.py`
  (default 5 rounds), `/var/tmp/DecodeCompileAB/dispatch_count.py`,
  `/var/tmp/mlx-omarchy-prof-b41e2b74/scripts/{bench_decode.py,bench_matrix.json}`.
