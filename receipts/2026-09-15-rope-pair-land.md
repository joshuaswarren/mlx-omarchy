# 2026-09-15 Rope-pair dispatch + targeted linear chain-settle land on origin/main (1014a76e) on jw16 — LAND
## Verdict
**LAND.** The winning decode-trio content (layout 519d7336 + non-detaching
nested eval d6b87ee4 + targeted linear chain-settle bd3e5726) is extracted
as a minimal branch off origin/main `1014a76e`. The feature commit is
`6db6377b` (10 files, +1191/−5); the branch tip carries this receipt. The
final wheel's default configuration (no env) measures **225
vk_compute_dispatches/token** vs the origin/main wheel's 249, both digest
pins are exact on all 20 legs of both A/B batteries, and the ctx1053 (bench
leg "ctx1024") median rises over the origin/main wheel in both (final
battery +2.8%, bring-up battery +6.0%; the two reference batteries on the
older `b41e2b74` base measured +0.7%/+2.7%). Carries no ring (ctx −2%,
rejected twice), no detaching nested eval (8f1ed713 was superseded by
d6b87ee4's depth guard), no megashader change, `qmm_vec.comp` untouched,
`63c1d3cf` not merged.
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
## Provenance note: commit hashes and the email rewrite
The first committed pair of commits (`2f20d485` + receipt `9f78b0fb`) was
authored with a private email that GitHub's push restrictions reject. Both
commits were rewritten to the noreply address (`6db6377b` + first receipt
`1c49674f`); `git diff 2f20d485 6db6377b` is empty — tree content is
byte-identical, only author/committer metadata changed. The wheel built
from the rewritten tree stamps `+1c49674f`, which exists in this branch's
history, and is the wheel the final battery below measured. Wheel stamps
`+2f20d485` (bring-up) and `+1014a76e` (base) are likewise historical
hashes of identical or ancestor trees.
## Measurements (jw16, MLX_DISABLE_COMPILE=1, HF_HUB_OFFLINE=1,
## Qwen2.5-0.5B-Instruct-4bit snapshot a5339a4131f135d0…, 5-round
## interleaved A/B vs the origin/main wheel)
Final battery (`jw16-out-rp-land-b`, wheel `+1c49674f`):
| leg | origin/main 1014a76e (249 vk) | branch (225 vk) | delta |
|---|---|---|---|
| short median | 168.12 | 176.77 | +5.2% |
| ctx1053 median | 141.17 | 145.17 | +2.8% |
Bring-up battery (`jw16-out-rp-land`, wheel `+2f20d485`, same tree):
| leg | origin/main 1014a76e (249 vk) | branch (225 vk) | delta |
|---|---|---|---|
| short median | 167.97 | 178.58 | +6.3% |
| ctx1053 median | 135.95 | 144.07 | +6.0% |
Dispatch probes (8 tokens, both wheels): branch default 225 ×3/3 exact
per battery; `MLX_OMARCHY_KV_DIRECT=0` → 249 (merged fallback intact);
base wheel 249. Pins exact on all 20 legs per battery
(`7fd25a869ff21678` short, `7da83f06ec9f001d` ctx1024); ab_decode asserts
per leg and stops on any mismatch — none. One flock hold on
`/tmp/m1-gpu.lock` per battery, inode 12 before and after, never unlinked,
nested `flock -n` refused. One pre-battery smoke probe of the first branch
wheel ran lock-free while the lock was free and no other holder existed —
recorded here for completeness. No SEGV anywhere: every A/B leg, probe,
and the kvdir0 fallback completed rc=0.
The origin/main base wheel measures ctx1053 median 135.95 (battery A) /
141.17 (battery B) — the A run sat below the `b41e2b74` base range from
the reference batteries (137.49/140.28), B inside it. Both batteries show
the branch arm above its own same-run base at ctx and short, which is the
comparison that matters for landing.
## Decision
`agent/rope-pair-linear-land` off `origin/main` (`1014a76e`) carries
exactly the 10-file feature diff (`6db6377b`), this receipt, and nothing
else. Ready to merge; the diff applies cleanly to `1014a76e` by
construction (single-parent commits, no merge commits).
## Artifacts
- Final branch wheel
  `mlx_omarchy-0.32.2.dev202609151843+1c49674f-cp314-cp314-linux_aarch64.whl`
  sha256 `96a2594a03b519991b40798c234e920c53f03a50bb054cfdcd1c01b50292bf70`
  (`/var/tmp/rope-pair-land/dist/`, installed in
  `/var/tmp/DecodeTrioRun/venv-rp-arm`).
- Bring-up branch wheel
  `mlx_omarchy-0.32.2.dev202609151832+2f20d485-cp314-cp314-linux_aarch64.whl`
  sha256 `eac805ba1608d8175e08ae88c68db026e7a7d062e1027dd228ffc3354a835ace`
  (superseded stamp, identical tree).
- Base wheel
  `mlx_omarchy-0.32.2.dev202609151835+1014a76e-cp314-cp314-linux_aarch64.whl`
  sha256 `08678917247e8265ebe1ca243416d5630860b7b31eea284647c7fac8ca3fb0bb`
  (`/var/tmp/main-base/dist/`, venv `/var/tmp/DecodeTrioRun/venv-rp-base`).
- Batteries: `/var/tmp/DecodeTrioRun/jw16-out-rp-land-b/` (final) and
  `/var/tmp/DecodeTrioRun/jw16-out-rp-land/` (bring-up) — dispatch probes
  ×3 default + kvdir0 + base probe, provenance lines, 5-round interleaved
  `ab.json`, wheel sha256s, lock records, started/finished stamps.
  Runner: `/var/tmp/decode-trio/run-rpland-battery.sh` (OUT retargeted per
  run); build logs `/tmp/rp-land-build.log`, `/tmp/rp-land-build2.log`,
  `/tmp/rp-base-build.log`; battery logs `/tmp/rpland-battery.log`,
  `/tmp/rpland-battery-b.log`.
- Harness reused unchanged: `/var/tmp/DecodeEpilogueFold/ab_decode.py`
  (default 5 rounds), `/var/tmp/DecodeCompileAB/dispatch_count.py`,
  `/var/tmp/mlx-omarchy-prof-b41e2b74/scripts/{bench_decode.py,bench_matrix.json}`.
