# Encoder wall attribution: where the 19.08 s goes on jwm1 (2026-09-15)

Attribution only. No kernel changed, no runner changed, `63c1d3cf` not merged.
Two phase-isolated `MLX_OMARCHY_GPU_PROFILE` captures of the encoder leg only,
plus per-submit timestamps from an instrumented stage copy of the runner.

## Verdict

The 19.08 s encoder wall (fused runner, islands ABC, `05015a76` wheel) is
**half GPU-busy ConvF32/MatmulF32, one quarter ANE one-shot worker subprocess,
one quarter host staging and graph gaps.** Instrumented capture: wall
19281.0 ms against the clean 19194.3 ms standalone leg (receipts/
`2026-09-15-encoder-fused-leftover.md`) and 19080.0 ms E2E — within 0.5%.

Three-way split of the 19281 ms (device and ANE-subprocess windows are
disjoint by construction: every `mx.eval` joins before `subprocess.run`
blocks the only submitting thread):

| bucket | ms | % of wall |
| --- | ---: | ---: |
| GPU busy (5202 vk dispatches, GPU ticks) | 9785.8 | 50.8% |
| ANE worker subprocess (72 one-shot submits) | 4999.5 | 25.9% |
| host: island staging + graph dispatch + gaps | 4495.7 | 23.3% |
| **total** | **19281.0** | 100.0% |

Host-clock buckets from the instrumented runner (sum to the same wall):

| bucket | ms | % of wall |
| --- | ---: | ---: |
| ANE worker subprocess | 4999.5 | 25.9% |
| island input stage — `mx.eval` drain + np convert | 10773.7 | 55.9% |
| island input stage — input file writes (~300 MB) | 428.4 | 2.2% |
| island output stage — read + `mx.array` host→dev (~234 MB) | 387.5 | 2.0% |
| host graph residual (statements, python, profiler) | 2691.8 | 14.0% |

GPU busy sits inside the eval-drain bucket (island inputs force the join);
the ~1.0 s difference between 10.77 s drain and 9.79 s busy is host dispatch
cost inside `mx.eval`. GPU idle-in-span is 9416.7 ms (GPU span 19202.6 ms):
the ANE waits plus host graph time between evals.

### Top GPU kernels (device ticks, `period_ns=1`)

| kernel | dispatches | GPU busy | % of busy | % of wall |
| --- | ---: | ---: | ---: | ---: |
| ConvF32 | 77 | 6134.0 | 62.7% | 31.8% |
| MatmulF32Coopmat | 194 | 1453.2 | 14.8% | 7.5% |
| Custom (fused LN/silu/GLU/sm chains) | 698 | 823.5 | 8.4% | 4.3% |
| ElementwiseF32 | 762 | 598.1 | 6.1% | 3.1% |
| CopyGeneralF16 | 605 | 319.2 | 3.3% | 1.7% |
| CastF16F32 | 1616 | 163.3 | 1.7% | 0.8% |
| everything else | 1052 | 294.5 | 3.0% | 1.5% |

ConvF32 shapes: `(1,2048,375)`×24 = 3968.3 ms, `(1,1024,375)`×48 = 1745.0 ms,
five stem convs = 420.6 ms. Fused custom kernels: 553 dispatches at grid
(1500,1,1) = 495.3 ms, 96 at grid (6000,1,1) = 320.9 ms (the GLU), tail 7.3 ms.

### ANE 5.0 s split (72 one-shot submits, 0 timeouts)

Per-island means from the run log (subprocess = spawn+parse+file IO+in-worker;
in-worker = program load + device exec, split by an `--iterations 1/2/4` probe
on the same bundles under the lock):

| island | submits | subprocess ms | spawn/parse/IO ms | in-worker ms | device exec ≈ ms | program load ≈ ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A (attn-a-kt, 2 programs) | 24 | 88.5 | 21.6 | 66.9 | ~38–40 | ~27 |
| B (select-8head) | 24 | 79.2 | 22.2 | 57.0 | ~17–24 | ~33–40 |
| C (pv) | 24 | 40.6 | 15.4 | 25.2 | ~12–17 | ~8–13 |
| **Σ** | 72 | **4999.5** | **≈1420** | **≈3580** | **≈1800** | **≈1700** |

Probe receipts (island-pv): iterations 1/2/4 → worker elapsed_ms 14/19/35;
island A → 59/94/180; island B → 60/84/110.

So the "leftover" 14 s (wall − ANE exec) decomposes as **9.79 s GPU busy
(67% ConvF32) + 4.50 s host**. There is no fourth bucket: no ANE/GPU overlap
exists to hide work in, and the per-dispatch host cost of the profiler
(~50 µs × 5202) is inside the 2.69 s residual, not hiding a gap.

## Priced next attack: ship a wheel built at ≥ `e55c1fae`

**Expected −6.5 s (−33.7%) on the encoder stage, zero code to write.**

The unit-window 1×1 `ConvF32 → MatmulF32` dispatch (`e55c1fae`, landed in
origin/main 2026-09-14 15:32) is **not in the `05015a76` wheel** every
parakeet receipt since has measured on — `05015a76` sits on a branch that
diverged before the conv land, whatever its build timestamp says
(`git merge-base --is-ancestor 05015a76 e55c1fae` is false).

Measured on this host today, identical runner, identical 5202 dispatch count,
`encoder_hidden` bit-identical to the pin `38c73261…` in **both** captures:

| capture | backend wheel | ConvF32 busy | GPU busy | wall |
| --- | --- | ---: | ---: | ---: |
| record | `05015a76` +diag (this task) | 6134.0 ms | 9785.8 ms | **19281.0 ms** |
| record | `108fd4b5` +diag (origin/main source) | **82.0 ms** | **4059.0 ms** | **12781.3 ms** |

The 24+48 conformer convs ride `MatmulF32` (336.3 ms) and the coopmat
matmuls; GPU busy drops 5726.8 ms, wall drops 6499.7 ms instrumented. This is
consistent with the clean −6253 ms already receipted on the pre-fused runner
(receipts/`2026-09-14-encoder-conv-f32.md`: 18070.7 → 11817.6 ms). The action
is a wheel/pin bump on the E2E path plus one re-run of the E2E receipts.

After that lands, the wall is ~12.8 s: GPU busy 4.06 s (MatmulF32Coopmat
1453 ms then Custom 823 ms), ANE subprocess 5.0 s (spawn 1.4 / load ~1.7 /
exec ~1.8 — island batching and a resident worker each address ~1.4–1.7 s and
are already owned elsewhere), host 3.7 s.

## Identity

- Host `jwm1-linux`, aarch64, 8 cores, kernel `7.1.6-1-1-ARCH`, `/dev/accel/accel0`,
  Vulkan `Mesa Honeykrisp`, device `Apple M1 (G13G B1)`, `timestamp_period_ns: 1`.
- Lock `/tmp/m1-gpu.lock` inode 29, `flock -w 900`, never stolen, never
  unlinked; held 18:16:17–18:17:13, 18:28:13–18:28:53 and 18:29:35–18:29:38
  local (-05:00); `flock -n` free after each; siblings announced between windows.
- Runner: fused `108fd4b5` file (`6c175adf…`) + additive timestamp
  instrumentation only (22 lines: `started_ns`/`submit_started_ns`,
  `stage_eval_ns`/`stage_write_ns`/`stage_out_ns`, run-level markers) — stage
  copy `/var/tmp/ParakeetE2EFusedLeftover-stage/vulkan_encoder_instr.py`
  sha256 `3fea3ec32e68b761672c2fe7a23f2c6783f255a41ff91726e2c82d3c12e69116`.
- Diag wheels built on jwm1 with `-DMLX_OMARCHY_GPU_PROFILING=ON`
  (recipe of `/var/tmp/mlx-omarchy-profile-enabled-b41e2b74/build.sh`),
  staged trees via `prepare-mlx.sh`, same mlx pin `1f8e74e3`:
  - `mlx_omarchy-0.32.2.dev202609152309+diag.108fd4b5` sha256
    `0251ed78b3c7ae17ed33296c1c5bb6a06ec581c7efcfdfcf16aa74effaebf907`
  - `mlx_omarchy-0.32.2.dev202609152323+diag.05015a76` sha256
    `e70732ec8b2666aa752ef35cd4fbaf6e33ff41434f03770e2900cd409c5ce8d0`
  - profiler gate: `MLX_OMARCHY_GPU_PROFILE` literal ×3 in both
    `libmlx.so` (`65bc0869…`, `2b89da0d…`); release wheels carry 0.
- Correctness: `encoder_hidden` = pin `38c73261f29230276ed76f1fc017b76b024156d79218bd5f1347fdc7e7d43ec7`
  in all four legs (both wheels × warm/record); 72 submits, 0 timeouts;
  vk compute dispatches 5202 = the fused receipt's count; island-B census
  unchanged; warm and record walls 19986/19281 ms (record quoted; first run
  populates the SPIR-V cache).
- Artifacts on jwm1: `/var/tmp/enc-wall-prof-b0/` (profile-record.jsonl
  `e75cbea1…`, run-report.json `cbaa4dd3…`), `/var/tmp/enc-wall-prof/`
  (profile-record.jsonl `5c0c99b9…`, run-report.json `37ba2d24…`).
- Interpreter `/home/joshuawarren/venv-agxgen/bin/python` +
  `PYTHONPATH=<diag-site>:MelFrontendPerf venv-cache`; `MLX_OMARCHY_SPIRV_CACHE`
  private per capture; no `HK_PERF`/`MLX_OMARCHY_GATED_BARRIERS`.
- Analysis: profile `d` records (GPU ticks) + instrumented report; kernel
  names from each commit's `compute.h` enum order.

## Not claimed

- No clean (unprofiled) re-measurement: profiled walls carry the profiler's
  per-dispatch cost; clean deltas are quoted from existing receipts.
- No code change landed; `63c1d3cf` not merged; no other fixture, no decode,
  no macOS, no ANE kernel or bundle change.
- In-worker load/exec split is a ±10 ms-per-island estimate from the
  iterations probe, not per-layer instrumentation.
- The 108fd4b5 capture ran today's tree, not a wheel anyone ships; it prices
  the conv-reorder win, it does not ship it.
