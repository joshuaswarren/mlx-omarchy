# 2026-09-13 jwm1 profile-enabled 1053-token Q4 kernel table

Date: 2026-09-13

Host: `jwm1-linux`. GPU lock inode 29 taken for the generate run only and released. ANE not touched. No reboot.

## Verdict

`b41e2b74` rebuilt with `-DMLX_OMARCHY_GPU_PROFILING=ON` into `/var/tmp/mlx-omarchy-profile-enabled-b41e2b74`. The m1053 protocol ran (`profile_generate.py`, 1053 chat-template tokens, `--max-tokens 2`, `MLX_DISABLE_COMPILE=1`). Generation succeeded (`Thank you`). `libmlx.so` contains `MLX_OMARCHY_GPU_PROFILE`. `profile.jsonl` was written.

Prefill-window GPU busy fraction **0.8354**. `QmmPrefillCoopmatF16` **163 / 702.559 ms / 75.17%** of prefill busy.

## Identity

- Source commit: `b41e2b74c330f910b24cab0e7516e306527858f0`, dirty false.
- Wheel: `mlx-omarchy==0.32.2.dev202609122355+diag.b41e2b74` (private prefix, grouped-base `dist/` unchanged).
- Wheel SHA-256 `e0ce679e545850a4b5ea2f1385e9061b0255e2632c8685ee69d1850016815244`.
- `libmlx.so` SHA-256 `d41f311ec19e23634213e420ec3f131ddb83509a87a16aee264b323d2fcc7051`.
- Python: `/var/tmp/mlx-omarchy-profile-enabled-b41e2b74/venv/bin/python`.
- Device: `Device(gpu, 0)` / `Apple M1 (G13G B1)`.
- Model: Qwen2.5-0.5B-Instruct-4bit, config SHA-256 `b045e57ea90b8f1b35f89f954b176a5c1faa02bd0af2c89bcec191239d66cef4`.
- Prompt: `bench_matrix` `ctx1024`, 4759 UTF-8 bytes, **1053** chat-template tokens.
- Agent model: `openai-codex/gpt-5.6-sol`; no routing fallback.

CMake: `option(MLX_OMARCHY_GPU_PROFILING ... OFF)` in `overlay/mlx/backend/omarchy/CMakeLists.txt`. Env var `MLX_OMARCHY_GPU_PROFILE` is a no-op unless that option is ON. Bounded rebuild reused 422 cached SPIR-V headers and recompiled mlx C++ with `-DMLX_OMARCHY_GPU_PROFILING` (17:42:43–17:45:52).

## Host phases (markers)

| Phase | ms |
| --- | ---: |
| load | 669.418 |
| prefill | 1260.591 |
| decode (2 tokens) | 27.562 |
| first inter-token | 19.752 |

Host prefill rate: 1053 / 1.260591 s = **835.32 tok/s**.

## GPU (prefill window vs pinned m1053)

Pinned profile: `receipts/2026-09-10-prefill-qmm-isa/distribution/m1053/` (older wheel, GPU profiler compiled in). Attribution: dispatch phase is the submit-record host time, same contract as `scripts/profile_analyze.py`.

| Metric | Pinned m1053 | b41e2b74 profile-enabled | Status |
| --- | ---: | ---: | --- |
| host prefill ms | 1321.839 | 1260.591 | measured |
| GPU busy fraction | 0.7788 | **0.8354** | prefill window |
| GPU busy ms | 943.089 | 934.581 | prefill window |
| GPU span ms | 1210.942 | 1118.761 | prefill window |
| prefill dispatches | 1231 | 1063 | |
| prefill submissions | 53 | 10 | |
| `QmmPrefillCoopmatF16` n / ms / share | 163 / 702.476 / 74.49% | **163 / 702.559 / 75.17%** | |
| `MatmulRbF16` n / ms / share | 46 / 109.419 / 11.60% | 46 / 106.376 / 11.38% | |

Analyzer whole-run headline (all phases): busy **83.33%** (951.961 / 1142.375 ms), `QmmPrefillCoopmatF16` 73.8% of all-phase busy. Prefill-window share is the comparison number.

## Prefill kernel table (top)

| Kernel | n | gpu ms | share |
| --- | ---: | ---: | ---: |
| QmmPrefillCoopmatF16 | 163 | 702.559 | 75.17% |
| MatmulRbF16 | 46 | 106.376 | 11.38% |
| SoftmaxF16 | 23 | 27.411 | 2.93% |
| QmmVecQ4MultiSubgroupF16 | 192 | 22.713 | 2.43% |
| SwigluF16 | 71 | 22.046 | 2.36% |
| CopyGeneralF16 | 71 | 10.845 | 1.16% |
| BinaryVecF16 | 117 | 8.230 | 0.88% |
| FastRmsNormF16 | 145 | 7.895 | 0.84% |

## Lock and safety

- `/tmp/m1-gpu.lock` inode 29 held exclusively during generate; nested `flock -n` returned 1.
- Lock not unlinked. Final `fuser` after the run: free (rc 1).
- `ane.ko` present in `lsmod`; no ANE command.
- Grouped-base `dist/` wheel and `.work/venv-run` left unchanged.

## Artifacts

`analysis.txt`, `profile.jsonl`, `markers.jsonl`, `prefill-summary.json`, `run.log`, `run-meta.json`, `provenance.json`, `lock.json`, `identity.json`, `host-phases.json`, `comparison.json`, `profiler-gate.json`, `commands.txt`, `rebuild.log`.
