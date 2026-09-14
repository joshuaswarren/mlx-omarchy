# 2026-09-13 jwm1 current-wheel 1053-token Q4 GPU phase profile

Date: 2026-09-13

Host: `jwm1-linux`. GPU lock inode 29 taken and released. ANE not touched. No reboot.

## Verdict

The installed `b41e2b74` wheel ran the m1053 protocol (`profile_generate.py`, 1053 chat-template tokens, `--max-tokens 2`, `MLX_DISABLE_COMPILE=1`, `MLX_OMARCHY_GPU_PROFILE` set). Generation succeeded (`Thank you`, two tokens). Host markers exist.

**No GPU kernel table.** This wheel is a release build: `MLX_OMARCHY_GPU_PROFILING` defaults OFF, and `libmlx.so` does not contain the `MLX_OMARCHY_GPU_PROFILE` literal. The env var is a no-op. Busy fraction and `QmmPrefillCoopmatF16` dispatch counts were not measured on this wheel.

A same-commit `--diagnostics` rebuild is required before those GPU numbers can exist. That rebuild was not done here: it would replace `dist/` in the installed tree and is not the installed venv.

## Identity

- Source commit: `b41e2b74c330f910b24cab0e7516e306527858f0`, dirty false.
- Wheel: `mlx-omarchy==0.32.2.dev202609122355+b41e2b74`.
- `libmlx.so` SHA-256 `03b3f4b9024927f90ada96950579aec5c603f5d6ec08f2337f12f9379f3e5515`.
- Python: installed venv `/home/joshuawarren/src/mlx-bf16-grouped-base-b41e2b74/.work/venv-run/bin/python`.
- Device: `Device(gpu, 0)`.
- Model: Qwen2.5-0.5B-Instruct-4bit, config SHA-256 `b045e57ea90b8f1b35f89f954b176a5c1faa02bd0af2c89bcec191239d66cef4`.
- Prompt: `bench_matrix` `ctx1024`, 4759 UTF-8 bytes, **1053** chat-template tokens.
- Agent model: `openai-codex/gpt-5.6-sol`; no routing fallback.

Positive control: `libmlx.so` contains `MLX_DISABLE_COMPILE` and does not contain `MLX_OMARCHY_GPU_PROFILE`.

## Host phases (markers)

| Phase | ms |
| --- | ---: |
| load | 733.616 |
| prefill | 1177.861 |
| decode (2 tokens) | 19.942 |
| first inter-token | 14.346 |

Host prefill rate: 1053 / 1.177861 s = **893.99 tok/s**.

## Comparison to pinned m1053

Pinned profile: `receipts/2026-09-10-prefill-qmm-isa/distribution/m1053/` (older wheel, GPU profiler compiled in).

| Metric | Pinned m1053 | b41e2b74 installed | Status |
| --- | ---: | ---: | --- |
| host prefill ms | 1321.839 | 1177.861 | measured; 10.9% faster host wall |
| GPU busy fraction | 77.88% | — | **unavailable** on release wheel |
| GPU busy ms | 943.089 | — | **unavailable** |
| GPU span ms | 1210.942 | — | **unavailable** |
| prefill dispatches | 1231 | — | **unavailable** |
| prefill submissions | 53 | — | **unavailable** |
| `QmmPrefillCoopmatF16` n / ms / share | 163 / 702.476 / 74.49% | — | **unavailable** |
| `MatmulRbF16` n / ms / share | 46 / 109.419 / 11.60% | — | **unavailable** |

Pinned kernel ranking remains the last GPU-timestamped m1053 picture. It is not a current-wheel measurement.

## Lock and safety

- `/tmp/m1-gpu.lock` inode 29 held exclusively; nested `flock -n` returned 1.
- Lock not unlinked. Final `fuser` after the run: free.
- `ane.ko` not unloaded; no ANE command.
- Installed venv and `dist/` wheel left unchanged.

## Artifacts

`markers.jsonl`, `run.log`, `run-meta.json`, `provenance.json`, `lock.json`, `identity.json`, `host-phases.json`, `comparison.json`, `profiler-gate.json`, `commands.txt`.
