# 2026-09-01 — macOS native MLX baseline (16m1mbp)

Purpose: native-MLX-on-Metal baseline for the mlx-omarchy README performance
section, using the same model and prompts as the Linux (jwm1-linux) MLX+mlx-lm
receipts. Run from a Linux workstation over SSH; every command below ran on
16m1mbp via `ssh -o BatchMode=yes 16m1mbp '<cmd>'`.

## Host identities (chip caveat)

| Host | Chip | GPU | CPU cores | Memory | OS |
|---|---|---|---|---|---|
| 16m1mbp (this run) | **Apple M1 Max** (`sysctl machdep.cpu.brand_string`) | **32-core**, Metal 4 | 10 (8P+2E) | 64 GB unified | macOS 26.6.2 (25G83), arm64 |
| jwm1-linux (Linux receipts) | Apple M1 (T8103) | 8-core | 8 | — | Linux, ane-linux-experiments work |

`system_profiler SPHardwareDataType` on 16m1mbp: `Chip: Apple M1 Max`,
`Total Number of Cores: 10 (8 Performance and 2 Efficiency)`.
`SPDisplaysDataType`: `Chipset Model: Apple M1 Max`,
`Total Number of Cores: 32`, `Metal Support: Metal 4`.
MLX reports Metal GPU `applegpu_g13s`, 68719476736 bytes memory.

**Caveat, stronger than expected:** 16m1mbp was assumed to be an M1 Pro (T6000).
It is an **M1 Max** (g13s). The Linux comparison box is a **base M1** (T8103,
8-core GPU, lower memory bandwidth: ~200 GB/s vs ~400 GB/s). Per-core
comparisons are cross-chip, not like-for-like. Both identities are stated
precisely so the README can compare honestly.

## Software (upstream PyPI, fresh venv)

- Venv: `~/src/mlx-bench-20260901/venv`, interpreter
  `/opt/homebrew/bin/python3.12` (Python 3.12.9; Xcode system `python3` is
  3.9.6, too old for mlx).
- Install: `pip install "mlx" "mlx-lm==0.31.3"` — resolver held the pin, no
  conflict.
- Resolved versions: **mlx 0.32.2**, **mlx-lm 0.31.3**, transformers 5.16.1.

## Models

- `mlx-community/Qwen2.5-0.5B-Instruct-4bit`, pinned to the Linux receipt
  revision `a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3` (snapshot_download with
  `revision=`; local snapshot path confirms the SHA).
- `mlx-community/Qwen2.5-0.5B-Instruct-bf16`, latest:
  `56d07e766edd7159fbe12ed12d9cf114bf38bf1e`.

## Command

Per run (both attempts identical):

```
python -m mlx_lm.generate --model <snapshot-path> \
  --prompt '<prompt>' --max-tokens 32 --temp 0 --seed 0
```

mlx-lm 0.31.3 applies the model chat template by default (no
`--ignore-chat-template`), same as the Linux runs on the same mlx-lm version.
Reported numbers are the second (warm) attempt; first (cold) attempt shown for
reference. Output below is verbatim CLI output.

## Results — 4-bit (prompt: "What is the capital of France? Answer in one word.")

Attempt 2 (warm, reported):

```
==========
Paris
==========
Prompt: 41 tokens, 596.365 tokens-per-sec
Generation: 2 tokens, 225.051 tokens-per-sec
Peak memory: 0.347 GB
```

Attempt 1 (cold, reference):

```
==========
Paris
==========
Prompt: 41 tokens, 50.376 tokens-per-sec
Generation: 2 tokens, 208.230 tokens-per-sec
Peak memory: 0.347 GB
```

## Results — bf16 (prompt: "Hi")

Attempt 2 (warm, reported):

```
==========
Hello! How can I assist you today?
==========
Prompt: 30 tokens, 341.549 tokens-per-sec
Generation: 10 tokens, 88.756 tokens-per-sec
Peak memory: 1.076 GB
```

Attempt 1 (cold, reference):

```
==========
Hello! How can I assist you today?
==========
Prompt: 30 tokens, 37.992 tokens-per-sec
Generation: 10 tokens, 91.603 tokens-per-sec
Peak memory: 1.076 GB
```

## Honest-read notes

- Both runs terminated on EOS well before `--max-tokens 32` (2 and 10
  generated tokens). Greedy decoding (temp 0, seed 0) makes this
  deterministic; the same prompts on Linux produce the same short outputs.
- Generation tok/s over 2- and 10-token runs is dominated by per-token
  overhead, not throughput: 4-bit ~4.4 ms/token (1/225), bf16 ~11.3 ms/token
  (1/88.8). Treat these as latency-flavored numbers, not steady-state tok/s.
- Prompt tok/s (warm) is a 41- and 30-token prefill; also short-run noisy.
- Peak memory (MLX Metal allocator): 0.347 GB (4-bit), 1.076 GB (bf16).
- CLI printed a deprecation notice for `python -m mlx_lm.generate`
  (suppressed here; `mlx_lm.generate` console script is the supported entry).
  It does not affect measurements.

## Reproduction

```
/opt/homebrew/bin/python3.12 -m venv ~/src/mlx-bench-20260901/venv
~/src/mlx-bench-20260901/venv/bin/pip install "mlx" "mlx-lm==0.31.3"
~/src/mlx-bench-20260901/venv/bin/python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("mlx-community/Qwen2.5-0.5B-Instruct-4bit", revision="a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3")); print(snapshot_download("mlx-community/Qwen2.5-0.5B-Instruct-bf16"))'
~/src/mlx-bench-20260901/venv/bin/python -m mlx_lm.generate \
  --model ~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3 \
  --prompt 'What is the capital of France? Answer in one word.' --max-tokens 32 --temp 0 --seed 0
```
> **Annotation 2026-09-03:** the decode tok/s figures in this document are EOS-truncated short-burst rates, not steady-state decode (generation stopped after 2-10 tokens under `--max-tokens 32`). They are not comparable across machines or wheels. See `receipts/2026-09-03-decode-metric-fix.md`; replacement protocol: `scripts/bench_decode.py` (pinned length, token-count assertion).

