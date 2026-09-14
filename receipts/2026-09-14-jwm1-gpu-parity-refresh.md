# jwm1 GPU Q4 parity refresh

Date: 2026-09-14

## Verdict

`/tmp/m1-gpu.lock` inode 29 was free. No CTS/`deqp`/`cmshape2` holder. The lock was taken with `flock -w 60` (not stolen) and released. Nested `flock -n` returned 1 while held.

The installed `b41e2b74` wheel on `jwm1-linux` still sits at 68.53% of native macOS decode on the 1053/32 Q4 leg and 60.38% of native 1053-token prefill. Short-prompt decode is 72.70% of native; short-prompt prefill remains faster than native.

Versus `receipts/2026-09-13-jwm1-perf-parity.md` the 1053-token rates are unchanged within 0.4%. This is a same-wheel refresh, not a new kernel result.

A phase profile ran in the same lock using the existing diagnostics venv. It did not require a CTS wait or a rebuild. Prefill GPU work is still dominated by `QmmPrefillCoopmatF16`.

## Identity

- Receipt checkout: `b28deb1cbf7899cab71df08d432dd47cb31862ce` (local `main`; working tree dirty from unrelated sibling work, not used for the measurement).
- Measured source checkout: `b41e2b74c330f910b24cab0e7516e306527858f0`.
- Installed wheel: `mlx-omarchy==0.32.2.dev202609122355+b41e2b74`.
- Wheel SHA-256: `81743cd1a631f6c5d8aa7ff9538d1ec4ddcb2a57c184da7c55345e48d349a240`.
- Python: `3.14.7`; mlx-lm: `0.31.3`.
- Host: `jwm1-linux`; `aarch64`; `nproc=8`; Apple M1 T8103, `Apple M1 (G13G B1)`.
- Kernel: `7.1.6-1-1-ARCH`.
- Vulkan: Honeykrisp, `Mesa 26.3.0-devel (git-6f6afc8968)`.
- Runtime device: `Device(gpu, 0)`.
- Agent model: `openai-codex/gpt-5.6-sol`; fallback: false.
- Model: `mlx-community/Qwen2.5-0.5B-Instruct-4bit` snapshot at `/home/joshuawarren/models/Qwen2.5-0.5B-Instruct-4bit-mlx`.
- Model config SHA-256: `b045e57ea90b8f1b35f89f954b176a5c1faa02bd0af2c89bcec191239d66cef4`.
- Safetensors index SHA-256: `54001cb4c11197119c206dde28e7be08e5872aab6c6d271aed339ec77e84f870`.
- Quantization: affine 4-bit, group size 64; hidden size 896, 24 layers.
- `scripts/bench_decode.py` SHA-256: `f5062d88f34b0845c1e59b0b35d2e33ae02180f636a0f65c543a4366ec2bef7f`.
- `scripts/bench_matrix.json` SHA-256: `df8eb9f3ed84182604379b7cde070ade706bfe590fbf83a35b1877cfaf43f258`.

Power during the locked run:

```text
/sys/class/power_supply/macsmc-ac/online=1
/sys/class/power_supply/tps6598x-source-psy-0-0038/online=1
/sys/class/power_supply/tps6598x-source-psy-0-003f/online=0
/sys/class/power_supply/macsmc-battery/status=Full
```

## Protocol

Same 1053-token Q4 decode/prefill protocol as `receipts/2026-09-13-jwm1-perf-parity.md`. Interpreter:

```text
/home/joshuawarren/src/mlx-bf16-grouped-base-b41e2b74/.work/venv-run/bin/python
```

One outer lock for the whole refresh:

```sh
timeout -k 10s 900s flock -w 60 /tmp/m1-gpu.lock \
  /home/joshuawarren/src/mlx-bf16-grouped-base-b41e2b74/.work/venv-run/bin/python \
  /tmp/jwm1-gpu-parity-refresh-run.py
```

Each decode leg was a fresh subprocess with:

```text
MLX_DISABLE_COMPILE=1
HF_HUB_OFFLINE=1
--tokens 32 --temp 0.0 --seed 0 --warmup-tokens 4
--wheel .../mlx_omarchy-0.32.2.dev202609122355+b41e2b74-cp314-cp314-linux_aarch64.whl
```

`bench_matrix.prompt_text` expanded `short` (2 UTF-8 bytes) and `ctx1024` (4759 UTF-8 bytes). `bench_decode.py` asserted 30 and 1053 chat-template tokens. Decode rate is over 31 inter-token gaps with EOS suppressed.

Provenance on both legs:

```text
provenance: mlx-omarchy 0.32.2.dev202609122355+b41e2b74 mx=0.32.2.dev202609122355+b41e2b74 verified=match harness=b41e2b74 core.cpython-314-aarch64-linux-gnu.so=sha256:4ad850da16c300ea libmlx.so=sha256:03b3f4b9024927f9
```

Native divisor remains `native-2026-09-06-summary.json` as selected in the 2026-09-13 receipt: Q4 short 150.57 decode / 294.10 prefill tok/s; Q4 1053-token 140.38 decode / 1840.90 prefill tok/s.

## Results

A positive delta means Linux is faster.

| Metric | jwm1 Linux | Pinned native base-M1 | Linux/native | Delta | Native/Linux | vs 2026-09-13 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q4 short decode, 30/32 | 109.4675 tok/s | 150.57 tok/s | 72.7021% | -27.298% | 1.3755x | 107.3405 |
| Q4 short prefill, 30 tokens | 333.6863 tok/s | 294.10 tok/s | 113.4601% | +13.460% | 0.8814x | 322.9687 |
| Q4 1K-context decode, 1053/32 | 96.2046 tok/s | 140.38 tok/s | 68.5316% | -31.468% | 1.4592x | 96.5263 |
| Q4 1K-context prefill, 1053 tokens | 1111.5801 tok/s | 1840.90 tok/s | 60.3824% | -39.618% | 1.6561x | 1112.2049 |

Pinned generated-ID digests matched:

```text
short:   7fd25a869ff21678, n=32
1053:    7da83f06ec9f001d, n=32
```

Wall clock for the locked script was 8 s (`started_unix=1789391785`, `finished_unix=1789391793`). Short leg 2.041 s; 1053 leg 3.767 s.

## Phase profile

No 2h CTS fight. Diagnostics wheel already present:

```text
/var/tmp/mlx-omarchy-profile-enabled-b41e2b74/venv/bin/python
mlx-omarchy==0.32.2.dev202609122355+diag.b41e2b74
libmlx.so contains MLX_OMARCHY_GPU_PROFILE
```

Command (same lock, after the uninstrumented legs):

```text
MLX_DISABLE_COMPILE=1
MESA_SHADER_CACHE_DISABLE=true
MLX_OMARCHY_GPU_PROFILE=/tmp/jwm1-gpu-parity-refresh-profile.jsonl
profile_generate.py --model Qwen2.5-0.5B-Instruct-4bit-mlx
  --prompt <bench_matrix ctx1024> --max-tokens 2 --temp 0 --seed 0
```

Generate rc=0, text `Thank you`, `profile.jsonl` 329396 bytes. Analyzer rc=0.

Host markers:

| Phase | ms |
| --- | ---: |
| load | 383.770 |
| prefill | 1204.479 |
| decode (2 tokens) | 23.509 |
| first inter-token | 18.208 |

Instrumented host prefill rate: 1053 / 1.204479 s = 874.24 tok/s. That is not the parity number; the uninstrumented 1111.5801 tok/s is.

Whole-run GPU: 1312 dispatches, 12 submissions, busy 942.136 ms / span 1084.801 ms = **86.85%**. Prefill-window: 1063 dispatches, 927.471 ms GPU busy.

| Kernel | n | gpu ms | share of whole-run busy |
| --- | ---: | ---: | ---: |
| QmmPrefillCoopmatF16 | 163 | 702.034 | 74.5% |
| MatmulRbF16 | 46 | 106.812 | 11.3% |
| QmmVecQ4MultiSubgroupF16 | 288 | 28.244 | 3.0% |
| SoftmaxF16 | 23 | 27.473 | 2.9% |
| SwigluF16 | 95 | 22.099 | 2.3% |

`QmmPrefillCoopmatF16` is 702.034 / 927.471 = 75.69% of prefill-window GPU busy, matching the 2026-09-13 profile-enabled table (163 / 702.559 ms / 75.17%).

## Lock and hardware safety

- Lock inode 29 held exclusively; nested `flock -n /tmp/m1-gpu.lock -c true` returned 1.
- Lock was not unlinked.
- After the run: `flock -n` free, `fuser` rc 1, no leftover decode/profile/`deqp` process.
- `ane.ko` remained loaded. No ANE command, unload, reboot, 1x896, or SET write.
- Grouped-base `dist/` wheel and `.work/venv-run` were not modified.
