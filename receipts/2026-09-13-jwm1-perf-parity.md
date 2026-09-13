# jwm1 base-M1 GPU performance parity against macOS Metal

Date: 2026-09-13

## Verdict

The current installed mlx-omarchy wheel on `jwm1-linux` reaches 71.29% of native macOS decode throughput on the 30/32-token Q4 leg and 68.76% on the 1053/32-token leg. Short-prompt prefill is 9.82% faster than the pinned native result. The largest referenced gap is 1053-token prefill: 1112.2049 tok/s on Linux versus 1840.90 tok/s on native Metal, or 60.42% of native and a 39.58% deficit.

The matching phase-isolated 1053-token Q4 profile already in the repository points first to shader throughput: GPU work occupies 77.88% of the profiled GPU span, and `QmmPrefillCoopmatF16` alone accounts for 74.49% of GPU-busy time. Submission gaps remain material but are not the largest measured component. This is a suspected stage, not a new current-wheel profile result. A fresh `b41e2b74` profile was attempted but emitted no profile data because another authorized CTS verification took `/tmp/m1-gpu.lock`; it is deferred until that workload releases the lock.

The 2048-square fp16 and bf16 matmul+add medians are valid Linux measurements but **UNREFERENCED** for macOS parity. No receipt in the inspected history contains the same operation, shape, seed, 3-warmup/20-measured timing window, and base-M1 native Metal host. The same-protocol `receipts/2026-09-13-jwm1-jw16-gpu-parity.md` is Linux-to-Linux and is not used as a macOS divisor.

## Identity

- Receipt checkout: `f7169fcb7c2df1713ee635ec5cdc57462292d90e`.
- Measured source checkout: `b41e2b74c330f910b24cab0e7516e306527858f0`.
- Installed wheel: `mlx-omarchy==0.32.2.dev202609122355+b41e2b74`.
- Wheel file: `mlx_omarchy-0.32.2.dev202609122355+b41e2b74-cp314-cp314-linux_aarch64.whl`, SHA-256 `81743cd1a631f6c5d8aa7ff9538d1ec4ddcb2a57c184da7c55345e48d349a240`.
- Python: `3.14.7`; mlx-lm: `0.31.3`.
- Host: `jwm1-linux`; architecture: `aarch64`; chip/GPU: Apple M1 T8103, `Apple M1 (G13G B1)`.
- Kernel: `7.1.6-1-1-ARCH`.
- Vulkan driver: Honeykrisp, `Mesa 26.3.0-devel (git-6f6afc8968)`.
- Runtime device: `Device(gpu, 0)`; every measured tensor operation explicitly selected `mx.gpu`.
- Agent model: `openai-codex/gpt-5.6-sol`; fallback: false.
- Model: `mlx-community/Qwen2.5-0.5B-Instruct-4bit` at revision `a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3`.
- Model config SHA-256: `b045e57ea90b8f1b35f89f954b176a5c1faa02bd0af2c89bcec191239d66cef4`.
- Safetensors index SHA-256: `54001cb4c11197119c206dde28e7be08e5872aab6c6d271aed339ec77e84f870`.
- Quantization: affine 4-bit, group size 64; hidden size 896, 24 layers.
- `scripts/bench_decode.py` SHA-256: `f5062d88f34b0845c1e59b0b35d2e33ae02180f636a0f65c543a4366ec2bef7f`.
- `scripts/bench_matrix.json` SHA-256: `df8eb9f3ed84182604379b7cde070ade706bfe590fbf83a35b1877cfaf43f258`.

Power supply status captured before and during measurement:

```text
/sys/class/power_supply/macsmc-ac/online=1
/sys/class/power_supply/tps6598x-source-psy-0-0038/online=1
/sys/class/power_supply/tps6598x-source-psy-0-003f/online=0
/sys/class/power_supply/macsmc-battery/status=Full
```

This establishes AC power rather than assuming it.

## macOS baseline inventory and selection

The inspected baseline checkout was `1242cb32358969b38d599136addc674d7bd47563`.

| Receipt | What exists | Comparability decision |
| --- | --- | --- |
| `1242cb32:receipts/2026-09-10-native-macos-metal-baseline/README.md` with base-M1 source `ea09eefa:receipts/native-baseline-2026-09-06/native-2026-09-06-summary.json` | Five-repeat base Apple M1 native Metal matrix. Q4 short: 150.57 decode / 294.10 prefill tok/s. Q4 1053-token: 140.38 decode / 1840.90 prefill tok/s. Same model revision and generated-ID digests. | **REFERENCE.** This is the current authoritative same-chip index and it specifies the same `bench_decode.py` protocol: `MLX_DISABLE_COMPILE=1`, greedy temp 0, seed 0, EOS suppressed, 32 requested tokens, 4 warmup tokens, decode over 31 inter-token gaps, prompts of 30 and 1053 tokens. |
| `ea09eefa:receipts/native-baseline-2026-09-06/native-2026-09-06-nocompile-summary.json` | Sibling base-M1 five-repeat summary: 150.84 / 294.1 tok/s short and 140.25 / 1837.7 tok/s at 1053 tokens. | Same-protocol supporting evidence, but not selected as divisor because the later authoritative index explicitly pins `native-2026-09-06-summary.json`. This avoids choosing between two near-identical native summaries after seeing the Linux result. |
| `receipts/2026-09-01-m1-same-chip-parity.md` | Base-M1 native Metal, but the old CLI stopped at EOS after 2 or 10 generated tokens and did not include the current 1053-token leg. | **UNREFERENCED** for the current pinned-length decode metrics. |
| `receipts/2026-09-01-macos-native-mlx-baseline.md` | Native Metal on an M1 Max, also EOS-truncated. | **UNREFERENCED**: wrong chip and wrong decode protocol. |
| `1242cb32:receipts/2026-09-10-native-macos-metal-baseline/` M1 Max and M1 Ultra matrices | Same decode protocol on larger chips. | Cross-chip context only. The receipt itself prohibits using these numbers as a base-M1 parity divisor. |
| All inspected macOS receipts | No base-M1 Metal 2048x2048 fp16 or bf16 `matmul(a,b)+bias` with seed 20260913, 3 warmups, and 20 measured synchronized evaluations. | Both new matmul metrics are **UNREFERENCED**. |

The v0.4.0 release values 78.60/100.3 tok/s for 30/32 and 44.95/272.9 tok/s for 1053/32 are Linux Honeykrisp release measurements, not macOS baselines. They are not used below.

## Protocol

All GPU runs used the installed interpreter at:

```text
/home/joshuawarren/src/mlx-bf16-grouped-base-b41e2b74/.work/venv-run/bin/python
```

Each completed command was bounded and serialized by the stable host lock:

```sh
cat <<'PY' | ssh -o ConnectTimeout=8 -o BatchMode=yes jwm1-linux \
  'timeout -k 10s 300s flock -w 60 /tmp/m1-gpu.lock /home/joshuawarren/src/mlx-bf16-grouped-base-b41e2b74/.work/venv-run/bin/python -'
# Matmul benchmark source described below.
PY

cat <<'PY' | ssh -o ConnectTimeout=8 -o BatchMode=yes jwm1-linux \
  'timeout -k 10s 600s flock -w 60 /tmp/m1-gpu.lock /home/joshuawarren/src/mlx-bf16-grouped-base-b41e2b74/.work/venv-run/bin/python -'
# Decode wrapper described below.
PY
```

Both processes recorded lock inode 29. A competing `flock -n /tmp/m1-gpu.lock -c true` executed from inside each locked process returned nonzero, proving exclusion during measurement.

### Matmul

For each dtype, the benchmark:

1. selected `mx.gpu`;
2. evaluated a valid 2x2 `matmul+add` and required exact float32-readback equality with `[[20.0, 21.0], [43.5, 49.5]]`;
3. reset `mx.random.seed(20260913)`;
4. created and evaluated 2048x2048 `a`, `b`, and `bias` tensors in the named dtype;
5. timed `mx.matmul(a, b) + bias` through `mx.eval(result)` with `time.perf_counter_ns()`;
6. discarded three warmups and retained twenty samples;
7. reported the sample median and `(2*N^3+N^2)/time` TFLOPS.

The only difference between the two legs was `mx.float16` versus `mx.bfloat16`.

### Decode and prefill

The wrapper expanded `short` and `ctx1024` with the checked-in `bench_matrix.prompt_text`; the prompt sizes were 2 and 4759 UTF-8 bytes before the chat template. `bench_decode.py` then asserted 30 and 1053 chat-template tokens. Each leg ran in a fresh subprocess with:

```text
MLX_DISABLE_COMPILE=1
--tokens 32 --temp 0.0 --seed 0 --warmup-tokens 4
--wheel /home/joshuawarren/src/mlx-bf16-grouped-base-b41e2b74/dist/mlx_omarchy-0.32.2.dev202609122355+b41e2b74-cp314-cp314-linux_aarch64.whl
```

The harness verified the loaded core extension and `libmlx.so` against that wheel before emitting rates:

```text
provenance: mlx-omarchy 0.32.2.dev202609122355+b41e2b74 mx=0.32.2.dev202609122355+b41e2b74 verified=match harness=b41e2b74 core.cpython-314-aarch64-linux-gnu.so=sha256:4ad850da16c300ea libmlx.so=sha256:03b3f4b9024927f9
```

## Results

A positive delta means Linux is faster. A negative delta is the remaining Linux deficit.

| Metric | jwm1 Linux | Pinned native base-M1 | Linux/native | Delta | Native/Linux factor | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Q4 short decode, 30/32 | 107.3405 tok/s | 150.57 tok/s | 71.2894% | -28.711% | 1.4027x | **REFERENCE** |
| Q4 short prefill, 30 tokens | 322.9687 tok/s | 294.10 tok/s | 109.8159% | +9.816% | 0.9106x | **REFERENCE** |
| Q4 1K-context decode, 1053/32 | 96.5263 tok/s | 140.38 tok/s | 68.7607% | -31.239% | 1.4543x | **REFERENCE** |
| Q4 1K-context prefill, 1053 tokens | 1112.2049 tok/s | 1840.90 tok/s | 60.4164% | -39.584% | 1.6552x | **REFERENCE** |
| fp16 2048x2048 matmul+add median | 31.6631015 ms, 0.5427157377 TFLOPS | none | — | — | — | **UNREFERENCED** |
| bf16 2048x2048 matmul+add median | 26.9131140 ms, 0.6385014937 TFLOPS | none | — | — | — | **UNREFERENCED** |

Both Q4 legs produced the pinned native generated-ID digests:

```text
short:   7fd25a869ff21678, n=32
1053:    7da83f06ec9f001d, n=32
```

### Matmul raw samples, milliseconds

```text
fp16 warmup: 35.716123, 31.671636, 31.686427
fp16 measured: 31.589933, 31.605645, 31.606437, 31.651269, 31.694058,
  31.670100, 31.725472, 31.677267, 31.653393, 31.660935,
  31.665518, 31.633936, 31.616520, 31.665268, 31.611187,
  31.688516, 31.675350, 31.666017, 31.654977, 31.701724
fp16 median: 31.6631015

bf16 warmup: 30.721194, 26.964608, 26.881612
bf16 measured: 26.833865, 26.924485, 26.909112, 26.932402, 26.873738,
  26.933401, 26.886029, 26.858947, 26.920944, 26.930989,
  26.869785, 26.885117, 26.897700, 26.885368, 26.917116,
  26.962363, 26.938406, 26.962155, 26.931323, 26.888242
bf16 median: 26.9131140
```

## Stage attribution

The matching existing phase-isolated Q4 m1053 profile is:

```text
1242cb32:receipts/2026-09-10-prefill-qmm-isa/distribution/m1053/prefill-summary.json
```

It records 1231 dispatches in 53 submissions, 943.089041 ms GPU busy over a 1210.941708 ms GPU span, and a 0.778806 busy fraction. `QmmPrefillCoopmatF16` uses 702.476299 ms, or 74.4867% of GPU-busy time. `MatmulRbF16` is second at 109.418831 ms, or 11.6022%. Inter-submission gaps total 239.993205 ms and intra-submission gaps total 27.859462 ms.

Therefore the largest current parity gap, long-context prefill, is suspected to be primarily shader-throughput limited, concentrated in Q4 prefill matrix work. Dispatch and submission overhead is still a secondary target. The old `receipts/2026-09-02-gpu-profile-decode.md` is not used for this conclusion because it predates the current wheel and measured the old one-core-host decode structure.

A fresh current-wheel profile command was submitted with `MLX_DISABLE_COMPILE=1`, the same 1053-token prompt, four output tokens, `MLX_OMARCHY_GPU_PROFILE`, host markers, and the same lock. It produced no profile result. The immediate lock-owner inspection found the authorized Mesa Dj CTS run holding inode 29 (`deqp-vk`, then `cmshape2`); parent coordination explicitly prohibited interruption. A fresh `b41e2b74` phase profile is the remaining attribution step after that lock becomes free.

## Lock and hardware safety

- Completed matmul and decode measurements each proved lock ownership by rejecting a nested nonblocking lock attempt.
- No benchmark process remained after either completed command.
- Final inspection found the stable lock inode still present and busy only because the separate Mesa Dj CTS verification had progressed to PIDs 4059-4061 running `cmshape2` under its own `flock`. No parity-benchmark process held the lock. The lock was not unlinked.
- `ane.ko` was loaded before and after the work. No command unloaded it or accessed ANE.
- No reboot, GRUB action, device-tree change, Mesa edit, router change, or service change was performed.
- The fresh profile was not forced around the lock holder and emitted no numbers.
