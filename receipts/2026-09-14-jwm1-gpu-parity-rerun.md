# jwm1 GPU Q4 parity rerun

Date: 2026-09-14

## Verdict

`/tmp/m1-gpu.lock` inode 29 was free. No CTS/`deqp`/`cmshape2` holder. The lock was taken with `flock -w 60` (not stolen) and released. Nested `flock -n` returned 1 while held.

The installed `b41e2b74` wheel on `jwm1-linux` sits at 67.70% of native macOS decode on the 1053/32 Q4 leg and 60.38% of native 1053-token prefill. Short-prompt decode is 71.25% of native; short-prompt prefill remains faster than native.

Versus `receipts/2026-09-14-jwm1-gpu-parity-refresh.md` the 1053-token prefill rate is unchanged (1111.5836 vs 1111.5801 tok/s). 1053-token decode is 1.22% lower (95.0352 vs 96.2046 tok/s). This is a same-wheel rerun under live `ane.ko`, not a new kernel result. No phase profile was taken.

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
- Runner SHA-256: `24efa9672cade615383d535cafede0046bf5397b362596baa498539bd05a227c`.
- Raw capture SHA-256: `7c8256fbc2e1819964aa432d471deab07575b2b30fad42c9ac4aa34208b5d377`.

Power during the locked run and after release:

```text
/sys/class/power_supply/macsmc-ac/online=1
/sys/class/power_supply/tps6598x-source-psy-0-0038/online=1
/sys/class/power_supply/tps6598x-source-psy-0-003f/online=0
/sys/class/power_supply/macsmc-battery/status=Full
```

## Protocol

Same 30/32 and 1053/32 Q4 decode/prefill protocol as `receipts/2026-09-14-jwm1-gpu-parity-refresh.md`. Interpreter:

```text
/home/joshuawarren/src/mlx-bf16-grouped-base-b41e2b74/.work/venv-run/bin/python
```

One outer lock for both legs:

```sh
timeout -k 10s 900s flock -w 60 /tmp/m1-gpu.lock \
  /home/joshuawarren/src/mlx-bf16-grouped-base-b41e2b74/.work/venv-run/bin/python \
  /tmp/jwm1-gpu-parity-rerun-run.py
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

No diagnostics profile, matmul, ANE unload, reboot, 1x896, or SET write.

## Results

A positive delta means Linux is faster.

| Metric | jwm1 Linux | Pinned native base-M1 | Linux/native | Delta | Native/Linux | vs 2026-09-14 refresh |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q4 short decode, 30/32 | 107.285 tok/s | 150.57 tok/s | 71.2526% | -28.747% | 1.4035x | 109.4675 |
| Q4 short prefill, 30 tokens | 331.0513 tok/s | 294.10 tok/s | 112.5642% | +12.564% | 0.8884x | 333.6863 |
| Q4 1K-context decode, 1053/32 | 95.0352 tok/s | 140.38 tok/s | 67.6985% | -32.301% | 1.4772x | 96.2046 |
| Q4 1K-context prefill, 1053 tokens | 1111.5836 tok/s | 1840.90 tok/s | 60.3826% | -39.617% | 1.6561x | 1111.5801 |

Pinned generated-ID digests matched:

```text
short:   7fd25a869ff21678, n=32, first=9707,0,2585 last=646,387,7881
1053:    7da83f06ec9f001d, n=32, first=13060,498,369 last=3897,553,279
```

Wall clock for the locked script was 6 s (`started_unix=1789392863`, `finished_unix=1789392869`). Short leg 2.002 s; 1053 leg 3.791 s.

### Raw bench_decode stdout

Short:

```text
provenance: mlx-omarchy 0.32.2.dev202609122355+b41e2b74 mx=0.32.2.dev202609122355+b41e2b74 verified=match harness=b41e2b74 core.cpython-314-aarch64-linux-gnu.so=sha256:4ad850da16c300ea libmlx.so=sha256:03b3f4b9024927f9
decode 107.29 tok/s over 31 tokens (32 requested, EOS suppressed)
prefill 0.091s (reported separately, excluded from decode)
prompt_tokens 30
decode mean per-token 9.3 ms
generated_ids sha256:7fd25a869ff21678 n=32 first=9707,0,2585 last=646,387,7881
{"decode_tps": 107.285, "device": "Apple M1 (G13G B1)", "engine": "bench_decode", "generated": 32, "ids_first": [9707, 0, 2585], "ids_last": [646, 387, 7881], "ids_sha256_16": "7fd25a869ff21678", "prefill_s": 0.09062, "prefill_tps": 331.0513, "prompt_tokens": 30}
```

1053:

```text
provenance: mlx-omarchy 0.32.2.dev202609122355+b41e2b74 mx=0.32.2.dev202609122355+b41e2b74 verified=match harness=b41e2b74 core.cpython-314-aarch64-linux-gnu.so=sha256:4ad850da16c300ea libmlx.so=sha256:03b3f4b9024927f9
decode 95.04 tok/s over 31 tokens (32 requested, EOS suppressed)
prefill 0.947s (reported separately, excluded from decode)
prompt_tokens 1053
decode mean per-token 10.5 ms
generated_ids sha256:7da83f06ec9f001d n=32 first=13060,498,369 last=3897,553,279
{"decode_tps": 95.0352, "device": "Apple M1 (G13G B1)", "engine": "bench_decode", "generated": 32, "ids_first": [13060, 498, 369], "ids_last": [3897, 553, 279], "ids_sha256_16": "7da83f06ec9f001d", "prefill_s": 0.947297, "prefill_tps": 1111.5836, "prompt_tokens": 1053}
```

## Lock and hardware safety

- Lock inode 29 held exclusively; nested `flock -n /tmp/m1-gpu.lock -c true` returned 1.
- Lock was not unlinked.
- After the run: `flock -n` free, `fuser` empty, no leftover decode/profile/`deqp` process.
- `ane.ko` remained loaded (`ane 65536 0 - Live`). No ANE command, unload, reboot, 1x896, or SET write.
- Grouped-base `dist/` wheel and `.work/venv-run` were not modified.
