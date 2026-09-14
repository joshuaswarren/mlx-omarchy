# jw16 M1 Max GPU Q4 decode/prefill rerun vs native Metal

Date: 2026-09-14

## Verdict

Same-protocol Q4 decode/prefill on `jw16mbp1-linux` (Apple M1 Max, Honeykrisp, GPU not ANE) completed with pinned generated-ID identity. Decode on the 30/32-token leg is 51.32% of native M1 Max Metal (147.2815 vs 287 tok/s). The 1053/32-token decode is 30.79% of native (87.4478 vs 284 tok/s). Long-context prefill is 25.75% of native (2072.0272 vs 8048 tok/s). Short-prompt prefill is 13.16% of native (199.8427 vs 1518 tok/s).

The 2048-square fp16 `matmul(a,b)+bias` median on the same locked run is 8.680269 ms / 1.9796694651 TFLOPS, matching the 2026-09-13 same-protocol jw16 median (8.712577 ms / 1.9723284498 TFLOPS) to 0.37% and the 2026-09-14 refresh median (8.717198 ms / 1.9712829155 TFLOPS) to 0.42%.

This is one locked pair plus the cheap matmul, not a thermally soaked multi-repeat median. `Jw16AneFull` held accel0 only; GPU lock inode 12 was free before this run and was not stolen.

## Identity

- Receipt checkout (local mlx-omarchy HEAD): `b28deb1cbf7899cab71df08d432dd47cb31862ce` (`git describe --always --dirty`: `v0.3.2-4-gb28deb1c-dirty`).
- Measured source commit (installed wheel): `b41e2b74c330f910b24cab0e7516e306527858f0`.
- Remote harness tree: HEAD `3db3cb9a1d6d21f5ea1e002e93a8452b03e2b84f` (`git describe --always --dirty`: `v0.4.2-228-g3db3cb9a`). Engine provenance still reports `harness=3db3cb9a-dirty`.
- Installed interpreter: `/home/joshuawarren/.local/share/mlx-omarchy-test-venv/bin/python` (Python 3.14.7).
- Installed wheel: `mlx-omarchy==0.32.2.dev202609122106+b41e2b74`.
- Wheel file: `mlx_omarchy-0.32.2.dev202609122106+b41e2b74-cp314-cp314-linux_aarch64.whl`, SHA-256 `cb13927379ed9b9bce7be90475347b3d3291906be5411e14d784ccce03357bae`.
- Provenance: `verified=match`; `core.cpython-314-aarch64-linux-gnu.so=sha256:4ad850da16c300ea`; `libmlx.so=sha256:f2d45e601f05dd22`.
- mlx-lm: `0.31.3`.
- Host: `jw16mbp1-linux`; architecture: `aarch64`; chip/GPU: Apple M1 Max T6001, `Apple M1 Max (G13C C0)`.
- Kernel: `7.1.6-1-1-ARCH`.
- Vulkan driver: Honeykrisp, `Mesa 26.2.2-arch1.1` (`mesa 1:26.2.2-1`).
- Runtime device: `Device(gpu, 0)`; both Q4 legs reported `device=Apple M1 Max (G13C C0)`.
- Agent model: `openai-codex/gpt-5.6-sol`; fallback: false.
- Model: `mlx-community/Qwen2.5-0.5B-Instruct-4bit` at revision `a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3`.
- Model config SHA-256: `b045e57ea90b8f1b35f89f954b176a5c1faa02bd0af2c89bcec191239d66cef4`.
- Safetensors index SHA-256: `54001cb4c11197119c206dde28e7be08e5872aab6c6d271aed339ec77e84f870`.
- `scripts/bench_decode.py` SHA-256: `f5062d88f34b0845c1e59b0b35d2e33ae02180f636a0f65c543a4366ec2bef7f`.
- `scripts/bench_matrix.json` SHA-256: `df8eb9f3ed84182604379b7cde070ade706bfe590fbf83a35b1877cfaf43f258`.
- Runner SHA-256: `75ba2716b09ac1ff1f230975cd96cbbab1f53dd20495369f63a12e8616121091`.
- Raw capture SHA-256: `5d7c5970ee2dd41d5736382c1ee3a0455d90e1cc0ce21219479d3e6b2414ec40`.

Power before, after each Q4 leg, and after the pair+matmul:

```text
/sys/class/power_supply/macsmc-ac/online=1
/sys/class/power_supply/macsmc-battery/status=Full
/sys/class/power_supply/tps6598x-source-psy-0-0038/online=0
/sys/class/power_supply/tps6598x-source-psy-0-003a/online=1
/sys/class/power_supply/tps6598x-source-psy-0-003b/online=0
/sys/class/power_supply/tps6598x-source-psy-0-003f/online=0
```

## Protocol

Matches `receipts/2026-09-14-jw16-gpu-parity-refresh.md` and `receipts/2026-09-13-jw16-q4-decode/`: `MLX_DISABLE_COMPILE=1`, greedy temp 0, seed 0, EOS suppressed, 32 requested tokens, 4 warmup tokens, decode over 31 inter-token gaps, `bench_matrix.prompt_text` for `short` (2 UTF-8 bytes, 30 chat-template tokens) and `ctx1024` (4759 UTF-8 bytes, 1053 chat-template tokens). Fresh subprocess per Q4 leg. Outer process held `/tmp/m1-gpu.lock` inode 12; nested `flock -n` exit 1 during measurement.

The cheap 2048² fp16 matmul+add used the same protocol as `receipts/2026-09-13-jwm1-jw16-gpu-parity.md` inside the same flock: `mx.gpu`, exact 2x2 `[[20.0, 21.0], [43.5, 49.5]]`, seed `20260913`, 3 warmups, 20 measured synchronized `matmul(a,b)+bias` evaluations, median, TFLOPS from `(2*N^3+N^2)/elapsed`.

Exact command:

```sh
cat /tmp/jw16-gpu-parity-refresh.py | ssh -o ConnectTimeout=8 -o BatchMode=yes -o ControlMaster=no -o IdentityAgent=none jw16mbp1-linux \
  'timeout -k 10s 900s flock -w 60 /tmp/m1-gpu.lock /home/joshuawarren/.local/share/mlx-omarchy-test-venv/bin/python -'
```

`llm-inference.service` stayed inactive. No llama or leftover python process. ANE, DTS, modules, and llama were not touched. No reboot, ANE unload, `1x896`, or SET 0xf. The lock was released, not unlinked.

Native M1 Max divisors are the assignment roundings of the 2026-09-10 16m1mbp 12-rep Metal matrix (precise medians 286.96 / 283.79 decode and 1517.55 / 8048.42 prefill). jwm1 Linux numbers below are from `receipts/2026-09-13-jwm1-perf-parity.md` and were not remeasured here.

## Results

JSON `decode_tps` / `prefill_tps` are the quoted rates.

| Metric | jw16 Linux M1 Max | native M1 Max Metal | jw16/native Max | 2026-09-14 refresh | jwm1 Linux M1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q4 short decode, 30/32 | 147.2815 tok/s | 287 tok/s | 51.32% | 150.1225 tok/s | 107.3405 tok/s |
| Q4 short prefill, 30 tokens | 199.8427 tok/s | 1518 tok/s | 13.16% | 197.1204 tok/s | 322.9687 tok/s |
| Q4 1K-context decode, 1053/32 | 87.4478 tok/s | 284 tok/s | 30.79% | 87.1002 tok/s | 96.5263 tok/s |
| Q4 1K-context prefill, 1053 tokens | 2072.0272 tok/s | 8048 tok/s | 25.75% | 2067.9814 tok/s | 1112.2049 tok/s |
| fp16 2048² matmul+add median | 8.680269 ms, 1.9796694651 TFLOPS | — | — | 8.717198 ms, 1.9712829155 TFLOPS | 31.589904 ms, 0.5439732735 TFLOPS |

Both Q4 legs produced the pinned native generated-ID digests:

```text
short:   7fd25a869ff21678, n=32, first=9707,0,2585 last=646,387,7881
1053:    7da83f06ec9f001d, n=32, first=13060,498,369 last=3897,553,279
```

The 2x2 fp16 matmul+add exact check passed: `[[20.0, 21.0], [43.5, 49.5]]`.

### Raw bench_decode stdout

Short:

```text
provenance: mlx-omarchy 0.32.2.dev202609122106+b41e2b74 mx=0.32.2.dev202609122106+b41e2b74 verified=match harness=3db3cb9a-dirty core.cpython-314-aarch64-linux-gnu.so=sha256:4ad850da16c300ea libmlx.so=sha256:f2d45e601f05dd22
decode 147.28 tok/s over 31 tokens (32 requested, EOS suppressed)
prefill 0.150s (reported separately, excluded from decode)
prompt_tokens 30
decode mean per-token 6.8 ms
generated_ids sha256:7fd25a869ff21678 n=32 first=9707,0,2585 last=646,387,7881
{"decode_tps": 147.2815, "device": "Apple M1 Max (G13C C0)", "engine": "bench_decode", "generated": 32, "ids_first": [9707, 0, 2585], "ids_last": [646, 387, 7881], "ids_sha256_16": "7fd25a869ff21678", "prefill_s": 0.150118, "prefill_tps": 199.8427, "prompt_tokens": 30}
```

1053:

```text
provenance: mlx-omarchy 0.32.2.dev202609122106+b41e2b74 mx=0.32.2.dev202609122106+b41e2b74 verified=match harness=3db3cb9a-dirty core.cpython-314-aarch64-linux-gnu.so=sha256:4ad850da16c300ea libmlx.so=sha256:f2d45e601f05dd22
decode 87.45 tok/s over 31 tokens (32 requested, EOS suppressed)
prefill 0.508s (reported separately, excluded from decode)
prompt_tokens 1053
decode mean per-token 11.4 ms
generated_ids sha256:7da83f06ec9f001d n=32 first=13060,498,369 last=3897,553,279
{"decode_tps": 87.4478, "device": "Apple M1 Max (G13C C0)", "engine": "bench_decode", "generated": 32, "ids_first": [13060, 498, 369], "ids_last": [3897, 553, 279], "ids_sha256_16": "7da83f06ec9f001d", "prefill_s": 0.508198, "prefill_tps": 2072.0272, "prompt_tokens": 1053}
```

### Matmul samples

Warmups, ms: `11.248525, 8.64006, 8.652478`

Measured, ms: `8.21206, 8.217768, 8.293143, 8.692144, 8.867978, 8.637853, 8.787061, 8.714102, 8.808811, 8.82702, 8.601227, 8.340435, 8.659186, 8.599269, 8.273643, 8.722436, 8.678644, 8.681894, 8.724061, 8.824728`

Median: `8.680269` ms. TFLOPS: `1.979669465082246`.

## Lock and hardware safety

- Lock inode 12; nested nonblocking flock rejected (`exit 1`) while the pair ran.
- After the pair: `flock -n /tmp/m1-gpu.lock` succeeded; inode 12 still present; no python or llama process; `llm-inference.service` inactive.
- No ANE access, module load/unload, device-tree change, llama enable/disable, SET 0xf, `1x896`, or reboot.
- The lock was not stolen (free before the run; `Jw16AneFull` acked GPU unused) and was not unlinked.
