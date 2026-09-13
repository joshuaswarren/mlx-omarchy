# jw16 M1 Max Q4 decode/prefill vs jwm1 and native Metal

Date: 2026-09-13

## Verdict

Same-protocol Q4 decode/prefill on `jw16mbp1-linux` (Apple M1 Max, Honeykrisp) completed with pinned generated-ID identity. Decode on the 30/32-token leg is 1.3740x jwm1 Linux and 51.40% of native M1 Max Metal. The 1053/32-token decode is 0.8943x jwm1 Linux and 30.42% of native M1 Max Metal. Long-context prefill is 1.8336x jwm1 Linux and 25.34% of native M1 Max Metal. Short-prompt prefill is slower than jwm1 Linux (0.5292x) and 11.26% of native M1 Max Metal.

This is one locked pair of legs, not a thermally soaked multi-repeat median. No second GPU run was taken after parent instruction to keep the fabric quiet for ANE cycle 8.

## Identity

- Receipt checkout: `b28deb1cbf7899cab71df08d432dd47cb31862ce`.
- Measured source commit (installed wheel): `b41e2b74c330f910b24cab0e7516e306527858f0`.
- Remote harness tree reported by provenance: `3db3cb9a-dirty`. Engine bytes still match the jwm1 receipt.
- Installed interpreter: `/home/joshuawarren/.local/share/mlx-omarchy-test-venv/bin/python` (Python 3.14.7).
- Installed wheel: `mlx-omarchy==0.32.2.dev202609122106+b41e2b74`.
- Wheel file: `mlx_omarchy-0.32.2.dev202609122106+b41e2b74-cp314-cp314-linux_aarch64.whl`, SHA-256 `cb13927379ed9b9bce7be90475347b3d3291906be5411e14d784ccce03357bae`.
- Provenance: `verified=match`; `core.cpython-314-aarch64-linux-gnu.so=sha256:4ad850da16c300ea`; `libmlx.so=sha256:f2d45e601f05dd22`.
- mlx-lm: `0.31.3`.
- Host: `jw16mbp1-linux`; architecture: `aarch64`; chip/GPU: Apple M1 Max T6001, `Apple M1 Max (G13C C0)`.
- Kernel: `7.1.6-1-1-ARCH`.
- Vulkan driver: Honeykrisp, `Mesa 26.2.2-arch1.1` (`mesa 1:26.2.2-1`).
- Runtime device: `Device(gpu, 0)`; both legs reported `device=Apple M1 Max (G13C C0)`.
- Agent model: `openai-codex/gpt-5.6-sol`; fallback: false.
- Model: `mlx-community/Qwen2.5-0.5B-Instruct-4bit` at revision `a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3`.
- Model config SHA-256: `b045e57ea90b8f1b35f89f954b176a5c1faa02bd0af2c89bcec191239d66cef4`.
- Safetensors index SHA-256: `54001cb4c11197119c206dde28e7be08e5872aab6c6d271aed339ec77e84f870`.
- `scripts/bench_decode.py` SHA-256: `f5062d88f34b0845c1e59b0b35d2e33ae02180f636a0f65c543a4366ec2bef7f`.
- `scripts/bench_matrix.json` SHA-256: `df8eb9f3ed84182604379b7cde070ade706bfe590fbf83a35b1877cfaf43f258`.
- Raw capture: `run.json` SHA-256 `9890ad3756fdbd320fecda28846338e00ba09f4536805faeaff867a166fa3f9a`.

Power before, after each leg, and after the pair:

```text
/sys/class/power_supply/macsmc-ac/online=1
/sys/class/power_supply/macsmc-battery/status=Full
/sys/class/power_supply/tps6598x-source-psy-0-0038/online=0
/sys/class/power_supply/tps6598x-source-psy-0-003a/online=1
/sys/class/power_supply/tps6598x-source-psy-0-003b/online=0
/sys/class/power_supply/tps6598x-source-psy-0-003f/online=0
```

## Protocol

Matches `receipts/2026-09-13-jwm1-perf-parity.md`: `MLX_DISABLE_COMPILE=1`, greedy temp 0, seed 0, EOS suppressed, 32 requested tokens, 4 warmup tokens, decode over 31 inter-token gaps, `bench_matrix.prompt_text` for `short` (2 UTF-8 bytes, 30 chat-template tokens) and `ctx1024` (4759 UTF-8 bytes, 1053 chat-template tokens). Fresh subprocess per leg. Outer process held `/tmp/m1-gpu.lock` inode 12; nested `flock -n` exit 1 during measurement.

```sh
cat <<'PY' | ssh -o ConnectTimeout=8 -o BatchMode=yes -o ControlMaster=no -o IdentityAgent=none jw16mbp1-linux \
  'timeout -k 10s 900s flock -w 60 /tmp/m1-gpu.lock /home/joshuawarren/.local/share/mlx-omarchy-test-venv/bin/python -'
# expand short/ctx1024 from bench_matrix.json; subprocess bench_decode.py
# --tokens 32 --temp 0.0 --seed 0 --warmup-tokens 4 --wheel <jw16 wheel>
PY
```

`llm-inference.service` stayed enabled and inactive. No llama process. ANE, DTS, modules, and llama were not touched. No reboot. The lock was released, not unlinked. No further GPU work after this pair.

## Results

JSON `decode_tps` / `prefill_tps` are the quoted rates. A positive jw16/jwm1 ratio means jw16 is faster.

| Metric | jw16 Linux M1 Max | jwm1 Linux M1 | native M1 Max Metal | jw16/jwm1 | jw16/native Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q4 short decode, 30/32 | 147.4849 tok/s | 107.3405 tok/s | 286.96 tok/s | 1.3740x | 51.40% |
| Q4 short prefill, 30 tokens | 170.9227 tok/s | 322.9687 tok/s | 1517.55 tok/s | 0.5292x | 11.26% |
| Q4 1K-context decode, 1053/32 | 86.3228 tok/s | 96.5263 tok/s | 283.79 tok/s | 0.8943x | 30.42% |
| Q4 1K-context prefill, 1053 tokens | 2039.498 tok/s | 1112.2049 tok/s | 8048.42 tok/s | 1.8336x | 25.34% |

jwm1 numbers: `receipts/2026-09-13-jwm1-perf-parity.md`. Native M1 Max medians: `2026-09-10-native-macos-metal-baseline` 16m1mbp 12-rep matrix. Native base-M1 (same protocol, not this host): 150.57 / 294.10 and 140.38 / 1840.90.

Both legs produced the pinned native generated-ID digests:

```text
short:   7fd25a869ff21678, n=32, first=9707,0,2585 last=646,387,7881
1053:    7da83f06ec9f001d, n=32, first=13060,498,369 last=3897,553,279
```

### Raw bench_decode stdout

Short:

```text
provenance: mlx-omarchy 0.32.2.dev202609122106+b41e2b74 mx=0.32.2.dev202609122106+b41e2b74 verified=match harness=3db3cb9a-dirty core.cpython-314-aarch64-linux-gnu.so=sha256:4ad850da16c300ea libmlx.so=sha256:f2d45e601f05dd22
decode 147.48 tok/s over 31 tokens (32 requested, EOS suppressed)
prefill 0.176s (reported separately, excluded from decode)
prompt_tokens 30
decode mean per-token 6.8 ms
generated_ids sha256:7fd25a869ff21678 n=32 first=9707,0,2585 last=646,387,7881
{"decode_tps": 147.4849, "device": "Apple M1 Max (G13C C0)", "engine": "bench_decode", "generated": 32, "ids_first": [9707, 0, 2585], "ids_last": [646, 387, 7881], "ids_sha256_16": "7fd25a869ff21678", "prefill_s": 0.175518, "prefill_tps": 170.9227, "prompt_tokens": 30}
```

1053:

```text
provenance: mlx-omarchy 0.32.2.dev202609122106+b41e2b74 mx=0.32.2.dev202609122106+b41e2b74 verified=match harness=3db3cb9a-dirty core.cpython-314-aarch64-linux-gnu.so=sha256:4ad850da16c300ea libmlx.so=sha256:f2d45e601f05dd22
decode 86.32 tok/s over 31 tokens (32 requested, EOS suppressed)
prefill 0.516s (reported separately, excluded from decode)
prompt_tokens 1053
decode mean per-token 11.6 ms
generated_ids sha256:7da83f06ec9f001d n=32 first=13060,498,369 last=3897,553,279
{"decode_tps": 86.3228, "device": "Apple M1 Max (G13C C0)", "engine": "bench_decode", "generated": 32, "ids_first": [13060, 498, 369], "ids_last": [3897, 553, 279], "ids_sha256_16": "7da83f06ec9f001d", "prefill_s": 0.516304, "prefill_tps": 2039.498, "prompt_tokens": 1053}
```

## Lock and hardware safety

- Lock inode 12; nested nonblocking flock rejected (`exit 1`) while the pair ran.
- After the pair: `flock -n /tmp/m1-gpu.lock` succeeded; inode 12 still present; no python or llama process; `llm-inference.service` inactive.
- No ANE access, module load/unload, device-tree change, llama enable/disable, or reboot.
- No further GPU work after this pair.
