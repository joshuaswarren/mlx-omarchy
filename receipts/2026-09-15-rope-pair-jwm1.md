# jwm1 rope-pair confirmation A/B — 2f9e6fa9 vs the 249-dispatch base
Date: 2026-09-15

## Verdict

All three rope-pair land clauses confirmed on jwm1. The origin/main
wheel (`+2f9e6fa`, receipts-only child of code commit `1c49674f`)
dispatches **225 vk/token** on jwm1 (base 249, TRIO=0 fallback 249),
every one of the 20 interleaved A/B runs hit both pinned digests
exactly, and decode medians rise vs the same-day interleaved base:
short +0.68%, ctx1053 +1.38%. Against the pinned native Metal divisor
jwm1 Linux stands at 74.6% (short) and 69.6% (1K context) of native
decode.

## Measurements

5-round interleaved A/B (arm order alternated per round), both legs,
fresh `bench_decode.py` subprocess per leg, `MLX_DISABLE_COMPILE=1
HF_HUB_OFFLINE=1`, `--tokens 32 --temp 0.0 --seed 0 --warmup-tokens 4`,
pins fatal (`ab_decode.py`). All runs 2026-09-15T19:07–19:08Z.

| Metric | base b41e2b74 | ropepair 2f9e6fa | cand/base | Native M1 | base/native | cand/native |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q4 short decode, 30/32 | 111.5737 tok/s | 112.3288 tok/s | +0.68% | 150.57 tok/s | 74.10% | 74.60% |
| Q4 short prefill, 30 tokens | 329.4497 tok/s | 376.6373 tok/s | +14.32% | 294.10 tok/s | — | — |
| Q4 1K-context decode, 1053/32 | 96.3391 tok/s | 97.6719 tok/s | +1.38% | 140.38 tok/s | 68.63% | 69.58% |
| Q4 1K-context prefill, 1053 tokens | 1111.715 tok/s | 1108.3298 tok/s | −0.30% | 1840.90 tok/s | — | — |

Per-run decode arrays (round order):
- short base: 105.744, 110.0933, 111.6183, 111.5737, 111.9076
- short ropepair: 112.286, 112.2409, 113.4778, 113.2955, 112.3288
- ctx1024 base: 95.8586, 96.5011, 96.3665, 96.3391, 96.2049
- ctx1024 ropepair: 98.3222, 98.0812, 97.41, 97.3227, 97.6719

Pinned generated-ID digests: 20/20 runs exact, both legs, both arms.
```text
short:   7fd25a869ff21678
ctx1024: 7da83f06ec9f001d
```
Today base medians sit above the single runs of
`receipts/2026-09-14-jwm1-gpu-parity-rerun.md` (107.285 short /
95.0352 ctx), so the like-for-like interleaved delta is the honest
comparison; the native divisor (150.57 / 140.38) is unchanged from
`native-2026-09-06-summary.json` as pinned in that receipt.

## Dispatch counts

Per-token `vk_compute_dispatches` (median of 7 decode tokens, 3 probes
each, `dispatch_count.py`, `MLX_DISABLE_COMPILE=1`):

| Config | vk/token | runs | gpu_primitive_dispatches | vk_submissions |
| --- | ---: | --- | ---: | ---: |
| base default (b41e2b74) | 249 | 249/249/249 | 858 | 2 |
| cand default (2f9e6fa) | **225** | 225/225/225 | 858 | 2 |
| cand `MLX_OMARCHY_FUSED_TRIO=0` | 249 | 249/249/249 | 858 | 2 |

The 23 rope pairs fire on jwm1 identically to jw16: the trio-plan
trace shows `rope_nodes=47` planned with zero rejections
(`rej_node_inputs=0 rej_node_offset=0 rej_node_dtype=0
rej_node_notforward=0`).

## Identity

- Measured source: origin/main `2f9e6fa927ee65e1dd1cafb928cec32e4e6e0a0a`
  (verified receipts-only diff vs code commit `1c49674f`).
- Cand wheel: `mlx_omarchy-0.32.2.dev202609151859+2f9e6fa-cp314-cp314-linux_aarch64.whl`
  SHA-256 `88fd03b293b13fb8150c4ecf5678455057b9fef27de153ff17ad84dcbe8bccc7`
  (built on jwm1 via `DEV_RELEASE=1 scripts/build-wheel.sh`).
- Base wheel: `mlx_omarchy-0.32.2.dev202609122355+b41e2b74-...`
  SHA-256 `81743cd1a631f6c5d8aa7ff9538d1ec4ddcb2a57c184da7c55345e48d349a240`
  — byte-identical to the wheel pinned in the 2026-09-14 parity rerun.
- Provenance `verified=match` on every run, both wheels; harness pinned
  to `b41e2b74` (`scripts/bench_decode.py`
  `f5062d88f34b0845c1e59b0b35d2e33ae02180f636a0f65c543a4366ec2bef7f`,
  `scripts/bench_matrix.json`
  `df8eb9f3ed84182604379b7cde070ade706bfe590fbf83a35b1877cfaf43f258`).
- Python 3.14.7; mlx-lm 0.31.3; host `jwm1-linux`, aarch64, kernel
  7.1.6-1-1-ARCH; Apple M1 (G13G B1); Vulkan Honeykrisp, Mesa
  26.3.0-devel (git-6f6afc8968); `Device(gpu, 0)`.
- Model `mlx-community/Qwen2.5-0.5B-Instruct-4bit` at
  `/home/joshuawarren/models/Qwen2.5-0.5B-Instruct-4bit-mlx`, config
  SHA-256 `b045e57ea90b8f1b35f89f954b176a5c1faa02bd0af2c89bcec191239d66cef4`,
  index SHA-256 `54001cb4c11197119c206dde28e7be08e5872aab6c6d271aed339ec77e84f870`
  — both equal to the pinned parity-rerun receipt.
- No diagnostics profile, matmul, ANE command, reboot, 1x896, or SET
  write. `63c1d3cf` untouched and not merged.

## Lock and hardware safety

One `flock` hold on `/tmp/m1-gpu.lock` (inode 29) for the whole
battery: nested `flock -n` refused while held, inode 29 before and
after, never unlinked, free again after the run. Power during the run:
AC online, battery Full.

## Artifacts

- Battery: `/var/tmp/DecodeCompileAB/run-jwm1-ropepair.sh`, log
  `jwm1-ropepair-run.log`, outputs in
  `/var/tmp/DecodeCompileAB/jwm1-ropepair-out/` (`ab.json`,
  `ab.txt`, `dispatch-*.txt`, `dispatch-trace.{txt,err}`,
  `wheel-sha256.txt`, `lock.txt`).
- Cand build tree: `/var/tmp/ropepair-jwm1-2f9e6fa9` (detached
  `2f9e6fa9`), build log `build.log`.
