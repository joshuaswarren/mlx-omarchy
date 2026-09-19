# mlx-omarchy

MLX on Apple GPU under Linux.

![Primitive coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/joshuaswarren/mlx-omarchy/main/docs/coverage.json)

[MLX](https://github.com/ml-explore/mlx) is Apple's array framework. Upstream it runs on Metal, so it runs on macOS. mlx-omarchy is the GPU backend that keeps `import mlx.core as mx` and `mx.gpu` on Apple Silicon Linux through Mesa's Honeykrisp Vulkan 1.4 stack. There is no Metal. GPU work never falls back to CPU tensors.

Open defects live in the [defect ledger](docs/known-defects.md).

## Demo

https://github.com/user-attachments/assets/7b2326f0-4679-4784-9622-e403b99be853

One-command install on an M1 running Omarchy, the first model download, the streamed answer with its measured tokens per second, and the launcher entry; 2:47, unedited, no narration. Also at [joshuaswarren.github.io/mlx-omarchy](https://joshuaswarren.github.io/mlx-omarchy/).

## Serve a local model

OpenAI-compatible server for Codex, omp, pi, Hermes, OpenClaw, and Claude Code:
[docs/serve.md](docs/serve.md). RAM-tier model IDs and exact commands are there.

## Hardware

Apple M1 is verified on [Omarchy](https://github.com/omarchy-mac/omarchy-mac) with Mesa Honeykrisp. Apple M1 Max GPU is measured; T6001 `/dev/accel/accel0` is live. Later SoCs follow.

## Install

On an M1 running Omarchy, one command installs the release wheel into a private
venv under `~/.local/share/mlx-omarchy`, adds `mlx-omarchy`, `mlx-omarchy-demo`,
and `mlx-omarchy-parakeet` (Parakeet download + ANE transcribe) to `~/.local/bin`,
and registers MLX Chat (Apple GPU) in the Omarchy launcher. The Install > AI menu
entry stays MLX (Apple GPU). It never replaces Mesa or edits Omarchy files.

```bash
curl -fsSL https://raw.githubusercontent.com/joshuaswarren/mlx-omarchy/main/install.sh | bash
mlx-omarchy-demo
```

[demo/README.md](demo/README.md) walks through what to expect. Remove it all
with `bash install.sh --uninstall`.

Manual install, or any other Linux box:

```bash
# Apple Silicon (M1, Honeykrisp) — Python 3.14
python3 -m venv ~/.venvs/mlx
~/.venvs/mlx/bin/pip install <the cp314 linux_aarch64 wheel from the latest release>
# https://github.com/joshuaswarren/mlx-omarchy/releases/latest
```

```bash
# Linux x86_64 dev box (software Vulkan, no ANE) — Python 3.11. The CLI ships,
# but ANE assets and the fd-protocol worker are absent; the wheel refuses when
# asked to use the ANE. Use the aarch64 wheel above for real runs.
python3 -m venv ~/.venvs/mlx
~/.venvs/mlx/bin/pip install <the cp311 linux_x86_64 wheel from the latest release>
```

Wheel filenames carry the build commit, so pin the exact URL from the release
you choose; the SHA256 sums live in the release notes and the `SHA256SUMS`
asset.

Building from source is covered in [docs/install-omarchy.md](docs/install-omarchy.md). Build dependencies: Python 3.10+, CMake 3.25+, Vulkan headers, a C++ compiler, and the BLAS/LAPACK packages named there.

Do not install the upstream `mlx` package beside this wheel; both provide the `mlx` module. `mlx-lm` depends on upstream `mlx`, so install it with `pip install --no-deps mlx-lm` and add its own dependencies (`transformers[sentencepiece] numpy protobuf pyyaml jinja2 huggingface_hub`) as `install.sh` does.

## Quick start

```python
import mlx.core as mx

x = mx.array([[1.0, 2.0], [3.0, 4.0]])
w = mx.array([[0.5], [0.25]])

def loss(w):
    return mx.exp(x @ w).sum()

value, grad = mx.value_and_grad(loss)(w)
print(value, grad)
```

Run a language model with [mlx-lm](https://github.com/ml-explore/mlx-lm):

```bash
pip install mlx-lm
python -m mlx_lm generate \
  --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --prompt "What is the capital of France? Answer in one word." \
  --max-tokens 32
```

## Performance

Qwen2.5-0.5B-Instruct-4bit, greedy decode, 32 generated tokens. Linux tok/s versus the pinned native Metal result on the same chip class, one same-protocol battery per host.

M1 versus native Metal (m1-test-host, single same-protocol battery on the v0.6.3 wheel, `ane-linux-experiments` `receipts/2026-09-17-m1-test-host-gpu-parity-refresh.md`; native from [receipts/2026-09-10-native-macos-metal-baseline/committed-base-m1/native-baseline-baseM1.json](receipts/2026-09-10-native-macos-metal-baseline/committed-base-m1/native-baseline-baseM1.json)):

| Prompt / generated | Decode tok/s | vs native | Prefill tok/s | vs native |
|---|---:|---:|---:|---:|
| 30 / 32 | 117.3 / 150.6 | 78% | 390.8 / 294 | +33% |
| 1053 / 32 | 105.7 / 140.4 | 75% | 1106 / 1841 | 60% |

Against the same-protocol 2026-09-14 rerun that is +9.4% short decode, +11.2% ctx decode, +18.1% short prefill, and flat (−0.5%) ctx prefill — the CDM barrier trim, the rope-pair trio, the SwiGLU store epilogue, and `map_mode=3` move decode and short prefill; the long-context prefill gap stays pinned on `QmmPrefillCoopmatF16` shader throughput.

M1 Max versus native Max (t6001-test-host, single same-protocol battery on the v0.6.1 wheel, `ane-linux-experiments` `receipts/2026-09-17-t6001-test-host-gpu-parity-refresh.md`; native from [receipts/2026-09-10-native-macos-metal-baseline/native-baseline-t6001-test-host/native-baseline-t6001-test-host.json](receipts/2026-09-10-native-macos-metal-baseline/native-baseline-t6001-test-host/native-baseline-t6001-test-host.json)):

| Prompt / generated | Decode tok/s | vs native | Prefill tok/s | vs native |
|---|---:|---:|---:|---:|
| 30 / 32 | 190.6 / 287 | 66% | 459.7 / 1518 | 30% |
| 1053 / 32 | 130.7 / 284 | 46% | 3845 / 8048 | 48% |

Same battery on the v0.7.1 wheel (`receipts/2026-09-19-q4-chainbatch-t6001-test-host.md` in `ane-linux-experiments`, 2026-09-19; two-pass decode included; digests fatal and exact):

| Prompt / generated | Decode tok/s | vs native | Prefill tok/s | vs native |
|---|---:|---:|---:|---:|
| 30 / 32 | 191.9 / 287 | 67% | 457 / 1518 | 30% |
| 1053 / 32 | 150.8 / 284 | 53% | 3884 / 8048 | 48% |

Digests match native (`7fd25a869ff21678` short, `7da83f06ec9f001d` ctx1053, every leg asserted). Against the same-protocol 2026-09-14 rerun that is +29% short decode, +49% ctx decode, +130% short prefill, and +86% ctx prefill, from the rope-pair trio, the SwiGLU store epilogue, the tile-M occupancy floor, and the SPIR-V disk cache. The M1 Max still runs the untrimmed CDM barrier: the Honeykrisp trim ships G13G-only because on G13X the designed bit set measured +13% short decode against −3.2% ctx1053, so it was not shipped. The batch-across-chains follow-up (`hk/cdm-chain-batch`) was screened on t6001-test-host on 2026-09-19 and corrupts generated IDs nondeterministically (16 vs ~2500 barriers per 2-token run, digests flip run-to-run) — the per-launch G13X barrier is load-bearing memory ordering between dependent launches, and that fixed per-token cost is not addressable in this driver generation.

M2 Max — third Vulkan device, numbers only (t6021-test-host-linux, same-protocol battery on the F1-fixed `b744f4dd` wheel, single locked pass, eager per the standing protocol, [`receipts/2026-09-18-t6021-test-host-third-vulkan-device.md`](https://github.com/joshuaswarren/ane-linux-experiments/blob/main/receipts/2026-09-18-t6021-test-host-third-vulkan-device.md)):

| Prompt / generated | Decode tok/s | Prefill tok/s | Generated-ID digest |
|---|---:|---:|---|
| 30 / 32 | 182.43 | 224.44 | `7fd25a869ff21678` (= native pin) |
| 262 / 128 | 155.35 | 1593.52 | `55215e22d7f1b864` |
| 1053 / 32 | 99.66 | 2416.61 | `7da83f06ec9f001d` (= native pin) |

Both native-pinned digests reproduce bit-exact on T6021 (G14C B1, Honeykrisp). No native-M2-Max Metal divisor run exists, so no versus-native percentages are claimed for this device.

To reproduce a leg:

```bash
MLX_DISABLE_COMPILE=1 python3 scripts/bench_decode.py \
  --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --prompt "What is the capital of France? Answer in one word." \
  --tokens 64
```

## Feature parity

The badge above is value-tested coverage of the 130 Mac-usable primitives: 126 / 130 as of 2026-09-10 (`docs/coverage.json`).

| Measure | Result | Date |
|---|---|---|
| Value-tested Mac-usable primitives | 126 / 130 | 2026-09-10 |
| Upstream MLX C++ cases on the GPU device | 251 / 251 | 2026-09-11 |
| Upstream MLX Python cases on the GPU device | 11,483 / 11,847 | 2026-09-11 |
| Standing battery | 30 / 30 | `8790c463` |

C++ and Python counts are the dated snapshot in [receipts/2026-09-11-upstream-suite/](receipts/2026-09-11-upstream-suite/). The battery closed 30 / 30 at commit `8790c463`.

Known gaps: `ReduceScatter` is unavailable on the Linux ring transport; `fast.CustomKernel` remains a Metal subset. The rest is in [docs/known-defects.md](docs/known-defects.md).

The compiled-bf16 tape gate lifted 2026-09-18 (`064b7301`): compile-ON is digest-identical to eager across the five-model recert matrix on the v0.7.0 wheel bytes (main `b283a16f`, wheel sha256 `2def345c…`) — Qwen3.5-9B-MLX-4bit `910abe30d4305271` (deterministic across two compiled reps), gemma-4-31b-it-4bit `9f1fe40101db3a4b`, Ministral-3-8B-4bit `d4735e3a265e16ee`, Ternary-Bonsai-8B-2bit `25dc382d3170a80c`, and Ternary-Bonsai-2-27B coherent at **1.44 tok/s** compiled-ON ([recert receipt](receipts/2026-09-18-v070-pretag-recert-t6001-test-host.md)). The fused-chain `(N,1,L)` broadcast misindexing behind the Bonsai-2-27B NaNs (F7) is fixed at `da43969e`; the in-model fused-bf16 fence is removed with it (the corrupt-matrix probe with fusion ON is clean on all three probed models, so bf16 tape nodes fuse again).

## Neural Engine

The Apple Neural Engine is an internal accelerator for static graph regions, not a user-facing `mx.ane` device.

The v0.6.0 wheel ships the public Parakeet reference encoder on ANE end to end on the M1 (`T8103`) **and** the M1 Max (`T6001`). Both laptops pass full E2E **104/104 transcript-exact** ([receipts/2026-09-16-parakeet-e2e-both-hosts.md](receipts/2026-09-16-parakeet-e2e-both-hosts.md): m1-test-host 8541.8 ms, t6001-test-host 6418.5 ms, transcript `db501a8c…`, `encoder_hidden` = pin `38c73261…` identical bytes on both hosts, island batch 1/1/0, `tdt_fallback_reason: null`).

macOS CoreML divisor for the same die (m1-test-host/T8103, [`ane-linux-experiments/receipts/2026-09-18-t8103-divisor-macos27.md`](https://github.com/joshuaswarren/ane-linux-experiments/blob/main/receipts/2026-09-18-t8103-divisor-macos27.md)): CoreML `transcribe` wall, median of all 10 runs, **259.9 ms `.ane` / 266.7 ms `.all`** on macOS 27.0 / CoreML 3600.25.2; the M1-Ultra reference (studio-host) is 292.2 / 305.8 ms on macOS 26.6.2 / CoreML 3520. **Cross-OS caveat: the two sides are different OS/CoreML generations, so ratios are indicative, not exact.** This is also a different stage definition than the Linux pipeline number above (CoreML `transcribe` wall vs whole mel → ANE → TDT pipeline) — the divisor is the reference-class target for the Linux ANE port, not a like-for-like comparison.

Linux ANE island encoder wall, re-baselined on the published v0.7.1 bytes (t6001-test-host/T6001, medians of 3, tag `50eeb290`, [`ane-linux-experiments/receipts/2026-09-19-encwall-v071-attribution.md`](https://github.com/joshuawarren/ane-linux-experiments/blob/main/receipts/2026-09-19-encwall-v071-attribution.md)): **AC serve 3359.9 ms, ACO serve 3779.0 ms** (launch 4278.6 / 4825.8 ms; placement families per [docs/ane-encoder-placement.md](docs/ane-encoder-placement.md)) — within noise of the 3422 / 3797 ms the `ffn-chain-fusion` lane carried, so the wall is unchanged across the timeline-stall fix. Per-round attribution puts GPU-side `matmul`/`const` synchronous drains feeding and between islands first (~1950 ms of the wall), then island round-trip latency (1291 ms): the parent's stdout read alone is 1023 ms across 48 rounds, **21 ms/round** against ~1.2 ms of device compute per island. A transport-cut screen that removed three host-side copies per byte moved the wall only −8 / −6 ms, inside run-to-run spread (NO-LAND): the wall is round-trip latency plus GPU drains, not host memcpy. The relay-bypass follow-up (2026-09-19, main `e14752ff`, [`receipts/2026-09-19-encwall-relay-bypass.md`](https://github.com/joshuawarren/ane-linux-experiments/blob/main/receipts/2026-09-19-encwall-relay-bypass.md)) removed the relay's parse/reframe hops entirely — the runner speaks the resident wire protocol through a single-threaded poll/splice pump — and cut the serve wall for real: **AC serve 2530.6 ms median (−26.4% vs 3440.5), ACO serve 3044.6 ms (−22.0% vs 3904.9)**, transcript pins identical across the 12-arm battery (3 reps × {AC,ACO} × {base,cand}).

Linux Parakeet E2E wall on the same v0.7.1 bytes (t6001-test-host/T6001, warm-up + 10 timed whole-pipeline runs, median of runs 2–10, [`ane-linux-experiments/receipts/2026-09-19-parakeet-e2e-v071-t6001-test-host.md`](https://github.com/joshuaswarren/ane-linux-experiments/blob/main/receipts/2026-09-19-parakeet-e2e-v071-t6001-test-host.md)): **4773.8 ms `.ane`** (all-10 median 4779.4; like-for-like without `audio_load` 4698.0), 104/104 pins on every measured run (transcript `db501a8c…`, `encoder_hidden` `38c73261…`, mel `5b54f4a9…`, 0 timeouts) — **16.3× the M1-Ultra 292.2 ms `.ane` divisor** (like-for-like 16.1×), down from 6418.5 ms (22×) on the v0.6.1-era stack. The pure-GPU `.all` analog (`MLX_OMARCHY_PLACED=""`, whole pipeline on Vulkan) runs **2165.9 ms** but diverges at the tensor level by design (97-token golden prefix), so it is a wall reference, not a transcript claim.

A 100/100 warm-run soak on the v0.5.1 wheel holds both pins with zero drift (see [`ane-linux-experiments/receipts/2026-09-16-parakeet-100-run`](https://github.com/joshuaswarren/ane-linux-experiments/tree/main/receipts/2026-09-16-parakeet-100-run), criterion 14 gate 10 closed). `MLX_OMARCHY_ANE_DEVICE=off` refuses by design.

`mlx-omarchy-parakeet download` and `mlx-omarchy-parakeet transcribe` ship in the wheel (v0.6.0 clean-install gate): install the aarch64 wheel into a Python 3.14 venv and the commands are on `PATH`. The downloader fetches the pinned reference plus an audio fixture (sha-verified); `transcribe` runs mel → ANE islands → TDT → transcript from a clean install with no dev clones. `transcribe` needs host numpy + protobuf plus soundfile or ffmpeg; it refuses naming them. The x86_64 wheel carries the CLI without ANE assets and refuses explicitly when asked to use the ANE.

TM-recovery discipline on the two laptops diverges: m1-test-host (T8103) holds no-reboot recovery on the standing `omarchy-ane main` tip (`44dd9bf8`); t6001-test-host (T6001) fails-safe — the recovery path is unproven at the head, so a hang aborts the run instead of risking the box (see [`ane-linux-experiments/receipts/2026-09-16-tm-recovery-t8103`](https://github.com/joshuaswarren/ane-linux-experiments/tree/main/receipts/2026-09-16-tm-recovery-t8103) and [`…/2026-09-15-tm-recovery-t6001`](https://github.com/joshuaswarren/ane-linux-experiments/tree/main/receipts/2026-09-15-tm-recovery-t6001)).

Plan and contracts: [docs/plans/2026-09-12-coreml-parakeet-ane-plan.md](docs/plans/2026-09-12-coreml-parakeet-ane-plan.md), [docs/ane-bundles.md](docs/ane-bundles.md), [docs/2026-09-13-encoder-parity-harness.md](docs/2026-09-13-encoder-parity-harness.md).

## Contributing

### Send us your hardware results

Want to help? Start at **[docs/contribute-data.md](docs/contribute-data.md)** —
the ten-minute version of "how do I contribute" with links to everything below.

The most useful thing an M-series owner can do is run the collector and submit the report. It records chip, kernel, Mesa and Vulkan versions, correctness probes, and a benchmark sweep; it redacts user names, host names, paths, and addresses before anything is written, shows you the exact payload, and sends nothing without your explicit consent. Reports feed the public [community dataset](https://mlx-omarchy-community-data.joshua-s-warren.workers.dev/v1/results), which decides what gets fixed next.

Three copy-paste paths. Pick the one that matches your machine.

**Linux — quick capability report (no install, no network, a few seconds):**

```bash
git clone https://github.com/joshuaswarren/mlx-omarchy.git
cd mlx-omarchy
python3 scripts/collect_quick.py
# or, in one command:
python3 scripts/collect_quick.py \
  --submit https://mlx-omarchy-community-data.joshua-s-warren.workers.dev
```

Quick mode is sufficient to capture the ANE devicetree (ane node, DARTs, PMGR
domains, AIC, phandles) at enough fidelity to author the omarchy-ane overlay
off-machine. Run the full report below when you want benchmark numbers or the
correctness sweep. On a non-Apple box the wheel installs and the quick
collector runs without complaint; the GPU path runs in software Vulkan and the
ANE section is simply absent.

**Linux — full report (needs the v0.6.0 aarch64 wheel on an Apple Silicon host; x86_64 dev box installs the cp311 wheel and runs the same script):**

```bash
git clone https://github.com/joshuaswarren/mlx-omarchy.git
cd mlx-omarchy

# Apple Silicon (M1/M1 Max/etc, Honeykrisp) — Python 3.14 + aarch64 wheel
python3.14 -m venv ~/.venvs/mlx-collect
~/.venvs/mlx-collect/bin/pip install \
  https://github.com/joshuaswarren/mlx-omarchy/releases/download/v0.6.0/mlx_omarchy-0.32.2.dev202609161852%2B2e252962-cp314-cp314-linux_aarch64.whl

# Linux x86_64 dev box — Python 3.11 + cp311 wheel (no ANE assets):
# python3.11 -m venv ~/.venvs/mlx-collect
# ~/.venvs/mlx-collect/bin/pip install \
#   https://github.com/joshuaswarren/mlx-omarchy/releases/download/v0.6.0/mlx_omarchy-0.32.2.dev202609161852%2B2e25296-cp311-cp311-linux_x86_64.whl

# preview only, writes nothing:
~/.venvs/mlx-collect/bin/python scripts/collect_deep.py

# archive + paste-ready cover text:
~/.venvs/mlx-collect/bin/python scripts/collect_deep.py \
  --out mlx-omarchy-deep.tar.gz

# publish in the same command (--submit requires --out):
~/.venvs/mlx-collect/bin/python scripts/collect_deep.py \
  --out mlx-omarchy-deep.tar.gz \
  --submit https://mlx-omarchy-community-data.joshua-s-warren.workers.dev
```

**macOS — native MLX reference report (Apple Silicon Mac, native MLX in a Python supported by that package):**

```bash
git clone https://github.com/joshuaswarren/mlx-omarchy.git
cd mlx-omarchy

python3.14 -m venv ~/.venvs/mlx-collect-macos
~/.venvs/mlx-collect-macos/bin/python -m pip install mlx==0.32.1

# preview only:
~/.venvs/mlx-collect-macos/bin/python scripts/collect_deep.py

# archive + publish:
~/.venvs/mlx-collect-macos/bin/python scripts/collect_deep.py \
  --out mlx-macos-reference.tar.gz \
  --submit https://mlx-omarchy-community-data.joshua-s-warren.workers.dev
```

Details of what is collected and how it is redacted are in [CONTRIBUTING.md](CONTRIBUTING.md) ([macOS setup](CONTRIBUTING.md#macos-setup), [Linux setup](CONTRIBUTING.md#linux-setup)). Query the dataset with `python3 scripts/query_community_data.py list`.

### Code

Start with the [contributor guide](docs/CONTRIBUTOR-GUIDE.md); it lists the open work and the verification each change needs. Development on any Linux machine works with a software Vulkan driver by setting `MLX_OMARCHY_ALLOW_NON_APPLE=1`; GPU kernel changes are verified on Apple hardware before release.

- [docs/roadmap.md](docs/roadmap.md): release plan
- [docs/compatibility.md](docs/compatibility.md): feature status by area
- [docs/known-defects.md](docs/known-defects.md): open and fixed defects

## License

MIT. The prepared MLX source keeps Apple's MIT license and copyright notices. This project is not affiliated with Apple.
