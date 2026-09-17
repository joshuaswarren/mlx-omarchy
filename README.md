# mlx-omarchy

MLX on Apple GPU under Linux.

![Primitive coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/joshuaswarren/mlx-omarchy/main/docs/coverage.json)

[MLX](https://github.com/ml-explore/mlx) is Apple's array framework. Upstream it runs on Metal, so it runs on macOS. mlx-omarchy is the GPU backend that keeps `import mlx.core as mx` and `mx.gpu` on Apple Silicon Linux through Mesa's Honeykrisp Vulkan 1.4 stack. There is no Metal. GPU work never falls back to CPU tensors.

Open defects live in the [defect ledger](docs/known-defects.md).

## Demo

https://github.com/user-attachments/assets/7b2326f0-4679-4784-9622-e403b99be853

One-command install on an M1 running Omarchy, the first model download, the streamed answer with its measured tokens per second, and the launcher entry; 2:47, unedited, no narration. Also at [joshuaswarren.github.io/mlx-omarchy](https://joshuaswarren.github.io/mlx-omarchy/).

## Hardware

Apple M1 is verified on [Omarchy](https://github.com/omarchy-mac/omarchy-mac) with Mesa Honeykrisp. Apple M1 Max GPU is measured; T6001 `/dev/accel/accel0` is live. Later SoCs follow.

## Install (v0.6.0)

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
~/.venvs/mlx/bin/pip install \
  https://github.com/joshuaswarren/mlx-omarchy/releases/download/v0.6.0/mlx_omarchy-0.32.2.dev202609161852%2B2e252962-cp314-cp314-linux_aarch64.whl
```

```bash
# Linux x86_64 dev box (software Vulkan, no ANE) — Python 3.11. The CLI ships,
# but ANE assets and the fd-protocol worker are absent; the wheel refuses when
# asked to use the ANE. Use the aarch64 wheel above for real runs.
python3 -m venv ~/.venvs/mlx
~/.venvs/mlx/bin/pip install \
  https://github.com/joshuaswarren/mlx-omarchy/releases/download/v0.6.0/mlx_omarchy-0.32.2.dev202609161852%2B2e25296-cp311-cp311-linux_x86_64.whl
```

Wheel filenames carry the build commit, so the exact URLs and SHA256 sums are in the release notes and the `SHA256SUMS` asset, not here.

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

Qwen2.5-0.5B-Instruct-4bit, greedy decode, 32 generated tokens. Linux tok/s versus the pinned native Metal result on the same chip class. All numbers measured against the v0.6.0 wheel (`2e252962`) and the Honeykrisp fork packaging recorded in [receipts/2026-09-16-termA-cdm-barrier-trim.md](receipts/2026-09-16-termA-cdm-barrier-trim.md) (m1-test-host, mesa `mesa-honeykrisp-omarchy 26.3.0.devel.hkf96e090-2`, 6 interleaved rounds) and [receipts/2026-09-15-rmsnorm-qkv-ab/t6001-test-host/ab.txt](receipts/2026-09-15-rmsnorm-qkv-ab/t6001-test-host/ab.txt) (t6001-test-host base arm, identical fork).

M1 versus native Metal (m1-test-host, packaged A/B from [receipts/2026-09-16-termA-cdm-barrier-trim.md](receipts/2026-09-16-termA-cdm-barrier-trim.md); native from [receipts/2026-09-10-native-macos-metal-baseline/committed-base-m1/native-baseline-baseM1.json](receipts/2026-09-10-native-macos-metal-baseline/committed-base-m1/native-baseline-baseM1.json)):

| Prompt / generated | Decode tok/s | vs native | Prefill tok/s | vs native |
|---|---:|---:|---:|---:|
| 30 / 32 | 117.0 / 150.6 | 78% | 323 / 294 | +10% |
| 1053 / 32 | 105.8 / 140.4 | 75% | 1112 / 1841 | 60% |

The worst M1 gap is long-context prefill, suspected `QmmPrefillCoopmatF16` shader throughput.

M1 Max versus native Max (t6001-test-host, base arm of [receipts/2026-09-15-rmsnorm-qkv-ab/t6001-test-host/ab.txt](receipts/2026-09-15-rmsnorm-qkv-ab/t6001-test-host/ab.txt); native from [receipts/2026-09-10-native-macos-metal-baseline/native-baseline-t6001-test-host/native-baseline-t6001-test-host.json](receipts/2026-09-10-native-macos-metal-baseline/native-baseline-t6001-test-host/native-baseline-t6001-test-host.json)):

| Prompt / generated | Decode tok/s | vs native | Prefill tok/s | vs native |
|---|---:|---:|---:|---:|
| 30 / 32 | 190.7 / 287 | 66% | 456.6 / 1518 | 30% |
| 1053 / 32 | 135.5 / 284 | 48% | 2039 / 8048 | 25% |

Digests match native (`7fd25a869ff21678` short, `7da83f06ec9f001d` ctx1053, all rounds HOLD). The M1 Max short-prefill and ctx-decode numbers predate the Honeykrisp CDM-barrier trim land (TermA only packaged and measured on the M1; the M1 Max 456.6 short prefill reflects the untrimmed baseline).

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

Known gaps: compiled bfloat16 graphs are refused (`MLX_DISABLE_COMPILE=1` runs them eagerly); `ReduceScatter` is unavailable on the Linux ring transport; `fast.CustomKernel` remains a Metal subset. The rest is in [docs/known-defects.md](docs/known-defects.md).

## Neural Engine

The Apple Neural Engine is an internal accelerator for static graph regions, not a user-facing `mx.ane` device.

The v0.6.0 wheel ships the public Parakeet reference encoder on ANE end to end on the M1 (`T8103`) **and** the M1 Max (`T6001`). Both laptops pass full E2E **104/104 transcript-exact** ([receipts/2026-09-16-parakeet-e2e-both-hosts.md](receipts/2026-09-16-parakeet-e2e-both-hosts.md): m1-test-host 8541.8 ms, t6001-test-host 6418.5 ms, transcript `db501a8c…`, `encoder_hidden` = pin `38c73261…` identical bytes on both hosts, island batch 1/1/0, `tdt_fallback_reason: null`). A 100/100 warm-run soak on the v0.5.1 wheel holds both pins with zero drift (see [`ane-linux-experiments/receipts/2026-09-16-parakeet-100-run`](https://github.com/joshuaswarren/ane-linux-experiments/tree/main/receipts/2026-09-16-parakeet-100-run), criterion 14 gate 10 closed). `MLX_OMARCHY_ANE_DEVICE=off` refuses by design.

`mlx-omarchy-parakeet download` and `mlx-omarchy-parakeet transcribe` ship in the wheel (v0.6.0 clean-install gate): install the aarch64 wheel into a Python 3.14 venv and the commands are on `PATH`. The downloader fetches the pinned reference plus an audio fixture (sha-verified); `transcribe` runs mel → ANE islands → TDT → transcript from a clean install with no dev clones. `transcribe` needs host numpy + protobuf plus soundfile or ffmpeg; it refuses naming them. The x86_64 wheel carries the CLI without ANE assets and refuses explicitly when asked to use the ANE.

TM-recovery discipline on the two laptops diverges: m1-test-host (T8103) holds no-reboot recovery on the standing `omarchy-ane main` tip (`44dd9bf8`); t6001-test-host (T6001) fails-safe — the recovery path is unproven at the head, so a hang aborts the run instead of risking the box (see [`ane-linux-experiments/receipts/2026-09-16-tm-recovery-t8103`](https://github.com/joshuaswarren/ane-linux-experiments/tree/main/receipts/2026-09-16-tm-recovery-t8103) and [`…/2026-09-15-tm-recovery-t6001`](https://github.com/joshuaswarren/ane-linux-experiments/tree/main/receipts/2026-09-15-tm-recovery-t6001)).

Plan and contracts: [docs/plans/2026-09-12-coreml-parakeet-ane-plan.md](docs/plans/2026-09-12-coreml-parakeet-ane-plan.md), [docs/ane-bundles.md](docs/ane-bundles.md), [docs/2026-09-13-encoder-parity-harness.md](docs/2026-09-13-encoder-parity-harness.md).

## Contributing

### Send us your hardware results

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
