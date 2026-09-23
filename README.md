# mlx-omarchy

MLX on Apple GPU under Linux.

![Primitive coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/joshuaswarren/mlx-omarchy/main/docs/coverage.json)

[MLX](https://github.com/ml-explore/mlx) is Apple's array framework. Upstream it runs on Metal. **mlx-omarchy** is the Omarchy GPU backend that keeps `import mlx.core as mx` and `mx.gpu` on Apple Silicon Linux through Mesa's Honeykrisp Vulkan 1.4 stack. There is no Metal. GPU work never falls back to CPU tensors.

This repo is a **patch-set and overlay**, not a hard fork of MLX history. Upstream source is fetched by pin (`mlx.lock` + `scripts/prepare-mlx.sh`); project files live under `overlay/`, and edits to upstream files stay in a small `patches/` series. The Python module name remains `mlx`. Do not install upstream `mlx` beside this wheel.

Open defects: [docs/known-defects.md](docs/known-defects.md).

## Demo

https://github.com/user-attachments/assets/7b2326f0-4679-4784-9622-e403b99be853

One-command install on an M1 running Omarchy, first model download, streamed answer with measured tokens/sec, and the launcher entry — 2:47, unedited. Also at [joshuaswarren.github.io/mlx-omarchy](https://joshuaswarren.github.io/mlx-omarchy/).

## Install

On Omarchy (Apple Silicon), one command installs the release wheel into a private venv under `~/.local/share/mlx-omarchy`, puts `mlx-omarchy`, `mlx-omarchy-demo`, and `mlx-omarchy-parakeet` on `~/.local/bin`, and registers **MLX Chat (Apple GPU)** in the Omarchy launcher. It never replaces Mesa or edits Omarchy package files.

```bash
curl -fsSL https://raw.githubusercontent.com/joshuaswarren/mlx-omarchy/main/install.sh | bash
```

Uninstall with `bash install.sh --uninstall`. Latest stable: [v0.7.2](https://github.com/joshuaswarren/mlx-omarchy/releases/tag/v0.7.2). Wheel filenames carry the build commit; pin the exact URL and check `SHA256SUMS` on the release.

Manual install (or any other Linux box):

```bash
# Apple Silicon (Honeykrisp) — Python 3.14 + aarch64 wheel from the latest release
python3 -m venv ~/.venvs/mlx
~/.venvs/mlx/bin/pip install <cp314 linux_aarch64 wheel URL>

# x86_64 dev box (software Vulkan, no ANE) — Python 3.11 + cp311 wheel
python3 -m venv ~/.venvs/mlx
~/.venvs/mlx/bin/pip install <cp311 linux_x86_64 wheel URL>
```

`mlx-lm` depends on upstream `mlx`, so install it with `pip install --no-deps mlx-lm` and add its own deps as `install.sh` does. Build-from-source and the Honeykrisp **fork driver** (required for verified M1 Omarchy numbers — stock Mesa is not enough): [docs/install-omarchy.md](docs/install-omarchy.md).

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

Local OpenAI-compatible serving (catalog CLI + qualification limits): [docs/serve.md](docs/serve.md). Kernel / serve-patch flags: [docs/kernel-flags.md](docs/kernel-flags.md). Serving tooling in this tree is not all in a published release yet — `docs/serve.md` marks what is qualified.

## Hardware

Verified on [Omarchy](https://github.com/omarchy-mac/omarchy-mac) with Mesa Honeykrisp / Vulkan 1.4:

| Chip | GPU (Vulkan) | Linux ANE |
|---|---|---|
| M1 (T8103) | Verified | Step 5 closed (full-ASR 104/104 bit-exact on fork driver; hybrid/full-encoder still pending) |
| M1 Max (T6001) | Measured | Live (`/dev/accel/accel0`) |
| M2 Max (T6021) | Verified (third silicon) | **Not** live-inference-qualified |

Apple GPU and Apple ANE are separate lanes. GPU qualification does not qualify ANE. Later SoCs follow. Receipts and dated detail live under `receipts/` and in [joshuaswarren/ane-linux-experiments](https://github.com/joshuaswarren/ane-linux-experiments).

## Performance

Qwen3.8-2B (4-bit mlx) on Apple M-series: upstream Metal on macOS vs omarchy Vulkan on Asahi/Omarchy. Protocol: greedy, 32 new tokens, 2 warmups, 512-token pure-prefill leg; medians of measured decode runs. Linux M1/M1 Max rows are the **v0.7.2** tag build (`fa103c867`); M1 Linux requires the Honeykrisp fork driver (see install doc).

| Hardware | OS / backend | Prefill→first token (tok/s) | Pure prefill 512 (tok/s) | Decode median (tok/s) |
|---|---|---|---|---|
| M1, 16 GB | Omarchy / Vulkan (Honeykrisp fork) | 46.7 | 128.7 | 34.3 |
| M1 Max, 64 GB | Omarchy / Vulkan | 62.3 | 267.7 | 56.8 |
| M2 Max, 96 GB | Omarchy / Vulkan | 47.2 | 75.8 | 45.0 |
| M1, 16 GB | macOS 27.0 / Metal | 101.3 | 345.4 | 49.5 |
| M1 Max, 64 GB | macOS 27.0 / Metal | 359.2 | 1019.7 | 179.5 |
| M2 Max, 96 GB | macOS 27.0 / Metal | 423.8 | 1234.7 | 220.6 |

Cross-OS token identity was never an acceptance bar; the bar is logit-level equivalence plus coherent decoding. Adapter evidence is captured per run (Vulkan loader trace naming the Apple physical device on Linux; mlx device identity on macOS). The backend refuses non-Apple GPUs by default.

Archival Qwen2.5 tables and older batteries stay in git history / linked receipts — they are **not** the current recommendation. Current text-generation guidance: [docs/serve.md](docs/serve.md).

## Feature parity

Value-tested Mac-usable primitives: **126 / 130** (2026-09-10, `docs/coverage.json`). Upstream MLX C++ on GPU: 251 / 251; Python: 11,483 / 11,847 (2026-09-11 snapshot in `receipts/2026-09-11-upstream-suite/`). Standing battery closed 30 / 30 at `8790c463`.

Known gaps include `ReduceScatter` on the Linux ring transport and `fast.CustomKernel` remaining a Metal subset. Everything else: [docs/known-defects.md](docs/known-defects.md).

## Neural Engine

ANE is an internal accelerator for static graph regions, not a user-facing `mx.ane` device. The wheel ships Parakeet reference encoder paths on ANE where qualified (M1 / M1 Max). `mlx-omarchy-parakeet download` / `transcribe` are on `PATH` after aarch64 install.

Plans and contracts: [docs/plans/2026-09-12-coreml-parakeet-ane-plan.md](docs/plans/2026-09-12-coreml-parakeet-ane-plan.md), [docs/ane-bundles.md](docs/ane-bundles.md). Driver / `libane` ABI live in [joshuaswarren/omarchy-ane](https://github.com/joshuaswarren/omarchy-ane).

## Contributing

The most useful thing an M-series owner can do is submit a redacted hardware report: [docs/contribute-data.md](docs/contribute-data.md). Quick capability capture (no install required for the light path):

```bash
git clone https://github.com/joshuaswarren/mlx-omarchy.git
cd mlx-omarchy
python3 scripts/collect_quick.py
```

Code contributors: start at [docs/CONTRIBUTOR-GUIDE.md](docs/CONTRIBUTOR-GUIDE.md). Dev machines without Apple GPU set `MLX_OMARCHY_ALLOW_NON_APPLE=1` (software Vulkan). GPU kernel changes still need Apple hardware before release.

- [docs/roadmap.md](docs/roadmap.md)
- [docs/compatibility.md](docs/compatibility.md)
- [docs/architecture.md](docs/architecture.md)
- [AGENTS.md](AGENTS.md) — agent contract for this repo

## License

MIT. Prepared MLX source keeps Apple's MIT license and copyright notices. Not affiliated with Apple.
