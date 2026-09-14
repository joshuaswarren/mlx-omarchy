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

## Install (v0.4.2)

On an M1 running Omarchy, one command installs the release wheel into a private
venv under `~/.local/share/mlx-omarchy`, adds `mlx-omarchy` and
`mlx-omarchy-demo` to `~/.local/bin`, and registers MLX Chat (Apple GPU) in
the Omarchy launcher. The Install > AI menu entry stays MLX (Apple GPU).
It never replaces Mesa or edits Omarchy files.

```bash
curl -fsSL https://raw.githubusercontent.com/joshuaswarren/mlx-omarchy/main/install.sh | bash
mlx-omarchy-demo
```

[demo/README.md](demo/README.md) walks through what to expect. Remove it all
with `bash install.sh --uninstall`.

Manual install, or any other Linux box:

```bash
# Apple Silicon (M1, Honeykrisp) - Python 3.14; any Linux box (x86_64, software Vulkan, development only) - Python 3.11
python3 -m venv ~/.venvs/mlx
~/.venvs/mlx/bin/pip install <wheel URL for your platform from https://github.com/joshuaswarren/mlx-omarchy/releases/tag/v0.4.2>
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

Qwen2.5-0.5B-Instruct-4bit, greedy decode, 32 generated tokens. Linux tok/s versus the pinned native Metal result on the same chip class.

M1 (`m1-test-host`) versus native Metal, [receipt](receipts/2026-09-13-m1-test-host-perf-parity.md):

| Prompt / generated | Decode tok/s | vs native | Prefill tok/s | vs native |
|---|---:|---:|---:|---:|
| 30 / 32 | 107.3 / 150.6 | 71% | 323 / 294 | +10% |
| 1053 / 32 | 96.5 / 140.4 | 69% | 1112 / 1841 | 60% |

The worst M1 gap is long-context prefill, suspected `QmmPrefillCoopmatF16` shader throughput.

M1 Max (`t6001-test-host`) versus native Max, [receipt](receipts/2026-09-13-t6001-test-host-q4-decode/):

| Prompt / generated | Decode tok/s | vs native | Prefill tok/s | vs native |
|---|---:|---:|---:|---:|
| 30 / 32 | 147.5 / 287 | 51% | 171 / 1518 | 11% |
| 1053 / 32 | 86.3 / 284 | 30% | 2039 / 8048 | 25% |

Digests match native. M1 Max numbers are stock Mesa 26.2.2 versus the M1's 26.3.0-devel fork, so this is not a pure silicon comparison.

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

T8103 ANE works: persistent module, schema-4 add-mul through the v2 H13 adapter on device. T6001 `/dev/accel/accel0` is live after SET genpd, with exact fp16 64-element add-mul and a 100/100 soak ([receipts/2026-09-13-t6001-test-host-ane-set-exec.json](receipts/2026-09-13-t6001-test-host-ane-set-exec.json), [receipts/2026-09-13-t6001-test-host-ane-soak/](receipts/2026-09-13-t6001-test-host-ane-soak/)). The 1x896 path is still forbidden.

The Core ML lane in this repo has inspect, eligibility, download, worker, cache, TDT, tokenizer, pad elimination, and the bool-surface ABI. Public encoder compile is not closed: named H13 holes remain ([receipts/2026-09-13-encoder-leftover-now.md](receipts/2026-09-13-encoder-leftover-now.md)).

Plan and contracts: [docs/plans/2026-09-12-coreml-parakeet-ane-plan.md](docs/plans/2026-09-12-coreml-parakeet-ane-plan.md), [docs/ane-bundles.md](docs/ane-bundles.md), [docs/2026-09-13-encoder-parity-harness.md](docs/2026-09-13-encoder-parity-harness.md).

## Contributing

### Send us your hardware results

The most useful thing an M-series owner can do is run the collector and submit the report. It records chip, kernel, Mesa and Vulkan versions, correctness probes, and a benchmark sweep; it redacts user names, host names, paths, and addresses before anything is written, shows you the exact payload, and sends nothing without your explicit consent. Reports feed the public [community dataset](https://mlx-omarchy-community-data.joshua-s-warren.workers.dev/v1/results), which decides what gets fixed next.

```bash
python3 -m venv ~/.venvs/mlx-collect
~/.venvs/mlx-collect/bin/pip install <mlx-omarchy wheel for your machine>
~/.venvs/mlx-collect/bin/python scripts/collect_deep.py \
  --out mlx-omarchy-deep.tar.gz \
  --submit https://mlx-omarchy-community-data.joshua-s-warren.workers.dev
```

`scripts/collect_quick.py` is the no-install version: hardware and driver identity only, a few seconds, no network. Details of what is collected and how it is redacted are in [CONTRIBUTING.md](CONTRIBUTING.md). Query the dataset with `python3 scripts/query_community_data.py list`.

### Code

Start with the [contributor guide](docs/CONTRIBUTOR-GUIDE.md); it lists the open work and the verification each change needs. Development on any Linux machine works with a software Vulkan driver by setting `MLX_OMARCHY_ALLOW_NON_APPLE=1`; GPU kernel changes are verified on Apple hardware before release.

- [docs/roadmap.md](docs/roadmap.md): release plan
- [docs/compatibility.md](docs/compatibility.md): feature status by area
- [docs/known-defects.md](docs/known-defects.md): open and fixed defects

## License

MIT. The prepared MLX source keeps Apple's MIT license and copyright notices. This project is not affiliated with Apple.
