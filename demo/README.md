# Demo: chat on the Apple GPU under Omarchy

Five minutes, one M1 running Omarchy (Asahi Linux). Watch it first: [the recorded run](https://joshuaswarren.github.io/mlx-omarchy/) (2:47, unedited). Nothing here changes the
Mesa driver, Hyprland, or any Omarchy file; everything lands under `$HOME`.

## 1. Install

```bash
curl -fsSL https://raw.githubusercontent.com/joshuaswarren/mlx-omarchy/main/install.sh | bash
```

What you should see, in order: the two runtime packages (`lapack`, `blas`)
installed through `omarchy-pkg-add`, the release wheel downloaded and checked
against `SHA256SUMS`, a private venv created in
`~/.local/share/mlx-omarchy/venv`, and a smoke test that prints
`device: Apple M1 (G13G B1)` followed by `matmul OK`. If the device line says
`llvmpipe`, the Vulkan driver is not the Apple one; see Troubleshooting.

## 2. Run the chat demo

From a terminal:

```bash
mlx-omarchy-demo
```

or open the Omarchy launcher (Super + Space) and pick **MLX Chat (Apple
GPU)**. The first run downloads `mlx-community/Qwen2.5-0.5B-Instruct-4bit`
(about 300 MB). Then it:

1. prints the mlx-omarchy version and the GPU it is running on,
2. answers one scripted question so you can see tokens stream, with the
   measured prompt and generation tokens per second on the last line,
3. drops into an interactive chat. Empty line or Ctrl-D exits.

Expected speed on an M1 with the stock Omarchy Mesa: roughly 60 generated
tokens per second for the 0.5B 4-bit model. macOS on the same chip is faster;
this project is not at performance parity yet, and the numbers on the last
line are the honest ones.

Try a larger model:

```bash
mlx-omarchy-demo --model mlx-community/Qwen2.5-1.5B-Instruct-4bit
```

## 3. Use it from your own code

`mlx-omarchy` is the interpreter of the private venv:

```bash
mlx-omarchy -c 'import mlx.core as mx; print(mx.device_info()); print((mx.ones((4,4)) @ mx.ones((4,4))).tolist())'
mlx-omarchy -m mlx_lm.generate --model mlx-community/Qwen2.5-0.5B-Instruct-4bit --prompt "Hello"
```

The API is upstream MLX (`import mlx.core as mx`, `mlx.nn`, `mlx_lm`), so
upstream examples run unchanged when they stay inside the supported
operations ([docs/compatibility.md](../docs/compatibility.md)).

## 4. Remove it

```bash
curl -fsSL https://raw.githubusercontent.com/joshuaswarren/mlx-omarchy/main/install.sh | bash -s -- --uninstall
```

## Troubleshooting

- `Python 3.14 is required`: Omarchy on Asahi ships Python 3.14; `pacman -Syu`
  if yours is older.
- `device: llvmpipe`: the Honeykrisp driver was not picked up. Check
  `vulkaninfo --summary` (package `vulkan-tools`) lists `Apple M1`; unset any
  `VK_ICD_FILENAMES`/`VK_DRIVER_FILES` you set for other software.
- Model download hangs: the demo reads `HF_HUB_OFFLINE`, `HF_ENDPOINT`, and
  the usual proxy variables, like any Hugging Face download.
- Something else: [docs/known-defects.md](../docs/known-defects.md), then an
  issue with the output of `mlx-omarchy -c 'import mlx.core as mx; print(mx.__version__, mx.device_info())'`.
