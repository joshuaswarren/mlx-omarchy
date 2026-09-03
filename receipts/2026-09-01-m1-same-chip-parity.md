# 2026-09-01 — Same-chip MLX parity (jwm1 macOS slice)

Purpose: replace the invalid M1-Max comparison in
`receipts/2026-09-01-macos-native-mlx-baseline.md` with a true **same-chip**
baseline. Run upstream MLX (Metal) via `mlx-lm` on the same Apple M1 (T8103)
that produced the Linux receipts, then return the box to Asahi/Omarchy.

**Measurement condition (added 2026-09-03):** the Linux column of this document was measured on jwm1-linux with only 1 of 8 CPU cores online (GRUB entry `Omarchy ANE test`, which supplies a static device tree and left cores 1-7 offline). The machine now boots all 8 cores. GPU-bound figures were materially unaffected (matmul TFLOP/s median 0.1556 -> 0.1576); the Linux tok/s figures are host-bound and may improve on re-measurement. The macOS MLX slice ran on the same box booted into macOS and is not affected.

## Host identity (chip match)

| Field | Value |
|---|---|
| Host | `JW-M1.local` |
| Chip | **Apple M1** (`sysctl machdep.cpu.brand_string` = `Apple M1`) |
| GPU/architecture (MLX) | `applegpu_g13g`, 8 GPU cores, 16 GB unified |
| `sysctl hw.ncpu` | 8 |
| `sysctl hw.memsize` | 17179869184 (16 GB) |
| `uname -m` | `arm64` |
| OS | macOS 13.7.8 (22H730) |

MLX device info, verbatim:

```
{'resource_limit': 499000, 'max_buffer_length': 8589934592,
 'architecture': 'applegpu_g13g', 'memory_size': 17179869184,
 'max_recommended_working_set_size': 11453251584, 'device_name': 'Apple M1'}
```

`mx.metal.device_info()` is the equivalent of `mx.device_info()` on mlx
0.30+; the 0.29.3 build used here only exposes the Metal backend helper, so
the assignment's `mx.device_info()` raises `AttributeError` (recorded).

## Software resolution (forced by OS)

System Python is 3.9.6 (`/usr/bin/python3` and Xcode bundled
`/Applications/Xcode.app/Contents/Developer/usr/bin/python3`); both too old
for current mlx. No `pyenv`, no `conda`. No Homebrew Python formula
installed. Per the assignment's "do not install system packages" rule, the
route is the user-space uv installer:

```
curl -LsSf https://astral.sh/uv/install.sh | sh
# uv 0.12.8 installed into /Users/joshuawarren/.local/bin
uv venv ~/src/mlx-bench-samechip --python 3.11
# uv downloaded CPython 3.11.16 (macos-aarch64-none, 25.9 MiB) to ~/.local/share/uv/python
uv pip install --python ~/src/mlx-bench-samechip/bin/python mlx==0.29.3 mlx-lm==0.30.2
```

The exact pin was forced, not chosen. uv refused `mlx-lm==0.31.3` because
its darwin floor is `mlx>=0.31.2`, and **every mlx release ≥ 0.30.0 ships
only `macosx_14_0_arm64`/15/26 wheels** — no `macosx_13_*` wheels exist
for the entire 0.30 and 0.32 lines. The highest mlx-lm whose darwin
requirement (`mlx>=0.29.2`) accepts an mlx with macOS-13 wheels is
`mlx-lm 0.30.2`, paired with `mlx 0.29.3` (the **last** mlx release with
`macosx_13_0_arm64` wheels; `cp311` wheel present).

PyPI wheel-tag query, verbatim (`/pypi/mlx/json`, last 12 releases):

```
0.30.0 ['macosx_14_0_arm64', 'macosx_15_0_arm64', 'macosx_26_0_arm64', 'manylinux_2_35_aarch64', 'manylinux_2_35_x86_64']
0.30.1 ... 0.32.2 — identical tag set (macOS 14/15/26 + Linux only)
LAST_MAC13_MLX: 0.29.3
   mlx-0.29.3-cp311-cp311-macosx_13_0_arm64.whl
```

mlx-lm version → darwin mlx floor (verbatim, `/pypi/mlx-lm/<v>/json`):

```
0.31.3 | mlx>=0.31.2; platform_system == "Darwin"
0.31.2 | mlx>=0.30.4; platform_system == "Darwin"
0.31.1 | mlx>=0.30.4; ...
0.31.0 | mlx>=0.30.4; ...
0.30.7 | mlx>=0.30.4; ...
0.30.6 | mlx>=0.30.4; ...
0.30.5 | mlx>=0.30.3; ...
0.30.4 | mlx>=0.30.3; ...
0.30.2 | mlx>=0.29.2; platform_system == "Darwin"   ← highest mlx-lm installable on macOS 13
```

Resolved venv: `~/src/mlx-bench-samechip` (CPython 3.11.16).
`uv pip list` (verbatim):

```
mlx                0.29.3
mlx-lm             0.30.2
mlx-metal          0.29.3
```

## Models (exact SHA pins, local snapshot)

Same pinned revisions as the Linux receipt:

- `mlx-community/Qwen2.5-0.5B-Instruct-4bit` @ `a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3`
- `mlx-community/Qwen2.5-0.5B-Instruct-bf16` @ `56d07e766edd7159fbe12ed12d9cf114bf38bf1e`

The Mac's direct `huggingface_hub.snapshot_download` stalled at ~13 KB/s on
the big safetensors blob (a 2.3 MB `.incomplete` file held for several
minutes before the connection died silently). Replaced with a
`huggingface_hub.snapshot_download` on the Linux workstation (full link,
~25 s for 4-bit) plus rsync `--append` over Tailscale at ~0.45 MB/s. Both
tarballs verified md5-identical on the Mac side:

```
md5 (/tmp/q4.tar)    = 8d4b4b1298fea791827855aa0059433b   (local md5sum matches)
md5 (/tmp/bf16.tar)  = 31a76b202d9625b51f499c9832e375db   (local md5sum matches)
```

Extracted into fresh dirs to avoid the partial HF-cache residue from the
stalled remote attempt:

- `~/src/mlx-bench-samechip/models/q4/` (q4.tar, 265 MiB safetensors)
- `~/src/mlx-bench-samechip/models/bf16/` (bf16.tar, 942 MiB safetensors)

## Command

Per run (both attempts identical apart from process state):

```
~/src/mlx-bench-samechip/bin/python -m mlx_lm generate \
  --model <snapshot-dir> --prompt '<prompt>' \
  --max-tokens 32 --temp 0 --seed 0
```

mlx-lm 0.30.2 applies the model chat template by default (no
`--ignore-chat-template`), same as the Linux runs. Reported numbers are the
**second (warm) attempt**; first (cold) attempt shown for reference. Output
below is verbatim CLI output.

## Results — bf16 (prompt: "Hi")

Attempt 2 (warm, reported):

```
==========
Hello! How can I assist you today?
==========
Prompt: 30 tokens, 377.882 tokens-per-sec
Generation: 10 tokens, 61.530 tokens-per-sec
Peak memory: 1.025 GB
```

Attempt 1 (cold, reference):

```
==========
Hello! How can I assist you today?
==========
Prompt: 30 tokens, 375.970 tokens-per-sec
Generation: 10 tokens, 61.140 tokens-per-sec
Peak memory: 1.025 GB
```

## Results — 4-bit (prompt: "What is the capital of France? Answer in one word.")

Attempt 2 (warm, reported):

```
==========
Paris
==========
Prompt: 41 tokens, 705.621 tokens-per-sec
Generation: 2 tokens, 290.281 tokens-per-sec
Peak memory: 0.320 GB
```

Attempt 1 (cold, reference):

```
==========
Paris
==========
Prompt: 41 tokens, 705.959 tokens-per-sec
Generation: 2 tokens, 289.654 tokens-per-sec
Peak memory: 0.320 GB
```

## Parity vs Linux Attempt-11 (same chip, same SHA, same prompt)

| Metric | Linux Attempt-11 (mlx-lm 0.31.3) | macOS 13.7.8 (mlx-lm 0.30.2) | macOS / Linux |
|---|---|---|---|
| **bf16** prompt tok/s | 18.343 | 377.882 | **20.60×** |
| bf16 generation tok/s | 2.467 | 61.530 | **24.94×** |
| bf16 peak memory (GB) | 0.993 | 1.025 | 1.032× |
| bf16 output text | `Hello! How can I assist you today?` | `Hello! How can I assist you today?` | **match** |
| **4-bit** prompt tok/s | 19.197 | 705.621 | **36.76×** |
| 4-bit generation tok/s | 4.223 | 290.281 | **68.73×** |
| 4-bit peak memory (GB) | 0.292 | 0.320 | 1.096× |
| 4-bit output text | `Paris` | `Paris` | **match** |

**Headline:** both generations reproduce the Linux text verbatim
(`Hello! How can I assist you today?` and `Paris`). Greedy-decode identity
holds across the mlx version difference (0.29.3 macOS vs ~0.30+/0.31.3
Linux); the kernels differ but bf16 / 4-bit greedy on these prompts
collapses to the same EOS as on Linux. Text match is **yes** for both.

**Caveat the README must carry:** throughput ratios of **20–69×** are
**not** a chip-vs-chip story. They are an environment-vs-environment
story — macOS-on-Metal (Apple's first-class MLX target) vs the M1 PCIe
Linux driver the Linux receipts came from. The numbers here are useful as
the upper bound for what `mlx-lm` on this M1 can produce when it runs on
the platform it was built for, and as a sanity that the same weights +
prompts decode to the same text on both platforms. They are **not**
evidence that the M1 is "20× faster than" itself.

## Honest-read notes

- Both runs terminated on EOS well before `--max-tokens 32` (10 and 2
  generated tokens). Greedy decoding (temp 0, seed 0) makes this
  deterministic; same short outputs on Linux.
- Generation tok/s over 2- and 10-token runs is dominated by per-token
  overhead, not throughput: 4-bit ~3.4 ms/token (1/290.3),
  bf16 ~16.3 ms/token (1/61.5). Treat these as latency-flavored numbers,
  not steady-state tok/s.
- Prompt tok/s (warm) is a 41- and 30-token prefill; also short-run noisy.
- Peak memory (MLX Metal allocator): 0.320 GB (4-bit), 1.025 GB (bf16).
- `mx.device_info()` is absent on mlx 0.29.3; the 0.30+ API was not
  available in this pin. `mx.metal.device_info()` is the equivalent
  capture and is recorded above.
- mlx-lm 0.29.3 had a missing `import logging` in `mlx_lm/utils.py`
  (visible in the cold path on a non-existent model dir — `logging.error`
  is called when the safetensors glob returns empty). It does not affect
  these runs because both model dirs contain the expected `model.safetensors`.
  Surfaced once on the first attempt against an empty/old `models/qwen25-05b-4bit`
  residue (logged as bf16 attempt 1/2 of the original run, then re-run
  against the correct path; those failures are excluded from the reported
  numbers).

## Reproduction

```
# Mac (SSH one quoted command per call)
ssh -o BatchMode=yes joshuawarren@100.67.134.6 'curl -LsSf https://astral.sh/uv/install.sh | sh'
ssh -o BatchMode=yes joshuawarren@100.67.134.6 \
  'export PATH=$HOME/.local/bin:$PATH; uv venv $HOME/src/mlx-bench-samechip --python 3.11'
ssh -o BatchMode=yes joshuawarren@100.67.134.6 \
  'uv pip install --python $HOME/src/mlx-bench-samechip/bin/python mlx==0.29.3 mlx-lm==0.30.2'

# Linux workstation: snapshot the pinned revisions into local dirs, tar, rsync to Mac.
# (See "Models" section above for the exact hashes and md5 verification.)

# Mac
ssh -o BatchMode=yes joshuawarren@100.67.134.6 \
  'mkdir -p $HOME/src/mlx-bench-samechip/models/{q4,bf16} && \
   tar -C $HOME/src/mlx-bench-samechip/models/q4    -xf /tmp/q4.tar && \
   tar -C $HOME/src/mlx-bench-samechip/models/bf16  -xf /tmp/bf16.tar'

# Mac warm runs
ssh -o BatchMode=yes joshuawarren@100.67.134.6 \
  '$HOME/src/mlx-bench-samechip/bin/python -m mlx_lm generate \
   --model $HOME/src/mlx-bench-samechip/models/bf16 --prompt "Hi" \
   --max-tokens 32 --temp 0 --seed 0'
ssh -o BatchMode=yes joshuawarren@100.67.134.6 \
  '$HOME/src/mlx-bench-samechip/bin/python -m mlx_lm generate \
   --model $HOME/src/mlx-bench-samechip/models/q4 --prompt "What is the capital of France? Answer in one word." \
   --max-tokens 32 --temp 0 --seed 0'
```
