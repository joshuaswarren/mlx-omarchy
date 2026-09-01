# M1 mlx-lm fp16 smoke — jwm1-linux — 2026-08-31

Outcome: **blocked, no tokens generated.** Two precise blockers found, both
root-caused. Generation never reached decode: blocker 2 kills `mx.load`
before any tensor is read.

## Host identity

- Host: `jwm1-linux` (192.168.3.66), Apple M1, Omarchy, Mesa Honeykrisp
- `uname -a`: `Linux jwm1-linux 7.1.6-1-1-ARCH #1 SMP PREEMPT_DYNAMIC Sat, 08 Aug 2026 17:14:04 +0000 aarch64 GNU/Linux`
- Python: 3.14.7 (fresh venv `~/src/mlx-omarchy/.work/venv-fp16`)

## Repo commit

- M1 repo `~/src/mlx-omarchy` pulled 312c32a -> `1b85a45` "fix(vulkan):
  materialize strided slice views at eval time" (`git log --oneline -1`
  verified after pull and after all work; tracked tree unmodified).
- Upstream mlx pinned by `mlx.lock`: 0.32.2 @ 1f8e74e3f12f31365464a6867c6579f0e9b29d85.

## Wheel

**No wheel was produced.** `./scripts/build-wheel.sh` failed at 1b85a45
(log: `wheel2.log`, ~35 min run). `dist/` is empty.

Reference only (prior day-build at 312c32a, NOT from this run):
`mlx_omarchy-0.32.2.dev20260831+7b709df-cp314-cp314-linux_aarch64.whl`,
sha256 `e83195d975d8b0088c3657f348e0b14b9001a6fc7fbf5b82f2cdda08f317f6a9`
(from old `m1-wheel.log`).

## Model

- Repo: `mlx-community/Qwen2.5-0.5B-Instruct-bf16` (bf16, not quantized;
  `torch_dtype: bfloat16` in config.json)
- Revision: `56d07e766edd7159fbe12ed12d9cf114bf38bf1e` (main at download)
- Path on M1: `/home/joshuawarren/models/Qwen2.5-0.5B-Instruct-bf16-mlx`
- Downloaded on the M1 via `huggingface_hub.snapshot_download` (11 files,
  `model.safetensors` 988,097,734 bytes). HF and PyPI both reachable from
  the M1; no scp transfer needed.

## Exact generation command (both iterations)

```
cd ~/src/mlx-omarchy
MLX_DISABLE_COMPILE=1 .work/venv-fp16/bin/python -m mlx_lm generate \
  --model /home/joshuawarren/models/Qwen2.5-0.5B-Instruct-bf16-mlx \
  --prompt "Hi" --max-tokens 8            # iteration 1
  ... --temp 0                            # iteration 2, identical error
```

`MLX_DISABLE_COMPILE` name verified in `.work/mlx/mlx/compile.cpp:220`
(`std::getenv("MLX_DISABLE_COMPILE")`). `MLX_OMARCHY_ALLOW_NON_APPLE` not
set (real device). mlx-lm 0.31.3 default is already greedy (DEFAULT_TEMP=0.0
in `mlx_lm/generate.py`), so iteration 2 isolates flags as irrelevant.

## Ordered blocker list

### Blocker 1 — wheel build fails at 1b85a45 (fatal for the wheel path)

mlx-omarchy-info fails to link; `pip wheel` aborts; no artifact.

Verbatim (`wheel2.log:2569-2571, 2684`):

```
main.cpp:(.text+0x1814): undefined reference to `mlx::core::omarchy::ane::load_bundle(std::filesystem::__cxx11::path const&)'
collect2: error: ld returned 1 exit status
make[2]: *** [tools/mlx-omarchy-info/CMakeFiles/mlx-omarchy-info.dir/build.make:102: tools/mlx-omarchy-info/mlx-omarchy-info] Error 1
```
```
make: *** [Makefile:136: all] Error 2
ERROR: Failed to build one or more wheels
```

Root cause (verified by symbol inspection on the M1):

- `tools/mlx-omarchy-info/main.cpp:262` calls `omarchy::ane::load_bundle`.
- `mlx/backend/omarchy/ane/bundle.h:41` declares
  `AneBundle load_bundle(const std::filesystem::path& dir);` **without**
  `MLX_API`.
- The `mlx` target compiles with `-fvisibility=hidden`
  (`flags.make`: `CXX_FLAGS = -O3 -DNDEBUG -std=gnu++20 -fPIC -fvisibility=hidden -fvisibility-inlines-hidden ...`),
  and on Linux shared builds `MLX_API` expands to
  `__attribute__((visibility("default")))` (`mlx/api.h:14-23`).
- Evidence: `bundle.cpp.o` defines
  `T _ZN3mlx4core7omarchy3ane11load_bundleERKNSt10filesystem7__cxx114pathE`,
  and libmlx.so carries it in `.symtab` but **not** in `.dynsym`, while
  `capability_report` (declared `MLX_API`, `device.h:220`) IS in `.dynsym`.
- The Python module `core.cpython-314-aarch64-linux-gnu.so` built and
  linked fine (`[100%] Built target core`); only the tool target dies.

Fix belongs to the repo (one-line `MLX_API` on the ane declarations;
source edits forbidden for this run).

### Blocker 2 — `mx.load` crashes on ANY safetensors file (fatal for generation)

Reached via a clearly-labeled supplemental probe (NOT the wheel path):
with no wheel available, the successfully-built python package from
`.work/mlx/build/lib.linux-aarch64-cpython-314/mlx` was copied into the
venv's `site-packages/mlx`, plus `libmlx.so` into `mlx/lib/` (RUNPATH is
`$ORIGIN/lib`). mlx-lm 0.31.3 installed `--no-deps` (its `mlx` dependency
is Darwin-gated, so plain Linux install only needs
numpy/transformers/sentencepiece/protobuf/pyyaml/jinja2 - all installed).
This staging is exactly what the wheel would install; it is not a claim
that a wheel exists.

Verbatim (both iterations, `utils.py:323` in mlx-lm during `load_model`):

```
    weights.update(mx.load(wf))
IndexError: vector::_M_range_check: __n (which is 0) >= this->size() (which is 0)
```

Deterministic: also fails for a 2x3 fp16 file mlx itself just saved via
`mx.save_safetensors` (`/tmp/probe_load.py` probe), so it is the loader,
not the model.

Root cause (verified by gdb `__cxa_throw` backtrace + source read):

- Throw site: `mlx::core::default_stream(mlx::core::Device)` called from
  `mlx::core::load_safetensors` (gdb frame #2/#3).
- `mlx/io/safetensors.cpp:122`:
  `auto stream = cu::is_available() ? to_stream(s) : to_stream(s, Device::cpu);`
  The omarchy build links the `no_cuda.cpp` stub where
  `cu::is_available()` returns `false`, so the CPU branch always runs.
- `mlx/stream.cpp:20-33`: the thread-local
  `default_streams[cpu]` is sized once by `cpu::device_count()`.
  The omarchy wheel builds with `-DMLX_BUILD_CPU=OFF`
  (`build-wheel.sh` CMAKE_ARGS), and the `no_cpu` stub returns
  `device_count() == 0`, so the cpu vector has size 0 and
  `.at(0)` throws `std::out_of_range`.
- Normal GPU ops are unaffected (default device is gpu, slot size 1);
  only code that forces the CPU stream crashes - which upstream mlx's
  io loaders do whenever CUDA is unavailable. `MLX_DISABLE_COMPILE=1`
  and any sampler flags are irrelevant: the crash precedes
  tokenization. Iterations 3-4 skipped as meaningless under the
  4-iteration cap; the fix is a source/build change
  (e.g. build with `MLX_BUILD_CPU=ON`, or make the io stream choice
  respect a GPU-capable backend), all forbidden here.

## Trace-counter note (per contract)

`mlx::core::omarchy::trace` counters (`gpu_primitive_dispatches`,
`vk_submissions`, `vk_compute_dispatches`, ... `backend/omarchy/trace.h`)
are C++ atomics with **no Python binding** (grep over `python/src` and
`mlx/*.py` found none), so zero-CPU-dispatch evidence is not reachable
from a Python mlx-lm run. Python-reachable device evidence was captured
instead: `mx.default_device()` = `Device(gpu, 0)` and `mx.device_info()`:

```
device_name: Apple M1 (G13G B1)
architecture: honeykrisp
driver: Mesa Honeykrisp, api_version: 1.4.354
shader_float16: 1, unified_memory: 1, host_visible_coherent: 1
total_memory: 8113881088
```

The `mlx-omarchy-info` binary that prints these counters is itself the
artifact Blocker 1 breaks.

## Constraints honored

No source edits, no commits (git tracked tree clean), no M1 config
changes, no sudo/reboot. Left on the M1: `.work/venv-fp16` (venv),
`~/models/Qwen2.5-0.5B-Instruct-bf16-mlx` (model), `/tmp/*` probe files,
`wheel2.log`/`dl.log`. Remote commands via single-ssh-call or scp'd
scripts; ControlMaster disabled throughout.
## Attempt 6 @ cd510d2 (2026-09-01) — FIRST TOKENS

Repo pulled to `cd510d26c88f3eea3297d6973672678e1f7494d3`
("feat(vulkan): gather index dtypes, threefry RandomBits, and sampling
ops"; verified with `git rev-parse HEAD`; `git describe` →
`v0.2.0-22-gcd510d2`). Repo HEAD was detached at 815c43c on arrival;
`origin/main` was already cd510d2, so checkout replaced pull. Both
attempt-5 blockers (uint32 Take indices, RandomBits) are GONE: the
greedy chat-template path generates tokens end to end.

### Wheel

- `dist/mlx_omarchy-0.32.2.dev20260901+cd510d2-cp314-cp314-linux_aarch64.whl`
- size 2580946 bytes
- sha256 `e75025f55a37ec791e7f93718ac18364028ce72646f67623de531e88faaaf2b9`
- build log `wheel8.log`; build ran ~9.7 min (`ps` etime 09:03 at the
  last "still running" poll, sha256 receipt line 36.5 s later)
- receipt line from `wheel8.log` (verbatim):
  `Created wheel for mlx-omarchy: filename=mlx_omarchy-0.32.2.dev20260901+cd510d2-cp314-cp314-linux_aarch64.whl size=2580946 sha256=e75025f55a37ec791e7f93718ac18364028ce72646f67623de531e88faaaf2b9`
- force-reinstalled into `.work/venv-fp16` (verbatim:
  `Successfully uninstalled mlx-omarchy-0.32.2.dev20260901+815c43c` /
  `Successfully installed mlx-omarchy-0.32.2.dev20260901+cd510d2`)
- `mx.__version__`: `0.32.2.dev20260901+cd510d2`,
  default device `Device(gpu, 0)`

### Result: TOKENS. Greedy paths fully working; temp>0 sampling blocked.

All runs: `MLX_DISABLE_COMPILE=1 .work/venv-fp16/bin/python -m mlx_lm
generate --model /home/joshuawarren/models/Qwen2.5-0.5B-Instruct-bf16-mlx
--prompt "Hi" ...`.

| # | Flags (beyond base) | Result |
|---|---|---|
| 1 | `--max-tokens 8` (chat template default) | OK. Output: `The phrase "The quick brown fox jumps`. `Prompt: 30 tokens, 18.584 tokens-per-sec` / `Generation: 8 tokens, 2.482 tokens-per-sec` / `Peak memory: 0.993 GB` |
| 2 | `--max-tokens 32` | OK. Output: `The phrase "The quick brown fox jumps over the lazy dog" is a classic nursery rhyme. It's a simple and relatable statement that captures the essence of`. `Prompt: 30 tokens, 18.080 tokens-per-sec` / `Generation: 32 tokens, 2.233 tokens-per-sec` / `Peak memory: 0.993 GB` |
| 3 | `--max-tokens 32 --temp 0.9` | FAIL (iteration 1): `RuntimeError: [omarchy] Equal is not implemented for the Omarchy Vulkan backend (dtype=bool, shape=[]). No CPU fallback is available in Omarchy builds.` — surfaced at `mx.async_eval(y, logprobs)` (`mlx_lm/generate.py:455`); full trace saved as M1 `iter6_temp09.err` |
| 4 | iter 3 `+ --top-p 1.0 --min-p 0.0 --top-k 1` | FAIL (iteration 2): failure moves EARLIER to top-k filtering: `RuntimeError: [omarchy] sort row length ArgPartition is not implemented for the Omarchy Vulkan backend (dtype=uint32, shape=[1,151936]). No CPU fallback is available in Omarchy builds.` |
| 5 | `--max-tokens 32 --ignore-chat-template` | OK. Output (leading blank lines verbatim): `\n\nI am trying to create a simple program that will take a string and return the number of vowels in it. I have tried using a for loop and a`. `Prompt: 1 tokens, 0.973 tokens-per-sec` / `Generation: 32 tokens, 2.209 tokens-per-sec` / `Peak memory: 0.993 GB` |

### Diagnosis (sampling blocker)

Op-probe matrix (real Vulkan device, verbatim): `log`, `exp`,
`argmax`, `minimum` OK; `mx.equal(scalar, scalar)` FAILS with the exact
iteration-1 error; `mx.random.categorical` fails identically with
`num_samples=1` AND default args. Root cause: `categorical_inverse_cdf`
(`.work/mlx/mlx/random.cpp:405`) unconditionally builds
`astype(equal(x, m), dtype)` for the isinf branch of a `where`;
`Equal` is `OMARCHY_UNSUPPORTED(Equal)`
(`overlay/mlx/backend/omarchy/primitives.cpp:759`). mlx_lm
`sample_utils.py` `make_sampler`: every `temp > 0` chain ends in
`categorical_sampling` → `mx.random.categorical` → Equal. No CLI flag
can avoid that call, and `--top-k` additionally needs unimplemented
`ArgPartition`. Flag budget: 2 of 5 iterations used; the remaining
three cannot change either missing kernel (source-verified).

Secondary observation (not on the mlx_lm path): the Python binding
`mx.random.uniform((4,))` raises
`Invalid type tuple received in array initialization.` — separate
binding gap.

### Next ordered blocker

1. `Equal` (bool) unimplemented on the Omarchy Vulkan backend — blocks
   EVERY `temp > 0` / sampling path via `mx.random.categorical`.
   Smallest fix: implement `Equal::eval_gpu` for bool (broadcast
   elementwise), or compose it from implemented ops. The i32
   `GreaterEqual` kernel (`primitives.cpp:856`) shows the pattern to
   copy.
2. `ArgPartition` (uint32, wide shapes) unimplemented — blocks
   `--top-k` filtering (`apply_top_k` → `mx.argpartition`). Subordinate
   to blocker 1: plain temp>0 fails on Equal first.
3. After blocker 1 (and 2 for top-k), rerun the sampling legs:
   `--temp 0.9` 32-token run for tok/s + peak memory.

### Artifacts left on M1 (`/home/joshuawarren/src/mlx-omarchy/`)

`wheel8.log`, `dist/mlx_omarchy-0.32.2.dev20260901+cd510d2-cp314-cp314-linux_aarch64.whl`,
`iter6_temp09.err`. Run outputs recorded verbatim above from live
output. Repo left at cd510d2 (detached), nothing committed.
