# M1 mlx-lm fp16 smoke — jwm1-linux — 2026-08-31

Outcome: **blocked, no tokens generated.** Two precise blockers found, both
root-caused. Generation never reached decode: blocker 2 kills `mx.load`
before any tensor is read.

**Measurement condition (added 2026-09-03):** all jwm1-linux timings in this document were taken with only 1 of 8 CPU cores online (GRUB entry `Omarchy ANE test`, which supplies a static device tree and left cores 1-7 offline). The machine now boots all 8 cores. GPU-bound figures were materially unaffected (matmul TFLOP/s median 0.1556 -> 0.1576); these host-bound figures (prompt/gen tok/s) may improve on re-measurement.

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

## Attempt 7 @ b9745b2 (qualification)

- Date 2026-09-01. M1 repo fetched and checked out to
  `b9745b235422e76b591dcf0a4731be75b3920e43`
  ("feat(vulkan): comparison, scan, searchsorted, and sampling ops";
  `git rev-parse HEAD` verified after checkout and after all work;
  detached HEAD, tracked tree unmodified, nothing committed).
- Headline: **temp > 0 sampling generates tokens.** Attempt 6's `Equal`
  blocker is resolved at this commit; every mlx-lm generation leg passes.

### Wheel

- `dist/mlx_omarchy-0.32.2.dev20260901+b9745b2-cp314-cp314-linux_aarch64.whl`
- size 2592264 bytes
- sha256 `63899c6b5f0bb6299204e38564d1abdc29189cfd3bb26bb55fca554ce74c3f31`
- build log `wheel9.log`; receipt line (verbatim):
  `Created wheel for mlx-omarchy: filename=mlx_omarchy-0.32.2.dev20260901+b9745b2-cp314-cp314-linux_aarch64.whl size=2592264 sha256=63899c6b5f0bb6299204e38564d1abdc29189cfd3bb26bb55fca554ce74c3f31`
- force-reinstalled into `.work/venv-fp16` (verbatim:
  `Successfully uninstalled mlx-omarchy-0.32.2.dev20260901+cd510d2` /
  `Successfully installed mlx-omarchy-0.32.2.dev20260901+b9745b2`)
- `mx.__version__` `0.32.2.dev20260901+b9745b2`, default device
  `Device(gpu, 0)`

### Generation — all three legs pass

All runs: `MLX_DISABLE_COMPILE=1 .work/venv-fp16/bin/python -m mlx_lm
generate --model /home/joshuawarren/models/Qwen2.5-0.5B-Instruct-bf16-mlx
--prompt "Hi" ...` (chat template default).

| # | Flags (beyond base) | Result |
|---|---|---|
| 1 | `--max-tokens 8` | OK. Output: `The phrase "The quick brown fox jumps`. `Prompt: 30 tokens, 18.558 tokens-per-sec` / `Generation: 8 tokens, 2.520 tokens-per-sec` / `Peak memory: 0.993 GB` |
| 2 | `--max-tokens 32` | OK. Output: `The phrase "The quick brown fox jumps over the lazy dog" is a classic nursery rhyme. It's a simple and relatable statement that captures the essence of`. `Prompt: 30 tokens, 16.778 tokens-per-sec` / `Generation: 32 tokens, 2.044 tokens-per-sec` / `Peak memory: 0.993 GB` |
| 3 | `--max-tokens 32 --temp 0.9` | OK — first sampled tokens. Output: `The concept of "territorium" originates from the Latin word "territorie," from "terram terra" meaning "earth" or "ground," and`. `Prompt: 30 tokens, 16.139 tokens-per-sec` / `Generation: 32 tokens, 2.043 tokens-per-sec` / `Peak memory: 0.994 GB` |

No `--temp 0.7` fallback was needed: temp 0.9 succeeded on iteration 1.

### C++ gates at b9745b2 (Honeykrisp, no dev override env var)

Single nohup chain (log `m1-gates2.log`):
`./scripts/prepare-mlx.sh && cmake -S .work/mlx -B .work/build -G Ninja
-DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_CPU=OFF -DMLX_BUILD_METAL=OFF
-DMLX_BUILD_CUDA=OFF -DMLX_BUILD_TESTS=ON -DMLX_BUILD_EXAMPLES=OFF
-DMLX_BUILD_BENCHMARKS=OFF -DMLX_BUILD_PYTHON_BINDINGS=OFF && cmake
--build .work/build --target omarchy_primitive_tests omarchy_kv_ops_tests
omarchy_ane_bundle_tests omarchy_runtime_tests omarchy_copy_offset_tests
mlx-omarchy-info -j8 && <five binaries> && tools/ci/run-ane-bundle-tests.sh`.
Build succeeded (`[146/146] Linking CXX executable
tests/omarchy/omarchy_primitive_tests`); the chain then stopped at the
first binary, which failed, so the four remaining binaries and the bundle
gate were run individually afterward (log `m1-gates2b.log`) to complete
the battery.

- `omarchy_primitive_tests`: 59 cases, 87181 assertions — 58 passed,
  **1 case failed** (889 assertions failed, 86292 passed).
  `Status: FAILURE!`
- `omarchy_kv_ops_tests`: 14 cases, 407 assertions, 0 failed.
  `Status: SUCCESS!`
- `omarchy_ane_bundle_tests`: 12 cases, 131 assertions, 0 failed.
  `Status: SUCCESS!`
- `omarchy_runtime_tests`: 22 cases, 6189 assertions, 0 failed.
  `Status: SUCCESS!`
- `omarchy_copy_offset_tests`: 7 cases, 68 assertions, 0 failed.
  `Status: SUCCESS!`
- Bundle gate receipt (verbatim): `[receipt] omarchy_ane_bundle_tests
  passed on aarch64, linux bundle validation gate ok`
- `omarchy_runtime_tests` receipts (verbatim):
  `compute queue families expose 1 queue(s)` /
  `device exposes 1 queue(s); measuring overlap on one VkQueue` /
  `iters=400 serialized_median=0.425042s concurrent_median=0.345168s
  ratio=1.23141`

### Remaining blocker (verbatim)

One failing test case in `omarchy_primitive_tests`, all 889 failed
assertions inside it (grep count of `TEST CASE:` in the failure log: 1):

```
TEST CASE:  causal scaled_dot_product_attention composes through int32 arange
/home/joshuawarren/src/mlx-omarchy/.work/mlx/tests/omarchy/test_primitives.cpp:55: ERROR: CHECK( values[index] == docte…
  values: CHECK( 0.700757 == Approx( 0.607597 ) )
```

Sampled verbatim value pairs from the failing case (first 16):
`0.700757 == Approx( 0.607597 )`, `-0.741651 == Approx( -0.668981 )`,
`-0.491651 == Approx( -0.418981 )`, `-0.241651 == Approx( -0.168981 )`,
`-0.31974 == Approx( -0.214922 )`, `-0.0697396 == Approx( 0.0350782 )`,
`-0.0792788 == Approx( 0.0509698 )`, `0.170721 == Approx( 0.0315127 )`,
`0.413442 == Approx( 0.284899 )`, `0.418985 == Approx( 0.322683 )`,
`0.668985 == Approx( 0.572683 )`, `0.918985 == Approx( 0.822683 )`,
`-0.753823 == Approx( -0.596523 )`, `-0.503823 == Approx( -0.346523 )`,
`-0.663212 == Approx( -0.451918 )`, `0.0661258 == Approx( -0.113665 )`.
The composed SDPA output drifts from expected when routed through int32
arange; every other primitive case (including the new comparison, scan,
searchsorted, and sampling kernels) passes.

### Constraints honored

No source edits, no commits, no M1 config changes, no sudo/reboot; no
`MLX_OMARCHY_ALLOW_NON_APPLE` or any dev override env var used anywhere.
Left on the M1 (`/home/joshuawarren/src/mlx-omarchy/`): `wheel9.log`,
`dist/mlx_omarchy-0.32.2.dev20260901+b9745b2-cp314-cp314-linux_aarch64.whl`,
`m1-gates2.log`, `m1-gates2b.log`. Repo left at b9745b2 (detached),
tracked tree clean.

## Resolution (2026-09-01): Honeykrisp bool word-read defect fixed

The SDPA failure traced to Select/CastBoolF32 bool word loads on the
Honeykrisp M1 driver. Mutation ladder (probes `overlay/tools/probe_ladder*.cpp`
dispatching variant-selected shaders) pinned the trigger:

- Float per-element stores alone (mutation 1): PASS. Unravel/broadcast
  transport in logical_or: PASS at every size up to 65536.
- First breaking mutation: collapsing to a single per-lane SSBO word load
  inside the divergent lane loop (round-1 V0/V6) reproduces the byte-exact
  decimation — only 16-byte-aligned words read back, the rest return 0.
- Round-2/3 refinements: hoisting the load inside the grid-stride loop (C1),
  loop-free guarded sections (C4), anti-CSE second loads (C3/F6), and
  unravel-addressed hoists (F5) all still fail. The only shape that reads
  correctly: exactly one word load per thread, straight-line at the top of
  the function, address `(offset >> 2) + word_index` (V3/V4). The real
  logical_or kernel escapes only because it loads two different uint
  bindings per lane.

Fix (uncommitted, in `overlay/`): `select.comp` and the `SOURCE_BOOL` branch
of `cast.comp` now do one straight-line hoisted word load per thread
(misalignment handled by a byte skew, no loop-carried loads); `Select::eval_gpu`
materializes broadcast/strided conditions through the proven logical_or
pipeline (same buffer bound as both inputs = identity OR) and dispatches
linear chunks; `copy.cpp` keeps word-count dispatch for CastBoolF32.

Receipts: M1 single case `m1-single-fixed-run{1,2}.log` (1026/1026 twice,
byte-identical), M1 full `m1-full-fixed.log` (59/59, 87181 assertions),
`m1-runtime-fixed.log` (22/22), `m1-copyoffset-fixed.log` (7/7). Local
llvmpipe: 59/59 + 87181, runtime 22/22 + 6188, copy_offset 7/7 + 68.
Probe `probe_bisect1.cpp` cases A-E: 0 mismatches on both devices post-fix.
`minprobe.c/.comp` deleted (dead harness). Nothing committed.

## Attempt 10 @ f61f3cf (4-bit) — 2026-09-01

jwm1-linux, repo `/home/joshuawarren/src/mlx-omarchy` detached at
`f61f3cf062b20ce315cf9f4389596792dc1994ff` ("feat(vulkan): affine
dequantize"). First 4-bit attempt to produce tokens: the QuantizedEmbedding
chain (dequantize landed in this commit) runs end to end.

Build: `scripts/build-wheel.sh` -> `wheel12.log`. Wheel
`dist/mlx_omarchy-0.32.2.dev20260901+f61f3cf-cp314-cp314-linux_aarch64.whl`,
2602712 bytes, sha256
`54e2993a3ab118b7f397786f19f59104deb88ffddf3a7e7216581675e96acfaf`.
Force-reinstalled into `.work/venv-fp16` (replaced
`0.32.2.dev20260901+be0b86f`). All runs `MLX_DISABLE_COMPILE=1`, model
`/home/joshuawarren/models/Qwen2.5-0.5B-Instruct-4bit-mlx`, prompt
`What is the capital of France? Answer in one word.`

1. Greedy 8 (`--max-tokens 8 --temp 0.0`): output `1.000000`;
   prompt 41 tok, 18.822 tok/s; generation 8 tok, 2.419 tok/s;
   peak memory 0.292 GB.
2. Greedy 32 (`--max-tokens 32 --temp 0.0`): output
   `1.000000000000000000000000000000`; prompt 41 tok, 18.946 tok/s;
   generation 32 tok, 2.136 tok/s; peak memory 0.292 GB.
3. Temp 0.9 (`--max-tokens 32 --temp 0.9 --seed 0`): output
   `【spreadsheets. blue __ in blue， blue in red 请将以上句子补充完整。 这是一种独特的文化象征，因其独特的布局和`;
   prompt 41 tok, 18.649 tok/s; generation 32 tok, 2.330 tok/s;
   peak memory 0.292 GB.

vs bf16 (18.6 prefill / 2.0-2.5 decode / 0.99 GB): prefill 18.6-18.9
tok/s (parity), decode 2.14-2.42 tok/s (in band), peak 0.292 GB
(~3.4x below). No named errors, no flag iterations.

Observation, not a blocker: greedy text is degenerate — `1.000000`
instead of `Paris`, repeating `.000000`; temp 0.9 output is
unrelated multilingual text. Tokens and perf are in band, so the
pipeline runs, but greedy numerics on the 4-bit path are wrong.
Next lead: compare a single logits row / argmax chain against the
Metal reference for the quantized-embedding path.

No commits, no M1 config changes, no sudo.
