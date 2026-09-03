# 2026-09-03 — published v0.3.2 wheel vs local builds: the delta is a source gap, not a build setting

Static comparison, done on the x86 dev box against unzipped wheels. No binary in
this receipt was executed; aarch64 objects were inspected with `readelf`,
`strings`, `cmp`, and per-file `sha256`. Compiler flags were read (read-only)
from the two build trees on jwm1 (Tailscale `100.84.184.102`).

## Wheel inventory and sha256

| wheel | source generation | bytes | sha256 |
|---|---|---|---|
| v0.3.2 published asset `dev202609030512` | `b17cc86` (release tag) | 6789976 | `61424114d983c8f3c02b6ae83125be9fe709a13a2fcb8b27ff0b0d02fb64e370` |
| local `dev202609030928` (`~/src/mlx-omarchy-decomp/dist/`) | `ceae628` (perf commit) | 6796874 | `63eece27f9901a73e7acfc62cde7e8feeead3f87b9d06740cc00322fbd6fcd9d` |
| local `dev20260903` (`~/src/mlx-omarchy/dist/`, the "README wheel") | pre-`f449ed2` stale generation, see below | 2872489 | `6e54ab2b8e808b78d052eae0d5dc76e4f4111f2afac3e572e9eec243d8c5f42c` |

Published asset downloaded with `gh release download v0.3.2`; hash matches
`receipts/2026-09-03-release-v0.3.2.md` byte for byte. Local hashes match the
inventory in `receipts/2026-09-03-decode-ab-and-affinity-jwm1.md`.

## Full difference list: published `0512` vs ceae628-era local `0928`

Same file inventory in both wheels (307 non-pyc files) except:
`gpu_profiler.h` exists only in `0928` (new in `ceae628`, compile-gated OFF, a
packaging-only presence in `include/`). `dist-info/` names differ by the dev
version segment only; METADATA is identical except the `Version:` line.
Content differs (sha256) in exactly six payload files:

- `mlx/lib/libmlx.so` — 20,924,000 vs 20,923,440 bytes (560-byte delta)
- `mlx/bin/mlx-omarchy-info` — embeds build info
- `mlx/include/mlx/backend/omarchy/device.h`, `encoder.h`, `vulkan.h` — the
  `ceae628` header changes, shipped as installed headers
- `mlx/share/cmake/MLX/MLXConfigVersion.cmake` — embeds the version string

`mlx/core.cpython-314-aarch64-linux-gnu.so` is byte-identical between the two.
Both `libmlx.so`: stripped, zero `.debug` sections, identical NEEDED set
(including `libopenblas.so.0`), 2 assert-class strings each, **0** profiler
strings each — profiling compiled out of both.

## Build-setting candidates, each eliminated

- Flags: `flags.make` from both build trees (`~/wt-rel031/.work/mlx` for the
  release, `~/src/mlx-omarchy-decomp/.work/mlx` for ceae628) is byte-identical:
  `CXX_FLAGS = -O3 -DNDEBUG -std=gnu++20 -fPIC -fvisibility=hidden -fvisibility-inlines-hidden -Wno-psabi`.
  Same build type, same NDEBUG, no LTO difference to look for (neither uses it).
- Pin: `git diff b17cc86 ceae628 -- mlx.lock setup.py` is empty. Same MLX
  0.32.2 tarball `1f8e74e3`.
- Machine/toolchain: both built on jwm1 the same morning, 05:12 and 09:28.
- Profiling: 0 profiler strings in both libraries; `--diagnostics` did not
  exist at `b17cc86` and is opt-in at main.
- `DEV_RELEASE=1` in the release invocation changes only the version string
  (`patches/mlx-version-time.patch`).

## Named cause

**Source gap, not build settings.** The tag `v0.3.2` was cut at `b17cc86`
(2026-09-03 05:02). `ceae628` — "perf: retire the per-dispatch host join and
the descriptor pool churn", the submission ring + descriptor cache, recorded
in its own message as "+9 percent tokens per second" — landed at 05:18, sixteen
minutes later, and is not in the release. The only same-generation local build
(`0928`) is exactly `b17cc86 + ceae628`: identical flags, pin, machine, section
layout, NEEDED set, and a 560-byte library delta. The measured decode deficit
of the published asset against that wheel (bf16 -11.3%, 4-bit -8.7%) matches
the commit's own +9% claim. The static evidence explains the measurement up to
the last digit that statics can reach: the 560-byte `.so` delta cannot be
proven here to be the ring/cache binary without executing aarch64 code, but
every alternative cause (flags, NDEBUG, pin, machine, profiling, strip state,
Python bindings) is eliminated by the hashes above.

## Correction of record: the `dev20260903` "README wheel" is a different generation

`receipts/2026-09-03-decode-ab-and-affinity-jwm1.md` states the `dev20260903`
and `dev202609030928` wheels are byte-identical in all non-pyc files. That is
false. `libmlx.so` differs: 5,300,664 bytes in `dev20260903` vs 20,923,440 in
`0928`, no `libopenblas` NEEDED, `encoder.h` at its pre-`ceae628` content, no
`gpu_profiler.h`, old date-only version scheme. `f449ed2` ("feat: build the CPU
backend", 2026-09-02 18:35) is the generation boundary; `dev20260903` predates
it despite its 06:06 build time on 09-03 — it was built from a stale tree.
Its decode numbers (2.002 / 3.847) come from a different backend composition
and are not evidence about the v0.3.2 delta. The README 8-core column measures
that stale-generation wheel.

## Verdict and action

- `scripts/build-wheel.sh` has **no defect**: the release path and the default
  dev path produce identical optimization settings. No change shipped.
- The published v0.3.2 is slow because it predates `ceae628`. The fix users
  need is the next release cut from current main (contains `ceae628`); users
  can recover the ~9-11% today by installing a wheel built from main.
- No rebuilt-wheel measurement is owed: no build-script change was made.
