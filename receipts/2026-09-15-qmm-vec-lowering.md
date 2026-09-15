# GEMV epilogue fold on jwm1: both gate conditions fail; compile flags were never the difference

Date: 2026-09-15
Task: close the `wave/DecodeEpilogueFold` question on jwm1 — the one host
the 2026-09-14 battery never measured — under the working hypothesis that
the fold broke the Honeykrisp pins because glslc/NIR flags differ from
`fast_norm.comp` / `fast_rope.comp`.

## Verdict

**The fold does not land.** On jwm1 Honeykrisp it saves exactly the priced
72 dispatches (249 → 177 vk/token, 2 → 1 submissions, fold-off restores
249) and holds the short pin, but flips the ctx1024 digest to the same
deterministic value jw16 produced, and decode tok/s does not rise:

| arm | vk/token | submissions | short pin `7fd25a869ff21678` | ctx1024 pin `7da83f06ec9f001d` | short decode |
|---|---|---|---|---|---|
| base `b79a4b68` | 249 | 2 | holds | `7da83f06ec9f001d` | 111.0 tok/s |
| cand `50870b69` (fold on) | 177 | 1 | holds | **`31267e7ed4c6d0dc`** | 109.2 tok/s |
| cand, `MLX_OMARCHY_FOLD_EPILOGUE=0` | 249 | — | holds | `7da83f06ec9f001d` | — |

The cand ctx1024 divergence is bit-identical to jw16's five-run result
(`31267e7ed4c6d0dc`, first ids `13060,498,369` match, argmax flips by
token 27), and the fold-off arm restores the pin on the same wheel, so
this is the fold, not the host or the wheel build. Pins fail AND
tok/s falls — both acceptance conditions fail on their own.

## The flags hypothesis is false

Every shader — `qmm_vec.comp`, `fast_norm.comp`, `fast_rope.comp`,
`elementwise.comp` — compiles through the same CMake function
`omarchy_shader` (`overlay/mlx/backend/omarchy/CMakeLists.txt:19-39`):
one invocation, `glslc -O --target-env=vulkan1.3 <source> -o <spv>`
(or `glslangValidator -V -Os` when glslc is absent), differing only in
the per-variant `-D` defines. There is no flag delta to equalize.

The alternative acceptance path — fast_rope's GLSL as a separately
compiled SPIR-V entry point running inside the GEMV dispatch — is not
available: one `vkCmdDispatch` executes one pipeline with one entry
point of one module; a second entry point would need a second dispatch,
which defeats the fold.

## The remaining Mesa hole (named)

Honeykrisp (Asahi AGX) NIR lowers the exp → cos/sin → product-subtract
rotation chain differently depending on the compilation unit it lands
in: the identical source-level trig chain rounds differently inside
`qmm_vec.comp` than inside `fast_rope.comp` (2026-09-14 receipt,
defect 2: hardened offset sweep fails 23–32 elements per run at 1–2 f16
ulps on jw16, host fp64 oracle reproduces neither arm, GEMV outputs and
sums bit-exact). jwm1 now confirms the same lowering on the second M1
Max — the divergence value is identical across both hosts. The lowering
is inside the driver's shader compiler and outside GLSL source control;
no source arrangement (shared function, exchanged bits through shared
memory — fixed in `eab327a8` —, matching flags) removes it. Fixing this
requires a Mesa-side change: make AGX transcendental scheduling /
approximation-bit selection invariant across compilation units (or
expose a guarantee that identical operation sequences lower identically
per module), then re-run this battery.

## Identity

- Base: `b79a4b68` (fold parent, main-land2 line), wheel
  `mlx_omarchy-0.32.2.dev202609151921+b79a4b68-cp314-cp314-linux_aarch64.whl`
  sha256 `a6b67549b951887696af373b15d26b6a3fc147a9af88019d92f0092774e87882`.
- Cand: `50870b69` (`wave/DecodeEpilogueFold`), wheel
  `mlx_omarchy-0.32.2.dev202609151923+50870b69-cp314-cp314-linux_aarch64.whl`
  sha256 `7bab10528439debdabd9d4e8f178ab359f0b46e78eadd59351c8681eccc11dca`.
- Host: jwm1-linux (M1 Max, Honeykrisp), Python 3.14.7, provenance
  `verified=match` on every arm.
- Every GPU step ran under one `flock` hold on `/tmp/m1-gpu.lock`
  (inode 29 before and after; nested `flock -n` refused; never
  unlinked). `63c1d3cf` not merged; no fold code merged to main.
- Wheels built with the jwm1 canonical `scripts/build-wheel.sh`
  CMAKE_ARGS (`MLX_BUILD_CPU=OFF`), 8 jobs, fresh per-side `.work`.

## Reproduce

```sh
ssh jwm1
flock /tmp/m1-gpu.lock bash /var/tmp/DecodeEpilogueFold/run-jwm1.sh
# build: /var/tmp/DecodeEpilogueFold/build-jwm1.sh
```

Artifacts in `jwm1/`: `dispatch-{base,cand,cand-off}.json`,
`digest.txt`, `ab.txt` (A/B aborts at the first ctx1024 digest
mismatch by design — the hard gate), `lock.txt`.
