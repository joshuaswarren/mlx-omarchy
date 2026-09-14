# Mel frontend SPIR-V disk cache

Date: 2026-09-14

## Verdict

`cached_compile` now keeps `glslc -O` SPIR-V on disk. On `jwm1-linux` the Parakeet mel frontend's first call in a new process drops from 9858.6 ms without the cache to 188.1 ms on a cache hit. Steady-state stays ~170 ms. Mel, mask, and encoder features stay byte-identical to the macOS golden on every arm. A separate-process cache hit writes the same `.spv` bytes as a cold compile.

The 9.8 s cost was the SPIR-V optimizer, not GPU work. Mesa already hid AGX codegen. The disk layer is the missing step before Mesa.

## Identity

- Landed commit: `a3360a0808ba1b54e6209d4e65cfcd66bc4e19b0` (`agent/mel-spirv-cache`, rebased onto `origin/main` `011a2134`).
- Overlay is byte-identical to `05015a76`; `git diff 05015a76 a3360a08 -- overlay/mlx/backend/omarchy/custom_kernel.cpp tests/custom_kernel_smoke.py` is empty.
- Did not rebase over TdtStackIntegration's projector pick (`38d5ebfb` is not on main).
- `custom_kernel.cpp` SHA-256: `44aadab049a9cffc59ef4667d080fbb0baa113c15bc79697535a9dfb0ec54e9c`
- `tests/custom_kernel_smoke.py` SHA-256: `f05b53c9d4075b8a0202b434fe842a1289978d0a8aca902c2eb7c5ddd1d013f5`
- Cache wheel: `mlx_omarchy-0.32.2.dev202609141626+05015a76-cp314-cp314-linux_aarch64.whl`
  SHA-256 `e0d513eb15c8698843d8afed0096ce775e80363913a6184450fe83b705d3d333`
- Baseline wheel: `mlx_omarchy-0.32.2.dev202609141629+b79a4b68-cp314-cp314-linux_aarch64.whl`
  SHA-256 `a3f698345c113330d9c9799162d9d3e5c4e2d05e7f5a1bdf6370c28c5a4d335b`
- Host: `jwm1-linux`; Apple M1 (G13G B1); Honeykrisp; `Device(gpu, 0)`; Python 3.14.7
- `glslc --version`: 2026.3 / SPIR-V 1.4.357.0
- Target env: `vulkan1.3` (`-O --target-env=vulkan1.3`)
- Lock: `/tmp/m1-gpu.lock` inode 29, `flock -n`, not stolen, not unlinked. Held 2026-09-14T17:06:03Z–17:06:45Z UTC. Nested lease removed. Post-run `flock -n` free.
- Agent model: `xai-oauth/grok-4.6`; fallback: true (not OpenAI).

## Cache contract

The disk key is `sha256` over GLSL (translated source including the `#version`/`#extension` preamble), `glslc --version`, invocation flags, compiler path, and target env. Root is `$XDG_CACHE_HOME/mlx-omarchy/spirv` or `$HOME/.cache/mlx-omarchy/spirv`. `MLX_OMARCHY_SPIRV_CACHE` relocates the directory; empty or `0` disables it. Writes are tmp+rename. A damaged magic or SPIR-V header falls back to compile.

Disable and identity tests live in `tests/custom_kernel_smoke.py`.

## Protocol

One exclusive GPU lock. Fresh process per arm. Same 10.435 s LibriSpeech fixture. `vulkan_mel.py` SHA-256 `4f61cd5cd1ebabb2a96d3270e347d3a6607be9b9c186f56663c5f9d14ba095c0`.

| Arm | Wheel | Env |
| --- | --- | --- |
| base | b79a4b68 | no disk cache in the binary |
| cache-off | 05015a76 | `MLX_OMARCHY_SPIRV_CACHE=0` |
| cache-cold | 05015a76 | empty private cache dir |
| cache-warm | 05015a76 | same dir, new process |
| cache-warm2 | 05015a76 | same dir, third process |
| cache-corrupt | 05015a76 | one `.spv` overwritten with `XXXX` |

## Results

`first_call_ms` is pass 0 of a new process (compile plus compute). `steady_ms` is a later whole `extract_chunk_features` call.

| Arm | first_call_ms | steady_ms | finals bit-exact | stages bit-exact |
| --- | ---: | ---: | --- | --- |
| base | 9858.6 | 172.4 | true | true |
| cache-off | 9760.9 | 177.5 | true | true |
| cache-cold | 9765.5 | 170.8 | true | true |
| cache-warm | 188.1 | 170.2 | true | true |
| cache-warm2 | 197.4 | 165.0 | true | true |
| cache-corrupt | 493.5 | 167.7 | true | true |

Every arm: 8 Vulkan compute dispatches, 13 GPU primitives, 1 submission. Mel golden SHA-256 `4ed24d7da64c17f58419ec45f15aa24ac9d1cfd93a4de51ee4f05e1d712b4d22`. All 13 intermediate stages matched the baseline dump.

Cache-warm first call is 52.4× faster than base (9858.6 → 188.1 ms). The earlier 400 ms figure was a `glslc` shim, not this binary.

After truncating one of eight entries, the next process recompiled that kernel (493.5 ms) and restored a valid SPIR-V file whose bytes matched the pre-corruption copy. The other seven entries stayed hits.

## Cache-hit identity test

On the same lock, with the cache wheel:

```text
test_spirv_cache_hit_serves_the_cold_compile_byte_for_byte ... ok
test_spirv_cache_is_disabled_by_the_environment ... ok
Ran 2 tests in 0.577s
OK
```

Two independent cold compiles produced one identically named `.spv` with identical bytes. A third process against the first directory did not change `st_mtime_ns`. `MLX_OMARCHY_SPIRV_CACHE=0` wrote no entries.

Cold compile of the eight mel kernels left 8 files under the private cache directory.

## Measurement files

| File | SHA-256 |
| --- | --- |
| measurements/base.json | `772e5d18a287988ce0d5199cd37393157a653a62fb90b770172db01d8ef93ca7` |
| measurements/cache-off.json | `86a065f2139e72471d7c9128882becb0bab030a844e8a70f9e36eb4855eecaf0` |
| measurements/cache-cold.json | `e87d33bd63cc3e27a8e8fee2c008eec14373c1b25cbd7c7ae178690bcb6ed398` |
| measurements/cache-warm.json | `c05d65d7e712903136013c47b62caf6cda773cdc20709c8822e12c1b0ba25353` |
| measurements/cache-warm2.json | `10d5a10c114b34ca1c7a99760385fe169421322860b58fde6b1ddf3eeec186af` |
| measurements/cache-corrupt.json | `06e5146b201e7f8ded3fad0611d295d6fbc9d3a626a20f426582f5738d2dfa11` |
