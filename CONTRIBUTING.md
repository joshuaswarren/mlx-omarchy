# Contributing

`mlx-omarchy` still needs more MLX features for real models.
You can help with small backend changes and M1 tests that others can repeat.

## Before you start

1. Read `docs/architecture.md` and `docs/roadmap.md`.
2. Check `docs/compatibility.md` for the current gate.
3. Run `scripts/prepare-mlx.sh` to create `.work/mlx`.
4. Open an issue before you change a public API, driver ABI, or bundle format.
5. Keep private Apple software, firmware, keys, and model weights out of the repository.

## Source rules

- Add new Omarchy files under `overlay/`.
- Put changes to current MLX files in `patches/`.
- Do not copy the MLX source into this repository.
- Update `mlx.lock` only with an exact commit and checked archive hash.
- Keep Omarchy changes behind `MLX_BUILD_OMARCHY`.

## Code rules

- Keep the public `mlx` import and `mx.gpu` device.
- Use Vulkan for all tensor work.
- Keep ANE choice inside the GPU evaluator.
- Never add a silent CPU tensor fallback.
- Reuse shared MLX code before you add Omarchy-only code.
- Keep model rules out of core operations.

## Proof required

Name the changed contract in each pull request.
Add the smallest test that proves the change.

For a hardware result, record these facts:

- source commit
- kernel
- Mesa
- device
- firmware
- command
- model hash
- output
- dispatch trace

For a speed result, also record these facts:

- warmup
- repeat count
- memory use
- temperature
- test tool and settings

Do not lower a tolerance or remove work to make a check pass.
Do not claim M1 results for a different Apple Silicon model.

## Driver changes

Test early driver work in [`ane-linux-experiments`](https://github.com/joshuaswarren/ane-linux-experiments).
Send stable DRM and `libane` changes to [`eiln/ane`](https://github.com/eiln/ane).
Do not copy a private driver fork into this repository.

## License

Contributions use the repository's MIT license.
Keep upstream copyright notices in patch context and prepared source.
