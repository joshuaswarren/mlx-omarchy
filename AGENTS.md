# Agent instructions

## Start here

Read these files before changing code:

1. `README.md`
2. `docs/architecture.md`
3. `docs/roadmap.md`
4. `docs/compatibility.md`
5. `docs/plans/2026-08-29-mlx-omarchy-ane-compatibility-plan.md`

Run `scripts/prepare-mlx.sh` before inspecting the pinned MLX source.
Inspect the affected backend under `.work/mlx`.
Keep project-owned files under `overlay/` and upstream edits under `patches/`.

Use the CUDA backend for the complete GPU boundary.
Use `mlx/backend/no_cpu/` and `mlx/backend/no_gpu/` for failure behavior.

## Fixed product contract

- Keep the public MLX Python import as `mlx`.
- Keep `mx.gpu` as the public accelerator device.
- Use Honeykrisp Vulkan for complete tensor execution.
- Treat ANE as an internal graph-region accelerator.
- Let CPU code schedule, copy, tokenize, and serve requests.
- Never run a tensor primitive on CPU in a release build.
- Return an exact compatibility error when Vulkan and ANE cannot run a primitive.
- Do not emulate Metal on Linux.
- Qualify M1 Omarchy before other Apple Silicon models.

Do not weaken these rules to make a model run.
Add the missing accelerator implementation or keep the test failing.

## Repository boundaries

- This repository owns the Omarchy backend, ANE partitioner, patches, packaging, tests, and releases.
- Upstream MLX source and history must not enter this repository.
- `joshuaswarren/ane-linux-experiments` owns hardware probes, format research, and unstable fixtures.
- `eiln/ane` owns the ANE DRM driver and `libane` ABI.

Do not copy experimental driver code here.
Prove a driver change in the experiment repository, then send the smallest stable ABI change to `eiln/ane`.

## Upstream MLX sync

Pin the upstream release, commit, archive URL, and SHA-256 in `mlx.lock`.
Fetch upstream source only through `scripts/prepare-mlx.sh`.
Add new project files under `overlay/`.
Keep modifications to existing MLX files in a small patch series.
Rerun runtime, primitive, and semantic gates after every baseline change.
Keep Omarchy work behind `MLX_BUILD_OMARCHY`.
Do not rename public MLX APIs to fit backend code.
Resolve an upstream change at the shared MLX contract before adding an Omarchy-only path.

This repository is independent from Apple.
Do not open an upstream MLX pull request or push to an Apple remote from an agent session.

## Implementation order

Work in the dependency order in `docs/roadmap.md`.
The Vulkan baseline does not wait for ANE.
ANE integration waits for the connected full-graph parity receipt.
The ANE exporter also waits for a hand-authored MIL compiler proof.
The exporter does not wait for dma-buf support.
General MLX lowering waits for the one-operation MIL proof and the stable `libane` ABI.

Implement primitives in model-driven order, but keep each primitive general.
Do not add a model-name branch inside a core tensor kernel.
Reuse backend-neutral MLX code before adding Omarchy code.
Run the U2 matmul and attention speed spike before broad primitive work.
Stop and revise the backend design if the spike misses its fixed go or no-go gate.
Partition ANE regions before `mx.compile` fusion.
Represent each selected region as an opaque `AneRegion` primitive.
Send the rest of the graph through Vulkan fusion.

## Hardware safety

New descriptor, synchronization, and dma-buf tests can reset the device or machine.
Use a bounded timeout for every new hardware path.
Run one new failure mode at a time.
Record the device state before and after the test.
Run ANE submits in a bounded worker that owns the device file descriptor and resident buffer objects.

Do not enable dma-buf mode by default until export, import, coherency, fencing, and recovery tests pass.
Use shared host staging buffers between the evaluator and worker.
Include staging copies and process IPC in every ANE crossover result.
Do not redistribute Apple private frameworks, compiler binaries, firmware, credentials, or model weights.

## Verification and receipts

A code claim needs command output from the same session.
A hardware claim also needs a receipt file.
Each hardware receipt must include:

- source commit
- kernel and Mesa versions
- Vulkan device and firmware identity
- ANE driver and compiler identity when applicable
- model and quantization hashes
- exact command
- numerical result
- backend dispatch trace
- timing and thermal procedure for performance claims
- Vulkan device reopen result or ANE worker, reboot, bring-up, and self-test recovery result

Generate `docs/compatibility.md` status from receipts when tooling exists.
Until then, update a row only when the linked receipt proves every named gate.

## Test rules

- Add one focused test for each new observable contract.
- Run the release-equivalent build with CPU primitive evaluation unavailable.
- Trace every ecosystem workflow at least once and require zero CPU tensor dispatches.
- Compare numerical output with a pinned macOS MLX build or locked fixture.
- Use the plan's exact Qwen reference model for the 32-token contract.
- Use pinned numerical tolerances and fixture-specific argmax or top-k checks for other models.
- Compare performance only on the same machine, model, quantization, prompts, and thermal procedure.
- Keep cold-start, warm steady-state, prefill, decode, and first-token results separate.
- Trace recurrent ANE state reuse and residency invalidation across decode steps.

Do not lower tolerances, shorten a stability run, or remove a failing workload to make a gate pass.

## Documentation

Update `docs/architecture.md` when a backend or memory boundary changes.
Update `docs/roadmap.md` only when a proof gate changes.
Update `docs/compatibility.md` from evidence, not intent.
Delete stale paths and replaced design text in the same change.
