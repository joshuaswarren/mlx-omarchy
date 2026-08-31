# M1 development gate receipt, 2026-08-31

Run on the physical target, not a development override.

## Identity

- Host: `jwm1-linux` (13-inch M1 MacBook Pro, `MacBookPro17,1`)
- Kernel: `Linux jwm1-linux 7.1.6-1-1-ARCH aarch64` (Asahi)
- Repository commit: `312c32a` on `feat/vulkan-primitives`
- Device: `Apple M1 (G13G B1)`, driver `Mesa Honeykrisp`, Vulkan `1.4.354`
- Capabilities: `shader_float16=1`, `shader_int16=1`, `storage_buffer_16bit_access=1`
- `non_apple_dev_override=0`

## Commands and results

Build and test (from the repository root):

```sh
./scripts/prepare-mlx.sh
cmake -S .work/mlx -B .work/build -G Ninja -DMLX_BUILD_OMARCHY=ON \
  -DMLX_BUILD_CPU=OFF -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF \
  -DMLX_BUILD_TESTS=ON -DMLX_BUILD_EXAMPLES=OFF -DMLX_BUILD_BENCHMARKS=OFF \
  -DMLX_BUILD_PYTHON_BINDINGS=OFF
cmake --build .work/build --target omarchy_primitive_tests \
  omarchy_runtime_tests omarchy_copy_offset_tests mlx-omarchy-info -j8
./.work/build/tests/omarchy/omarchy_primitive_tests
(cd .work/mlx && MLX_OMARCHY_BUILD_DIR=../build tools/ci/run-omarchy-runtime.sh)
./tools/ci/run-ane-bundle-tests.sh
```

Results:

- `omarchy_primitive_tests`: 17 cases, 1273 assertions, 0 failed.
  Covers elementwise, suffix reduce, dense and transposed Matmul, AddMM,
  FP16 and emulated BF16, casts, strided copy, scalar fill, grad, jvp,
  vmap, compile no_fuse, and named exclusion errors.
- `omarchy_runtime_tests`: 22 cases, 6189 assertions, 0 failed.
- `omarchy_copy_offset_tests`: 7 cases, 68 assertions, 0 failed.
- `omarchy_ane_bundle_tests`: 12 cases, 131 assertions, 0 failed
  (`linux bundle validation gate ok` on aarch64).
- `mlx-omarchy-info --trace-smoke`: buffer round trip OK on the device.

## Wheel and clean install

```sh
./scripts/build-wheel.sh
./tools/ci/run-clean-omarchy-install.sh
```

- Wheel: `mlx_omarchy-0.32.2.dev20260831+7b709df-cp314-cp314-linux_aarch64.whl`
- sha256: `e83195d975d8b0088c3657f348e0b14b9001a6fc7fbf5b82f2cdda08f317f6a9`
- Clean-install smoke (fresh venv):

```text
[receipt] import mlx.core: ok, default device Device(gpu, 0), Apple M1 (G13G B1)
[receipt] add: ok
[receipt] matmul: ok
[receipt] value_and_grad: ok, gradient matches host expectation within 1e-4
clean install verified
```
