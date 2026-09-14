# Receipt: packaged install on origin/main

Host-only. Not a hardware pass. No GPU lock, no ANE execute, no fleet
access, no merge.

`install.sh` is the aarch64 Omarchy installer. This host is x86_64, so
that script refuses before it writes. The documented development path
(`docs/install-omarchy.md`: build wheel, then a clean prefix / CI venv)
is what this receipt proves.

## Source

- Repository: `joshuaswarren/mlx-omarchy`
- Commit: `e8a2696584be327a7c4b552d2a7825dfb3c07362` (`origin/main`)
- Worktree: `~/.config/superpowers/worktrees/mlx-omarchy/PackagedInstall`
- Host: `omp-studio-local` `Linux 6.17.2-1-pve` `x86_64`
- Python: 3.11.2

## Prefix

```
/tmp/mlx-omarchy-packaged-install-nVkvdb
```

Empty directory from `mktemp -d`. Wheel installed only into
`$PREFIX/venv`. Launcher `$PREFIX/bin/mlx-omarchy-info` wraps the
wheel's `mlx/bin/mlx-omarchy-info`, same as `install.sh`.

## Commands

```sh
./scripts/build-wheel.sh
python3 -m venv "$PREFIX/venv"
"$PREFIX/venv/bin/pip" install dist/mlx_omarchy-0.32.2.dev202609141905+e8a26965-cp311-cp311-linux_x86_64.whl
MLX_OMARCHY_ALLOW_NON_APPLE=1 \
  VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json \
  "$PREFIX/venv/bin/python" -c 'import mlx.core as mx; print(mx.__version__, mx.default_device())'
"$PREFIX/bin/mlx-omarchy-info" --check-bundle \
  receipts/2026-09-14-decoder-activations-on-ane/bundle-tanh
MLX_OMARCHY_WHEEL=$PWD/dist/mlx_omarchy-0.32.2.dev202609141905+e8a26965-cp311-cp311-linux_x86_64.whl \
  MLX_OMARCHY_ALLOW_NON_APPLE=1 \
  ./tools/ci/run-clean-omarchy-install.sh
```

## Wheel

```
mlx_omarchy-0.32.2.dev202609141905+e8a26965-cp311-cp311-linux_x86_64.whl
size: 8026694 bytes
sha256: 8c0f649ab0bc3d06061458422b9264a3ed386610451414bc6ad244c352e56322
contains: mlx/bin/mlx-omarchy-info, mlx/bin/mlx-omarchy-coreml, mlx/lib/libmlx.so
```

No `mlx-omarchy-ane-worker` (host build, `MLX_OMARCHY_ANE_DEVICE` off).

## Import

```
mlx 0.32.2.dev202609141905+e8a26965 default_device=Device(gpu, 0) type=DeviceType.gpu
add [4.0, 6.0]
```

Device is llvmpipe (development device, not Honeykrisp).

## CI clean install

```
[receipt] wheel: mlx_omarchy-0.32.2.dev202609141905+e8a26965-cp311-cp311-linux_x86_64.whl
[receipt] import mlx.core: ok, default device Device(gpu, 0), llvmpipe (LLVM 15.0.6, 256 bits) (development device, not Honeykrisp)
[receipt] add: ok
[receipt] matmul: ok
[receipt] value_and_grad: ok, gradient matches host expectation within 1e-4
clean install verified
```

## check-bundle OK

Fixture: `receipts/2026-09-14-decoder-activations-on-ane/bundle-tanh`

```
[receipt] bundle: .../receipts/2026-09-14-decoder-activations-on-ane/bundle-tanh
[receipt] graph: schema4-tanh-512
[receipt] graph_hash: de23b14cc252e0431282d4a25ee62a76ffb7a84028c88605a80cfc554c81fc00
[receipt] task_descriptors: 1
[receipt] input x: index=0 dtype=float16 shape=[1,512,1,1] byte_size=1024 stride=32768
[receipt] output y: index=0 dtype=float16 shape=[1,512,1,1] byte_size=1024 stride=32768
[receipt] logical_result 0 y: dtype=float16 shape=[1,512,1,1] tensor=y element_offset=0 element_count=512 conversion=identity
[receipt] state: none
[receipt] intermediate: none
[receipt] dispatch_plan: 0
[receipt] dispatch 0: program=0 payload=program-0.anec operation=tanh encoder=h13-oracle-parity scratch_bytes=0 payload_size=2688 td_size=628 td_count=1 sources=1 destinations=1
[receipt] anec input x: channel=5 logical_bytes=1024 allocation_bytes=32768 element_offset=0 element_count=512 physical_elements=512 nchw=[1,512,1,1,64,64]
[receipt] anec output y: channel=4 logical_bytes=1024 allocation_bytes=32768 element_offset=0 element_count=512 physical_elements=512 nchw=[1,512,1,1,64,64]
[receipt] payload anec: program-0.anec sha256=fcd9620cea5aace87096b3be3de1b28c66adf3d3d091eb69b79dd4e93399e8a2 byte_size=6784
[receipt] compiler: host_build=Linux 6.17.2-1-pve x86_64 toolchain=mil-hwxc 417554c3e22de9124623765717acee25e9860f19 sha256:d86bebb25b2b5023b3071b612654c42c79a1b93f51c471e07cfc1c1df82520c2 target=h13
[receipt] driver_abi_major: 1
[receipt] provenance: repo=mil-hwxc commit=417554c3e22de9124623765717acee25e9860f19
[receipt] OK: bundle valid
```

Exit 0. `load_bundle` only; no Vulkan or ANE device opened for this
command.

`receipts/2026-09-14-decoder-activations-on-ane/bundle-sigmoid` also
exits 0.

## install.sh on this host

```
error: mlx-omarchy runs on Apple Silicon (aarch64); this machine is x86_64.
install_sh_exit=1
```

No files written under `$PREFIX` by `install.sh`. Expected.

## Not claimed

- Not Honeykrisp. Import used llvmpipe under `MLX_OMARCHY_ALLOW_NON_APPLE=1`.
- `install.sh` was not run on aarch64 / Python 3.14.
- `receipts/fixtures/exported/ane-add-fp16-1x512` (and the other two
  exported add/mul fixtures, plus `mil-oneop-bundle`) fail
  `channel map is positional` on this commit. That is the derived-channel
  loader, not an install failure.
- No ANE worker in this wheel.

## Model

- resolved_model: `xai-oauth/grok-4.6`
- fallback: false
