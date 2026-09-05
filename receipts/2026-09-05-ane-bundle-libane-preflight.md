# 2026-09-05 ANE bundle/libane preflight

## Scope

- Worktree: `/tmp/mlx-openai-ane`
- Branch: `openai/ane`
- Base commit: `00be38a21081bdc8527034fe78f35bb2e58c1697`
- Review model: `openai-codex/gpt-5.6-sol`
- libane reference inspected read-only: `/tmp/eiln-ane` at `fbe40bed471d83aa0c73fd0eb56fdc5366313321`

This work validates bundle files without opening an ANE or Vulkan device. It did not edit or execute `ane-linux-experiments`, use SSH, run on `jwm1`, load a driver, or claim ANE runtime or performance support.

## ABI review

The parser was checked against `libane/ane.h` and `libane/ane.c` at the reference commit and against the paired driver sources. The reviewed contract is:

- packed `struct anec` is `0x6a8` bytes at file offset 0;
- executable bytes begin at file offset `0x1000`;
- libane copies `anec.size` bytes into channel 0;
- driver channel 1 is reserved and must be unbound;
- outputs start at channel `4`, and inputs start at `4 + destination_count`;
- libane allocates a separate tile-aligned bootstrap channel from `td_size`;
- the driver requires nonzero `td_size`, `td_count`, and `tsk_size`, encodes `td_size` in 32-bit words, and places the kernel at the 16-byte-aligned end of the task stream;
- libane tile and untile paths copy 16-bit elements.

## Corrections

The original slice did not fully enforce that ABI. This review corrected the following paths:

- require channel 0 to contain the full executable payload copied by libane;
- require reserved kernel channel 1 to have zero allocation;
- require task-descriptor bytes to fit the loaded payload and use 32-bit word units;
- reject zero task size and task sizes that reach the end of the command channel;
- validate the kernel at `align_up(task_size, 16)`, not at the unaligned task end;
- reject overflowing source/destination channel-count arithmetic;
- validate ANEC tile geometry with libane's actual 16-bit copy width and reject other manifest dtypes on ANEC-bound inputs, outputs, and state;
- remove the false requirement that manifest workspace cover libane's separate bootstrap allocation;
- decode the packed header explicitly as little-endian rather than copying bytes into native integers;
- pass the exporter target as an explicit `--target` argument instead of a hidden `h13` constant.

`mlx-omarchy-info --check-bundle` now prints the parsed ANEC envelope and each bound source and destination channel. The bundle documentation states that this is a device-free preflight, not runtime qualification.

## Verification

### Scoped build

```sh
cmake --build .work/build --target omarchy_ane_bundle_tests mlx-omarchy-info -j1
```

Result: both targets built successfully. The build compiled the changed bundle parser, test source, and bundle-reporting tool.

### Bundle regression tests

```sh
./.work/build/tests/omarchy/omarchy_ane_bundle_tests
```

```text
[doctest] test cases:   26 |   26 passed | 0 failed | 0 skipped
[doctest] assertions: 2709 | 2709 passed | 0 failed |
[doctest] Status: SUCCESS!
```

### Existing exported fixtures

```sh
.work/build/tools/mlx-omarchy-info/mlx-omarchy-info --check-bundle receipts/fixtures/exported/ane-add-fp16-1x512 && \
.work/build/tools/mlx-omarchy-info/mlx-omarchy-info --check-bundle receipts/fixtures/exported/ane-add-fp16-1x896 && \
.work/build/tools/mlx-omarchy-info/mlx-omarchy-info --check-bundle receipts/fixtures/exported/ane-mul-fp16-1x512
```

All three commands printed `[receipt] OK: bundle valid`. The parsed contracts were:

```text
ane-add-fp16-1x512: payload_size=16384 td_size=628 td_count=1 task_size=504 kernel_size=1024 sources=1 destinations=1; input/output channel_bytes=32768
ane-add-fp16-1x896: payload_size=16384 td_size=628 td_count=1 task_size=504 kernel_size=1792 sources=1 destinations=1; input/output channel_bytes=65536
ane-mul-fp16-1x512: payload_size=16384 td_size=628 td_count=1 task_size=504 kernel_size=1024 sources=1 destinations=1; input/output channel_bytes=32768
```

### Exporter syntax and CLI

```sh
python3 -m py_compile overlay/tools/ane-export/ane_export.py
python3 overlay/tools/ane-export/ane_export.py --help
```

Both commands succeeded. Help includes:

```text
--target TARGET       ANECompiler hardware target passed to ane-compile-hwx
```

## Remaining gates

ANE runtime and performance remain unsupported. This preflight does not provide:

- an mlx-omarchy worker that owns the ANE device and live buffers;
- MLX graph partitioning or general graph lowering to exporter descriptors;
- on-device parity for a validated bundle on M1 hardware;
- failure recovery and bounded worker lifecycle evidence;
- same-chip numerical and performance qualification that includes copy and IPC cost.

A fixture validation is not an M1 hardware qualification or an ANE performance result.
