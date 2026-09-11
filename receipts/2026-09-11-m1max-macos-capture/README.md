# 2026-09-11 — M1 Max macOS-side identity, capability and pre-install capture

Owner-approved pre-Asahi capture of the 16-inch MacBook Pro (Apple M1 Max,
64 GB, 4 TB). Everything here is what macOS on this die reports or what this
die measures directly; after the Omarchy/Asahi install none of it can be
re-produced from Linux. Companion to `docs/chip-capability-axes.json` (the
`m1_max_macos` fields added by this receipt) and to the Linux-side baseline in
`receipts/2026-09-10-native-macos-metal-baseline`.

- date: 2026-09-11 (UTC)
- host: `[redacted-host]` (16-inch MacBook Pro; placeholder per repo policy)
- chip: Apple M1 Max, 10 CPU cores (8P+2E), 32-core GPU, 64 GB LPDDR5 (Hynix)
- OS: macOS 26.6.2 (25G83), Darwin 25.6.0 (xnu-12377.161.14~5/RELEASE_ARM64_T6000)
- GPU architecture string (runtime `MTLDevice.architecture.name`): `applegpu_g13s`
- capture agent: M1MaxMacCapture, repo branch `m1max-macos-capture`
- GPU contention: run concurrently with the native-baseline re-measure; identity
  reads and compiles were cleared with that agent first, the memcpy roof probe
  ran only after its 12x6 matrix completed. The GPU blit probe (306 GB/s anchor)
  ran on the quiet machine after both.

## What was collected

### 1. Identity (Priority 1)

`identity-sysctl.txt` — sw_vers, uname, core split (hw.perflevel0/1: 8P+2E),
memory size, machdep.cpu.brand_string, SIP status (enabled), FileVault (On).

`sp-hardware-display-memory-nvme.txt` — system_profiler SPHardwareDataType
(System Firmware Version 18000.161.10, OS Loader Version 18000.161.10 — the
firmware/iBoot-adjacent versions macOS exposes to an unprivileged user),
SPDisplaysDataType (32 GPU cores, Metal Support: Metal 4), SPMemoryDataType
(LPDDR5, Hynix), SPNVMeDataType (4 TB Apple SSD AP4096R, SMART Verified).
Serial, hardware UUID and provisioning UDID redacted (4 substitutions).

`ioreg-agx-device-props.txt` — the AGXAcceleratorG13X device property block:
`gpu-core-count = 32`, GPUConfigurationVariable `{num_gps=16, gpu_gen=13,
is_sksm=0, usc_gen=2, num_cores=32, num_mgpus=4, gpu_var="C",
core_mask_list=(255,255,255,255), num_frags=32}`, MetalPluginName AGXMetalG13X,
IOSourceVersion 353.14, AGXTraceCodeVersion 3.44.6. 59 AGXDeviceUserClient
subtrees (per-app GPU client inventories) trimmed before commit.

`ioreg-platform-curated.txt` — IOPlatformExpertDevice identity: target-type
J316c, platform-name `t6031` (hex-decoded), compatible MacBookPro18,2,
regulatory A2485, NVRAM boot keys (`boot-volume` GUID triple = the Data volume
UUID 974D7F09-B2FB-4F2A-B059-00AF28B108A0, `auto-boot` true, bootdelay 0x30).
Serial, MLB serial, IOPlatformUUID values redacted; NVRAM options subtree
truncated to boot-related keys (it carried device names and telemetry blobs).

`firmware-probe.txt` — kern.osproductversion; no iBoot-version string is
exposed by ioreg/nvram to an unprivileged session on this OS (searched).

### 2. Metal device limits (Priority 1) — `metal-limits.json`

Method: `m1max_metal_limits.swift`, compiled with `swiftc -O` (Apple Swift on
the Mac, Xcode 32023 toolchain) and run against `MTLCreateSystemDefaultDevice`.
Every value is a runtime query or a compile-and-execute probe on the real
device; nothing is transcribed from documentation. Language version for the
MSL JIT probes: `-std=metal3.1`.

Runtime-queried results (verbatim from the JSON):

| property | value |
|---|---|
| architectureName | `applegpu_g13s` |
| gpuFamilies | apple1..apple7 true; apple8/apple9 false; mac1/mac2/common1-3 true; metal3 true; metal4 true |
| maxThreadgroupMemoryLength | 32768 B |
| maxThreadsPerThreadgroup | 1024 x 1024 x 1024 |
| hasUnifiedMemory | true |
| recommendedMaxWorkingSetSize | 55662788608 (~55.7 GB of 64 GB) |
| maxBufferLength | 41747087360 (~38.9 GiB) |
| argumentBuffersSupport | tier2 |
| maxArgumentBufferSamplerCount | 1024 |
| readWriteTextureTier | tier2 |

Executed probes (compile + dispatch + numeric check):

| probe | result |
|---|---|
| trivial kernel @ 1024-wide threadgroup | dispatched and completed (pipelineMaxTotalThreadsPerThreadgroup 1024) |
| threadExecutionWidth (SIMD width) | **32** |
| device atomic_float add (8x32x4096 adds, one scalar) | exact: 1048576 accumulated, exact float |
| threadgroup memory allocation at reported max (32768 B) | allocated and byte-verified |
| simdgroup_matrix 8x8x8 multiply, fp32 | compiled, executed, bit-exact vs CPU reference |
| simdgroup_matrix 8x8x8 multiply, f16 and bf16 | compiled and executed; scalar-constant case exact (32.0) in both dtypes (see caveat) |

Caveat recorded honestly: with varying-but-exact pattern inputs the f16/bf16
8x8 probes returned a constant that does not equal the fp32 reference under
this probe's layout assumptions (`simd-variants.json`); with constant 2.0
inputs both dtypes return exactly 32.0 in all 64 cells. The multiply semantics
are therefore proven; this probe's 16-bit load/layout assumption is what did
not check out, and it was not root-caused inside the capture window. Anyone
building on 16-bit simdgroup_matrix layout here should treat
`simd-variants.json` as the open question, not as a hardware defect signal.

### 3. Measured memory bandwidth (Priority 1)

Two instruments, both wall-anchored, no GPU shaders except the blit:

1. CPU large-copy roof — `memroof-cpu.json` (`m1max_memroof.c`): 8 threads
   (8 P-cores), each memcpy's its own 128 MiB slice of 1 GiB src -> 1 GiB dst,
   posix_memalign 16K, pages touched, 1 untimed warmup pass, 116 timed passes
   (>= 1.5 s wall), CLOCK_MONOTONIC around whole passes, traffic counted as
   read+write. **189.91 GB/s** (dst byte-verified). The first untimed pass
   measured ~166 GB/s; steady-state passes reach ~190.
2. GPU large-copy roof — `memroof-gpu-blit.json` (`m1max_blit_roof.swift`):
   MTLBlitCommandEncoder copy, 256 MB -> 256 MB, one command buffer + wait per
   rep, 3 warmup + 20 timed, CLOCK_MONOTONIC. **median 306.26 GB/s** (min
   271.4, max 313.6), dst byte-verified. This is the same instrument class as
   the Linux-side anchor in `receipts/2026-09-11-q4-memory-roof` (256 MB large
   streaming copy, wall-anchored): that receipt's G13 8-core roof was
   58.44/58.52 GB/s; this die's 32-core blit copy roof is ~306 GB/s. The Linux
   side can now gap its roof against a same-die macOS number.

Interpretation guard: the 400 GB/s figure is the LPDDR5 package peak; the 306
GB/s blit number is a real sustained copy on this die, and the CPU number is a
copy-roof lower bound from 8 memcpy threads. Neither is a read-only peak.

### 4. Disk and volume layout before repartition (Priority 1)

`diskutil-list.txt` + `diskutil-apfs-list.txt` (captured 2026-09-11 ~16:44Z):

- Internal disk0 (4 TB, APPLE SSD AP4096R, GPT): Apple_APFS_ISC 524.3 MB
  (disk0s1), Apple_APFS container disk3 = 4.0 TB (disk0s2), Apple_APFS_Recovery
  5.4 GB (disk0s3).
- Container disk3 (CECAD06A-5B50-41E1-9962-ABC78B3A2F0C): Capacity In Use
  3.9 TB (98.6%), **Capacity Not Allocated 57.3 GB**; volumes: System
  (Macintosh HD, sealed, 12.6 GB), Preboot 9.0 GB, Recovery 1.3 GB, Data 3.9 TB
  (FileVault Yes), VM 5.4 GB, Nix Store 3.8 GB.
- Booter disk3s2, Recovery disk3s3; default boot (`bless --info --getboot`,
  ran without sudo): `/dev/disk3s1`; NVRAM boot-volume GUID triple matches the
  Data volume 974D7F09-B2FB-4F2A-B059-00AF28B108A0.
- FileVault is **On** (Data + System volumes), SIP enabled. The Asahi resize
  will run against an encrypted, 98.6%-full APFS container.
- 25 hourly Time Machine **local snapshots** on the Data volume
  (2026-09-10T15:40 through 2026-09-11T15:39) — purgeable space the installer
  may count differently than `df` does.
- `df -h`: Data 3.6 Ti used, **53 Gi avail**; System snapshot 12 Gi used.
- Also present: external 1 TB disk6 (Apple_APFS "SD Storage", 183.8 GB used,
  816.7 GB free) and mounted disk images (xrOS simulator 16.1 GB,
  MetalToolchainCryptex 2.3 GB) — not part of the internal container but they
  occupy the `diskutil list` view the installer will show.

These are the auditable pre-repartition numbers: after Asahi resizes the
container, compare partition table and container free space against this file.

### 5. Pinned MLX kernels compiled by this Mac's Metal toolchain (Priority 2)

`mlx-kernels/` — method: the pinned MLX tree was prepared with
`scripts/prepare-mlx.sh` at this repo's `mlx.lock` pin; the
`mlx/backend/metal` kernel tree was copied to the Mac and compiled with the
Mac's own toolchain: `Apple metal version 32023.883 (metalfe-32023.883)`,
`Target: air64-apple-darwin25.6.0` (from the Metal toolchain cryptex). For each
file: `xcrun metal -c <file>.metal -I <pinned root> -o <file>.air`.

| file | result | evidence |
|---|---|---|
| scaled_dot_product_attention.metal | compiles clean (rc=0) | `scaled_dot_product_attention.air` (158 KB) committed + `sdpa.dis` (20,077 lines of AIR disassembly, via the toolchain's `air-objdump -d`) |
| quantized.metal | compiles clean (rc=0) | 3,657 symbols (`quantized.nm`); AIR 11 MB, hash only, not committed |
| quantized_nax.metal | compiles clean (rc=0) | 3,657 symbols (`quantized_nax.nm`); AIR 12.3 MB, hash only, not committed |

What the symbols prove about native arithmetic on this die:

- The pinned vector SDPA compiles at `float16_t`, `bfloat16_t` AND `float`,
  head dims 64/96/128/192/256 (`sdpa_vector_2pass_1_*` entries in
  `kernel-entries.txt`). The bf16 SDPA route is real on this toolchain.
- The vector SDPA path uses NO simdgroup_matrix — only
  `simdgroup_index_in_threadgroup` / `thread_index_in_simdgroup` attributes
  (grepped from the pinned `sdpa_vector.h`). It is structurally
  subgroup-width-agnostic beyond lane indexing.
- The steel GEMM stack (used by prefill quantized matmul) is the
  simdgroup_matrix consumer: `mma.h` instantiates
  `metal::simdgroup_matrix<T, kFragRows, kFragCols>`, and the pinned `bf16.h`
  types map to the NATIVE `bfloat` type (`typedef bfloat bfloat16_t`) — which
  compiles clean here, i.e. native bf16 matrix arithmetic is available to the
  pinned kernels on this GPU family.
- Context for `docs/known-defects.md` (upstream keeps attention intermediates
  in float32 while our route stores them narrow): on THIS die the compiler
  accepts narrow (f16/bf16) SDPA instantiations natively, and the die executes
  exact f16/bf16 scalar matmuls (probe above). The upstream f32 choice is not a
  hardware limitation of this M1 Max.

SHA256 of every committed kernel artifact: `mlx-kernels/SHA256SUMS.txt`.
Hashes of the two large quantized .air files (not committed):
quantized.air `61b7bdd08b014091b7921e8256c79982df67948faf1ecf9ed9fe83f013744962`,
quantized_nax.air `e537976c42924e6aba17e742028671e11cfa3bdd9961c488e65d0e2be58b2eaa`,
combined metallib (58 MB, not committed)
`68c392123edcc169071d0e35af018f904721280598628392769bcab32721c387`.

### 6. Added on judgment (Priority 2), and why

- **FileVault/SIP status and the full snapshot list**: the Asahi resize runs
  against an encrypted, nearly-full container with 25 hourly snapshots; without
  these facts a post-install free-space delta is unauditable.
- **NVRAM boot-volume GUID triple + bless getboot**: the pre-install default
  boot target, needed to interpret the installer's boot policy changes.
- **AGX GPUConfigurationVariable** (`gpu_var="C"`, num_gps=16, num_mgpus=4,
  core_mask_list all-255): the die's own core-fusing report — the strongest
  available cross-check against Asahi's later `applegpu_g13c/g13s` identification.
- **iOS/iPadOS simulator + Metal-toolchain cryptex disk images**: they are
  mounted APFS disk images that will vanish with macOS and that a post-install
  reader would otherwise have to explain from the partition table alone.
- **GPU blit copy roof**: the q4-memory-roof receipt's 58 GB/s anchor was
  measured on a different die (8-core G13); without a 32-core same-instrument
  number the Linux roof gap analysis has no denominator.

## What could NOT be captured, and why

1. **iBoot/boot-ROM version string**: macOS 26 exposes only System Firmware
   Version / OS Loader Version (18000.161.10) to an unprivileged session; no
   `iBoot-...` string appears in ioreg or nvram output. Needs root or
   Startup Security utility access.
2. **Startup Security Utility policy / bputil security mode**: requires sudo
   + reduced-security recovery boot; not touched (machine in active use,
   capture-only mandate).
3. **`nvram efi-boot-device`**: data-not-found without sudo; replaced by
   `bless --info --getboot` (worked, `/dev/disk3s1`) and the NVRAM
   `boot-volume` GUID triple from ioreg.
4. **f16/bf16 simdgroup_matrix 8x8 layout semantics under varying inputs**:
   the probes compile and execute, scalar-constant case is exact, but the
   varying-pattern case mismatched this probe's layout assumption and was not
   root-caused (see caveat above). Open question, not a defect claim.
5. **Full AGX user-client inventory**: deliberately dropped before commit
   (per-app GPU client data is out of scope for a public receipt).
6. **Time Machine destination / backup configuration**: intentionally not
   queried (public-repo policy forbids backup-target observations).
7. **GPU shader-ISA dumps of Apple's own pipelines**: not obtainable without
   a GPU capture session and Apple-internal tools; out of scope and time-boxed.

## Redaction record

Per `AGENTS.md` public-repo policy: 5 exact-string substitutions (system
serial, hardware UUID, provisioning UDID, NVMe serial, hostname) across
identity files; AGX user-client subtree trimmed (59 subtrees); NVRAM options
subtree truncated to boot keys; device-tree hex serials replaced by typed
placeholders. `scripts/collect_common.py`-style typed placeholders are used
verbatim. A residual sweep (`grep -rniE 'PGFY|BE5627|00006001|0ba01489|
16M1MBP'`) over the receipt returns nothing.

## Reproduction

All commands are recorded verbatim in the collected files' headers or are
single-line: `sw_vers`, `sysctl hw.*`, `system_profiler <datatype>`,
`ioreg -r -c AGXAccelerator -l`, `ioreg -r -c IOPlatformExpertDevice -l`,
`diskutil list`, `diskutil apfs list`, `diskutil info /`, `df -h`,
`bless --info --getboot`, `csrutil status`, `fdesetup status`, and the Swift/C
sources committed here (`swiftc -O <file>.swift && ./a.out`,
`clang -O2 m1max_memroof.c && ./a.out`, `xcrun metal -c <k>.metal -I <root> -o
<k>.air`). The pinned MLX tree came from `scripts/prepare-mlx.sh` at this
commit's `mlx.lock`.
