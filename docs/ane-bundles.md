# ANE bundles

## External preservation mirror

The maderix HWX compiler source and the "Inside the M4 Apple Neural
Engine" article series are mirrored at
<https://github.com/joshuaswarren/ane-research-mirror>. We cite
articles and sections in this doc and never copy prose or code into
this repo. See `docs/ane-hwx-format-notes.md` for the field-by-field
reconciliation; this doc only carries the bundle-schema-relevant
findings.


An ANE bundle is a directory that carries one compiled region for Linux
execution: a `manifest.json` plus the payload files it lists. The Linux
runtime validates every manifest field before it maps or submits a
descriptor. A bundle that fails any check never reaches the device. This is
the U6 validation layer in
`docs/plans/2026-08-29-mlx-omarchy-ane-compatibility-plan.md`.

Status: the Linux validation layer, its tests, and the macOS exporter
(`tools/ane-export/`) are in place. The loader now checks the libane `.anec`
header against the manifest before a worker can map buffers. Lowering general
MLX graphs to the exporter's descriptors and M1 execution remain open work; see
`docs/compatibility.md`.
See `docs/ane-hwx-format-notes.md` for the HWX container and task-descriptor
format reference (external guide evaluation).

## Layout

```
my-bundle/
  manifest.json      required, manifest_version 1
  model.anec         required, exactly one "anec" payload
  weights.bin        optional "weights" payload
```

The directory holds regular files only. Every file must be `manifest.json` or
a listed payload. An unlisted file is an error: a bundle cannot carry payloads
the manifest does not describe.

## Manifest schema (manifest_version 1)

All checks fail closed. Each error names the field and the reason.

| Field | Type | Checks | Receipt ground |
| --- | --- | --- | --- |
| `manifest_version` | int | must be `1` | Versioned format per R9/R10. |
| `name` | string | non-empty | Region name used in traces and release keys. |
| `graph_hash` | string | 64 lowercase hex | Identity of the lowered graph; the export receipt hashes artifacts with `shasum -a 256` (`qwen-hwx-export-convert.log`). |
| `task_descriptors` | int | positive | Task counts are receipted: 1403 and 1175 for the 13- and 11-layer Qwen graphs, 3072 for the production graph (`qwen-linux-terminal-task.log`). |
| `inputs` | tensor list | see below | Descriptor buffer geometry: `source-nchw=(1, 2048, 1, 1, 64, 64)` (`qwen-linux-task-layout.log`). |
| `outputs` | tensor list | see below | `output=0x90c000` with matching NCHW (`qwen-linux-terminal-task.log`). |
| `state` | tensor list | see below | Device-resident K/V state with stable indices across decode steps: 2 kv heads, 128 context, 256 dim (`qwen-linux-kv-state-validation.json`). |
| `workspace` | tensor list | exactly one entry | One scratch buffer bound at submit time before enqueue: `workspace=0x66000` (`qwen-linux-terminal-task.log`). |
| `payloads` | payload list | exactly one `anec`, at most one `weights`; unique safe relative paths | MIL `program(1.3)` plus `weights.bin` per KTD6; sha256 per payload per `qwen-linux-kv-state-validation.json` (`anec_sha256`, `hwx_sha256`). |
| `compiler.macos_build` | string | non-empty | `build 20A2411` (`qwen-bigsur-compiler-attempt.log`). |
| `compiler.anecompiler` | string | non-empty | `Monterey ANECompiler 5.5.0` (`qwen-linux-kv-state-validation.json`); `zin_ane_compiler v4.2.1` on Big Sur. |
| `firmware.min`, `firmware.max` | dotted integer versions | `min <= max` | The macOS 26 compiler emits TD encodings "this firmware/KMD cannot execute" (`ane-static-graph-loop.log`), so a bundle must record the firmware range it was proven against (R19). |
| `provenance.source_repo` | string | non-empty | Experiment receipts name their source repo. |
| `provenance.source_commit` | string | 40 lowercase hex | `7f80713` push-before-hardware discipline (`qwen-linux-task-layout.log`). |
| `provenance.exported_at` | string | `YYYY-MM-DD` | Receipt dating convention. |
| `release_asset.model`, `release_asset.model_sha256` | strings | model sha256 is 64 lowercase hex | Release assets are keyed by exact model, shapes, compiler, firmware, and graph hash (R19, KTD10); reference model sha256 in `docs/compatibility.md`. |

### Tensor entries

Each entry in `inputs`, `outputs`, `state`, and `workspace` has:

| Field | Checks |
| --- | --- |
| `name` | non-empty, unique inside its list |
| `index` | non-negative integer, unique inside its list; this is the descriptor buffer index |
| `dtype` | one of `float16`, `float32`, `bfloat16`, `int32`, `uint8` |
| `shape` | non-empty array of positive integers |
| `byte_size` | positive; must equal `dtype size x product(shape)` exactly |
| `stride` | positive; at least `byte_size` |

The kernel window needs no separate manifest field: it lives inside the
`anec` payload at converter-defined offsets (`kernel@content+0xea300`,
`qwen-linux-task-layout.log`), and `task_descriptors` cross-checks the
converted stream.

### Tile alignment

`stride` must be a multiple of `0x4000` for `inputs`, `outputs`, and `state`.
Descriptor DMAs address these tensors through tile-aligned rows: the converter
fixes `TILE_SIZE = 0x4000` (`ane-linux-experiments/tools/hwxv2-to-anec.py`),
and task KDMA offsets step `0x28000..0x3c000` in `0x4000` increments
(`ane-static-graph-loop.log`).

`workspace` is exempt. Its size is the compiler's choice of scratch bytes, and
the receipts record sizes that are not `0x4000` multiples, such as `0x66000`
(`qwen-linux-terminal-task.log`).

The manifest `stride` is the logical DMA stride recorded for compatibility.
The full libane staging allocation is derived from the `.anec` header as
`tiles[channel] * 0x4000`; `mlx-omarchy-info --check-bundle` prints those
channel byte counts beside the manifest fields. The current libane tile and
untile ABI copies 16-bit elements, so this ANEC channel preflight accepts only
`float16` and `bfloat16` input, output, and state tensors.

## Validation order

`load_bundle(dir)` performs these steps in order. Steps 1 and 2 touch no
payload; a changed graph, shape, compiler, or firmware field fails before any
payload mapping, and nothing here accesses a device.

1. **Directory exists.** A missing directory returns the distinct
   `AneBundleNotFound` outcome. Callers treat it as "region stays on Vulkan";
   it is a normal runtime condition, not an error.
2. **Manifest parse and field validation.** JSON parse, unknown-field
   rejection, then every field check in the tables above.
3. **Directory scan.** Regular files only; every file must be `manifest.json`
   or a listed payload.
4. **Payload presence.** Each listed payload must exist.
5. **Payload byte size.** The file size must match the manifest.
6. **Payload sha256.** The digest must match the manifest.
7. **ANEC header and channel ABI.** After digest verification, the loader
   parses the libane header at file offset 0 and confirms the executable
   payload starts at `0x1000`. It checks the task count and source and
   destination counts against the manifest, proves the payload and task stream
   fit command channel 0, requires reserved kernel channel 1 to be unbound,
   checks the task-descriptor word units and aligned kernel envelope, and
   checks each manifest tensor against its 16-bit tile/NCHW allocation.

## Failure contract

| Condition | Outcome |
| --- | --- |
| Bundle directory missing | `AneBundleNotFound`; the affected region stays on Vulkan |
| `manifest.json` unreadable or invalid JSON | named manifest error |
| Any field missing, wrong type, or out of contract | named manifest error |
| Extra file, missing payload, size mismatch, hash mismatch | named bundle error |
| ANEC task or channel count, payload or task envelope, reserved channel, task-descriptor units, kernel envelope, dtype, or NCHW mismatch | named bundle error before worker/device access |

## Check a bundle

`mlx-omarchy-info --check-bundle <dir>` validates one bundle directory and
prints its parsed contract as `[receipt]` lines: graph name and hash, task
descriptor count, every tensor, libane ANEC header fields, each input/output/
state channel binding with channel bytes and NCHW, payload names and digests,
and the compiler and firmware identity. The check runs `load_bundle` only. It
opens no Vulkan device, so it works on any Linux host.

Exit codes:

| Code | Meaning |
| --- | --- |
| 0 | Bundle valid; contract printed. |
| 1 | Invalid; the loader's named error is printed. |
| 2 | Directory not found; the region stays on Vulkan. |

```
./.work/build/tools/mlx-omarchy-info/mlx-omarchy-info \
  --check-bundle receipts/fixtures/mil-oneop-bundle
```

The reference example is `receipts/fixtures/mil-oneop-bundle/`, the real
compiled one-op MIL add artifact from
`receipts/2026-08-31-mil-oneop-proof.md` ([1, 512] fp16 add, input `t1`,
output `t2`, const through `weights.bin`). Field derivations:

- `graph_hash` is the sha256 of the MIL program `model.mil`; the payload
  digests hash the shipped `model-512.anec` and `weights.bin`.
- `task_descriptors` is 1, the converter's `td-count=1` for this graph.
- The compiled task stream needs no scratch (`workspace=0x0` in the
  conversion receipt), so the manifest records the one-tile submit-time
  scratch: 0x4000 bytes.
- The compiler identity and firmware range record the compile host:
  macOS 26.6.2 (25G83), ANECompiler 9.509.0, coremlcompiler 3520.5.1.
  On-device execution is not yet proven; see the proof receipt.

New bundles are produced with the macOS exporter, `tools/ane-export/`
(runs on macOS with Xcode + ANECompiler; Linux validates only). It turns a
JSON descriptor into the MIL program, `weights.bin`, the compiled `hwx`,
the converted `.anec`, and a manifest in this schema.
`receipts/fixtures/exported/` carries its output for add [1, 512],
add [1, 896], and mul [1, 512]; see `receipts/2026-09-01-ane-exporter.md`.

## Community submissions

The exporter is that user-runnable tool (KTD6): `tools/ane-export/`. A
community bundle submission must reproduce through it and pass exactly these
checks on Linux, with no private Apple frameworks, compiler binaries,
firmware, or model weights in the bundle (R18). Release assets carry bundles
keyed by exact model, shapes, compiler, firmware, and graph hash (KTD10); a
missing asset leaves the region on Vulkan.

## Tests

`omarchy_ane_bundle_tests` builds fixture bundles at runtime and covers the
manifest contract, libane ANEC header and channel exposure, state bound as both
source and destination, validation ordering, graph identity repeatability, the
not-found contract, unknown payloads, and malformed ANEC envelopes, channels,
task fields, dtypes, and NCHW geometry. Run:

```
./tools/ci/run-ane-bundle-tests.sh
```

## External reference: maderix (Inside the M4 ANE)

This section reconciles the maderix bundle/runtime findings against
`docs/ane-bundles.md` and the macOS exporter. Each claim is tagged
**[D]** (demonstrated by the author with hardware evidence) or
**[H]** (hypothesised). The author's hardware is M4 (H16G); see the
**M4 vs M1 transfer** subsection below.

### Container and section layout

- **[D]** The HWX is a custom Mach-O with magic `0xBEEFFACE`,
  `cputype = 0x80`, `cpusubtype = 0x07` for H16G. The `cpusubtype`
  byte names the ANE generation (Part 4 "What the ANE Is"). The
  M1 subtype is a different value, but the dispatch table shape
  (H11..H18) is what the freedomtan guide already documents.
- **[D]** Sections: `__TEXT.__text` (TD stream), `__KERN_0` (weights /
  activation LUT), `__FVMLIB` (in/out descriptors). The kernel /
  LUT and the IO descriptors ride the Mach-O, **not** the TD stream.
  Our converter's `__TEXT,__const` placement on M1 is the same fact
  under a different compiler version; the contract that "kernel
  payload is a Mach-O section" transfers cleanly.
- **[D]** The TD stream is a register-write record, not a CPU-style
  instruction stream. It configures DMA engines and the compute
  array for one graph execution (Part 4 "How Programs Reach the
  Hardware").

### Physical tensor layout (Part 4b "Physical tensor layout")

- **[D]** Logical rows shorter than 64 bytes are padded to a 64-byte
  physical row in the tensor record. Higher-level sizes (plane,
  batch, allocation) derive from that physical stride. The HWX
  writer records both the logical and physical views; the runtime
  fills and reads IOSurfaces through the physical strides. This is
  why our `0x4000` (`= 16384`) tile stride exists: a row pad of
  64 bytes across 256 elements at fp16 is exactly 512 bytes, and
  `0x4000` is the natural SRAM row for the 64-bank scratchpad.
- **Reconciled with our work:** the converter's `TILE_SIZE = 0x4000`
  is one concrete instance of this rule. The author's 64-byte
  row-pad is what travels; the specific stride depends on tensor
  geometry. Until the exporter names both logical and physical
  strides, mismatches will be silent.

### Compiler / firmware identity fields

- **[D]** The author's compile host was macOS 26.3 (build 25D125) with
  `ANECompiler 9.202.0` for H16G; the framework
  `DTCompiler = com.apple.compilers.llvm.clang.1_0` is just the
  Clang build toolchain marker (Part 4b "The compiler process and
  its front ends"). The framework statically links LLVM/MLIR; the
  `.mlir` input goes through `MLIRContext`/`parseSourceFile`/
  `PassManager::run`, but those breakpoints did **not** fire for
  any MIL compile the author traced.
- **[D]** `libORTools.dylib` (Google OR-Tools CP allocator) loads
  only for two-branch-convolution graphs; plain chains stay on the
  internal solver. This is a macOS framework detail and does not
  affect bundle payloads.
- **Reconciled with our work:** the manifest already records
  `compiler.macos_build` and `compiler.anecompiler`. The maderix
  evidence is consistent. No new field is needed for what the
  author proved; the missing field is **`compiler.dt_compiler`**
  for the build-tool marker, which is private-Apple and of no use
  to a Linux validator.

### Compiler options the exporter can capture (Part 4b "Compiler options and retained debug output")

- **[D]** `ANECCreateCompilerOptionsCFString` is the supported
  hook for feeding options into `ANECCompile`. The CFDictionary
  → flag mapping the author tabulated:
  - `TargetArchitecture = H16G` → `-t H16G`
  - `CompileANEProgramForDebugging` → `--debug` (larger debug HWX
    plus retained debug logging)
  - `DebugMask = 0xffffffff` → `--debug_mask=0xffffffff` (reaches
    gated checkpoints; the graph writer is stubbed in release)
  - `DumpStatusDictionaryToFile` → `--fdump-status-dictionary-to-file`
    (status plist with live IO + max DRAM)
  - `DumpParallelScore` → `--dump-parallel-score=true` (init/refine
    partition JSON)
  - `DisableOptimizations` → `--O0`
  - `OptLvlOne` → `--O1`
- **[D]** With `--debug`, the compiler retains `probe.status.plist`
  (live IO and max DRAM), `init.json` and `refine.json` (partition
  scores), and a larger debug HWX. Named graph checkpoints
  (`before_fusion`, `after_fusion`, `after_engine_lowering`,
  `after_fusion2`, `after_mir_opt`, `after_reg_spill`) are reached
  in release but the two writers are **stubs** (one writer is a
  single `ret` instruction). The author worked around this with
  live LLDB object tracing.
- **Reconciled with our work:** the macOS exporter can capture the
  retained status plist and partition JSON as side-files at
  export time, indexed by `provenance.export_run_id`. They are not
  bundle payloads and do not need to ship with the bundle, but
  they belong in the export receipt. See follow-up F1 below.

### Debug plist contents (Part 4b "Compiler options and retained debug output")

- **[D]** The status plist records: the compiled input, the maximum
  DRAM usage (1,073,624 bytes for the author's 64-channel
  convolution probe), the procedure name, and every live tensor's
  dimensions, type, interleave, and strides. This is the closest
  thing to a structured "compiled tensor layout" we have, and it
  comes from the private framework itself rather than from our
  reverse-engineered offsets.
- **Reconciled with our work:** the manifest's `inputs`, `outputs`,
  `state`, and `workspace` arrays are filled from the MIL graph at
  export time. The status plist provides the **compiler's own view**
  of the same tensors after fusion, lowering, and tiling. A future
  exporter could dump both and cross-check that they agree on
  dimensions and strides.

### Fusion speedup (Part 4 "Performance" → "DMA")

- **[D]** A fused matmul + bias + relu runs 5.7× faster than the same
  three operations submitted separately, because the compiler emits
  `DMA_INTER` (intermediate stays in SRAM) instead of
  `DMA_STORE` followed by `DMA_LOAD`.
- **Reconciled with our work:** the macOS exporter already produces
  fused MIL where possible; the converter does not need to change.
  But the Linux-side validation path could check that the HWX
  contains `DMA_INTER` records when the MIL graph claims fusion —
  a missing `DMA_INTER` where one is expected would catch a
  compiler regression. This is follow-up F2 below.

### Where the maderix findings would change the manifest schema

- **No required changes.** Every contract field in
  `docs/ane-bundles.md` (manifest_version 1) is still the right
  contract for M1-bound bundles. Nothing the author demonstrated on
  M4 contradicts what we validate.
- **Possible additions (none required):** `td_stream_size` and the
  set of compute-mode words in the HWX are useful provenance but
  not validation criteria. They are recording only; they would not
  be hashed into `graph_hash`. Defer until a community submission
  makes the field worth standardising.
- **W8A8.** The author proved three HWX mode words (`0x93418005`,
  `0xb1418005`, `0x91418005`) and one inter-block DMA word
  (`0x80049240`) are required together. We do not encode W8A8; if
  the exporter ever sees an int8 tensor, it should fail closed
  with a clear error. The maderix evidence says we cannot get a
  valid W8A8 block by setting only one of those words.

### Suggested exporter follow-ups

1. **F1.** Capture the `--debug` plist and partition JSON in the
   export receipt (out-of-band from the bundle). Add an
   `export_run_id` field on the receipt that links to them. The
   plist carries dimensions, strides, and interleave for every
   live tensor; cross-checking it against the manifest's tensor
   list catches silent compiler-side layout drift.
2. **F2.** After the converter turns HWX into ANEC, scan the TD
   stream for `DMA_INTER` records when the MIL graph claims fusion
   and warn (not error) if the expected count is missing. A
   compiler regression that loses fusion would surface here.
3. **F3.** When the exporter sees an int8 tensor, fail closed with
   a clear error pointing at this doc and the W8A8 mode words.
   Do not silently emit a bundle that the runtime cannot execute.

### M4 (H16G) versus M1 (H11/H12/H13) transfer

The maderix bundle-runtime findings are M4-specific. Below is what
transfers directly into our bundle contract and what does not.

| Claim | Transfers? | Why |
| --- | --- | --- |
| `cpusubtype = 0x07 = H16G` | **Pattern transfers.** | The freedomtan guide already tabulates `cpusubtype` → generation; M1 is a different value but the indexing is the same. |
| `__TEXT.__text` for the TD stream | **Confirmed by us on M1.** | Already in our converter. |
| `__KERN_0` / `__FVMLIB` section names | **Likely version-only difference.** | Section naming shifted between compiler versions; the contract — kernel payload rides a Mach-O section, not the TD stream — is what travels. |
| 64-byte physical row pad for narrow tensors | **Pattern transfers.** | Apple compilers pad narrow rows to 64 bytes across generations; the exact stride depends on geometry. |
| `ANECCreateCompilerOptionsCFString` option plumbing | **Confirmed.** | The CFDictionary → flag mapping is the supported hook in `ANECompiler.framework`; this is how the macOS exporter should set `--t` and `--debug`. |
| Status plist contents | **Schema transfers; values are compiler- and graph-specific.** | We can capture the same fields in our export receipt; the values will differ for our M1-bound graphs. |
| `--O0` / `--O1` byte-identical output | **No transfer.** | The author's observation is that lowering and fusion happen at every level for the conv+ReLU probe. Our M1 compiler may differ; do not assume. |
| `libORTools.dylib` for two-branch graphs | **No transfer.** | macOS framework linkage detail; irrelevant to our Linux-side bundle contract. |
| `0x93418005 / 0xb1418005 / 0x91418005` W8A8 mode words | **M4-specific.** | Tied to the M4 compute pipeline. We do not encode W8A8 anyway. |
| `0xBEEFFACE` magic | **Confirmed by us on M1.** | Same magic. |
| 33-pt piecewise-linear LUT for activations | **Pattern transfers; values do not.** | The "non-direct activations ride a LUT in the kernel section" structure is what travels; the LUT contents are HAT-driven per generation. |

**Bottom line on transfer.** The bundle contract is generation-agnostic
enough that nothing the author proved on M4 invalidates the manifest
schema in `docs/ane-bundles.md`. The cross-generation findings are
confirmation, not contradiction. Where the M4 work suggests adding
plumbing (debug plist capture, fusion check, W8A8 fail-closed), it is
exporter-side and Linux-side tooling that benefits, not the bundle
payload itself.

## Attribution

Format facts above paraphrase the cited freedomtan guide sections
(BSD 3-Clause; quote limits respected, no code copied) and the cited
maderix articles and source files (MIT; quoted under fair use with
attribution to article and section). The original articles and
source are preserved at the mirror repo above.

