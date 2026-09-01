# ANE bundles

An ANE bundle is a directory that carries one compiled region for Linux
execution: a `manifest.json` plus the payload files it lists. The Linux
runtime validates every manifest field before it maps or submits a
descriptor. A bundle that fails any check never reaches the device. This is
the U6 validation layer in
`docs/plans/2026-08-29-mlx-omarchy-ane-compatibility-plan.md`.

Status: the Linux validation layer, its tests, and the macOS exporter
(`tools/ane-export/`) are in place. Lowering general MLX graphs to the
exporter's descriptors and M1 execution remain open work; see
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

## Failure contract

| Condition | Outcome |
| --- | --- |
| Bundle directory missing | `AneBundleNotFound`; the affected region stays on Vulkan |
| `manifest.json` unreadable or invalid JSON | named manifest error |
| Any field missing, wrong type, or out of contract | named manifest error |
| Extra file, missing payload, size mismatch, hash mismatch | named bundle error |

## Check a bundle

`mlx-omarchy-info --check-bundle <dir>` validates one bundle directory and
prints its parsed contract as `[receipt]` lines: graph name and hash, task
descriptor count, every tensor, payload names and digests, and the compiler
and firmware identity. The check runs `load_bundle` only. It opens no Vulkan
device, so it works on any Linux host.

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

`omarchy_ane_bundle_tests` builds fixture bundles at runtime and covers:
exact contract exposure, every single-field mutation with its named error,
mutation-before-payload ordering, graph identity repeatability, the
not-found contract, and unknown-payload rejection. Run:

```
./tools/ci/run-ane-bundle-tests.sh
```
