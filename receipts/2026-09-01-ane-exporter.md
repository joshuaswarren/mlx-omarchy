# U6 exporter: versioned ANE artifact path (2026-09-01)

## Verdict: ACCEPTED (with a recorded op boundary)

`tools/ane-export/ane_export.py` turns a JSON descriptor into a compiled,
manifest-packaged ANE bundle. Three bundles were exported on 16m1mbp and all
validate on Linux via `mlx-omarchy-info --check-bundle` (exit 0 each):

- `ane-add-fp16-1x512` - byte-identical reproduction of the
  `mil-oneop-bundle` fixture (same graph hash, same payload digests from a
  fresh compile).
- `ane-add-fp16-1x896` - new shape, proves parameterization.
- `ane-mul-fp16-1x512` - a second op, proving the op axis, not just the shape
  axis.

`matmul` did not compile: ANECompiler 9.509.0 rejected the hand-authored MIL
text in two forms (attempts 2 and 3 of the 3-attempt new-op budget). The
boundary is recorded below with the exact rejected program.

## Host and toolchain identity

| Item | Value |
|---|---|
| Compile host | 16m1mbp, MacBook Pro, arm64 |
| macOS | 26.6.2 (25G83) - `sw_vers` re-run this session |
| ANECompiler.framework | 9.509.0 (probed via `plutil ... CFBundleShortVersionString`) |
| coremlcompiler | 3520.5.1 (`xcrun coremlcompiler version`) |
| Compile interpreter | `~/src/mil-oneop-proof-20260831/.venv7/bin/python` (py 3.12) |
| Validation host | omp-studio-local, x86_64, kernel 6.17.2-1-pve (joshuawarren@local) |
| Checker | `./.work/build/tools/mlx-omarchy-info/mlx-omarchy-info` rebuilt this session (`scripts/prepare-mlx.sh` + `cmake --build .work/build --target mlx-omarchy-info -j4`) |

Scratch dirs: new work in `~/src/ane-export-20260901/` (exporter + run log);
compile/convert tools used in place from the proven
`~/src/mil-oneop-proof-20260831/` (`ane-compile-hwx` binary 52336 B,
`hwxv2-to-anec-patched.py`). `~/src/ane-linux-experiments` untouched.

## Tool

`overlay/tools/ane-export/ane_export.py` (stdlib-only): descriptor JSON ->
MIL `program(1.3)` text (coremlc 3520.5.1 dialect, buildInfo replicated from
the retained capture) -> `weights.bin` (0x40 header + 0x40 `0xDEADBEEF` blob
record, payload at 0x80) -> `./ane-compile-hwx` (ANECCompile, h13) ->
`hwxv2-to-anec-patched.py` (parses `td-count` and `workspace` from its output
into the manifest) -> `bundle/` with manifest.json + payloads only. Manifest
fields match `overlay/mlx/backend/omarchy/ane/manifest.cpp` exactly:
`graph_hash` = sha256 of `model.mil`, compiler identity probed live
(`sw_vers`, framework plist, `xcrun coremlcompiler version`), firmware range
26.0..26.6 by flag, provenance repo/commit by flag (mlx-omarchy
`c17bb1b82490b661ef6a3acd1fa008265383802b`), `release_asset.model_sha256` =
anec digest.

Descriptor form: `{"op": "add"|"mul"|"matmul", "input_shape": [...],
"weight_shape": [... for matmul], "const_value": 0.25}`.

## Op scoping (new-op budget: 3 compile attempts)

| # | Op | Form | Result |
|---|---|---|---|
| 1 | `mul` [1,512] | `mul(x = t1, y = t0)` | compiled, converted, validated |
| 2 | `matmul` [1,16]x[16,32] | `matmul(x = t1, y = t0)` | `ANECCompile=1 callback_status=1`, no model.hwx |
| 3 | `matmul` [1,16]x[16,32] | explicit `transpose_x = false, transpose_y = false` | `ANECCompile=1 callback_status=1`, no model.hwx |

Attempt 3 program (verbatim, `capture/model.mil`, weights sha
`9436a956...c6143` = the standard 512-element fp16 blob):

```
program(1.3)
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3520.4.1"}, {"coremlc-version", "3520.5.1"}})]
{
    func main<ios18>(tensor<fp16, [1, 16]> t1) {
        tensor<fp16, [16, 32]> t0 = const()[name = string("t0"), val = tensor<fp16, [16, 32]>(BLOBFILE(path = string("@model_path/weights.bin"), offset = uint64(64)))];
        tensor<fp16, [1, 32]> t2 = matmul(x = t1, y = t0, transpose_x = false, transpose_y = false)[name = string("t2")];
    } -> (t2);
}
```

The failure is the same deterministic `callback_status=1` shape as the
inline-const rejection in the 2026-08-31 proof, with no diagnostic text.
Boundary: this compiler build does not accept hand-authored 2-D `matmul` MIL
text in either spelling. Elementwise `add`/`mul` remain the proven region
family; matmul regions need either the coremlc-produced dialect form
(capture a real matmul with coremlc and diff) or a different lowering route.
Budget spent; no further matmul attempts were made.

## Export and validation record

Compile+convert (every line from `export.log` on 16m1mbp):

```
add [1,512]:  ANECCompile=0 callback_status=0 ; wrote=...ane content=0x4000
              task-stream=0x1f8 td-count=1 workspace=0x0 EXIT=0
add [1,896]:  ANECCompile=0 callback_status=0 ; wrote=...ane content=0x4000
              task-stream=0x1f8 td-count=1 workspace=0x0 EXIT=0
mul [1,512]:  ANECCompile=0 callback_status=0 ; wrote=...ane content=0x4000
              task-stream=0x1f8 td-count=1 workspace=0x0 EXIT=0
matmul:       ANECCompile=1 callback_status=1 EXIT=1 (both attempts)
```

`workspace=0x0` for every compiled graph, so the manifests record the
one-tile 0x4000 submit-time scratch, matching the fixture's derivation.

Checker output on omp-studio-local (all three exit 0, `bundle valid`):

```
ane-add-fp16-1x512  graph_hash 0417bba9b253e8ada7ab60357aaa7cd05d0000dde58d2c578e6174c310ad3151
  anec 479a30e8197bce9130f5d78939ab2862b83d015ca27e8623b8855747981f527e (20480 B)
  weights 9436a9563e7e7c17767424814e9d418774cad1a507715a75891a15c24c6c6143 (1152 B)
ane-add-fp16-1x896  graph_hash 9048326d6335b6bec29296d0ce521fa528c09c4558d93e15bf1830d7ec777da1
  anec 79d307207318285d9e2aa8a52b9ac06a3460c09c6628c26dfa14a9acd01a8271 (20480 B)
  weights 41aa2b92c429c8790c55c37015e02d719c28eb7aa4ca26064fec95fae40dc0bb (1920 B)
ane-mul-fp16-1x512  graph_hash fcd8aed34f664f6972866e46c49486a8ba70e92eccc61d9078f32f49261ce65d
  anec 01c61e430633d4d94b3ca9b8dc426cc03f650df49a3deca6ca6845de9712d8ec (20480 B)
  weights 9436a9563e7e7c17767424814e9d418774cad1a507715a75891a15c24c6c6143 (1152 B)
```

The add [1,512] digests equal the `mil-oneop-bundle` fixture's exactly: the
exporter reproduces the proven artifact byte for byte from a fresh compile.
Full `[receipt]` lines for each bundle are in the session transcript; the
checker ran `load_bundle` only and opened no device.

## Deviations and notes

1. **Interpreter pitfall:** bare `python3` on 16m1mbp resolves to Xcode's
   Python 3.9, which cannot run the converter (`bytes | mmap.mmap` union
   syntax). The exporter drives the converter with its own interpreter, so
   the driver invokes everything under the proof's 3.12 venv python.
2. **matmul rejected** as recorded above; the exporter keeps the
   `transpose_x/transpose_y` spelling so a future dialect fix is a one-line
   change.
3. No repo commits; all artifacts under `receipts/` and `overlay/`.

## Files added (nothing committed)

```
overlay/tools/ane-export/ane_export.py     exporter (macOS tool)
overlay/tools/ane-export/README.md         workflow + community-submission note
receipts/fixtures/exported/ane-add-fp16-1x512/{manifest.json,model.anec,weights.bin}
receipts/fixtures/exported/ane-add-fp16-1x896/{manifest.json,model.anec,weights.bin}
receipts/fixtures/exported/ane-mul-fp16-1x512/{manifest.json,model.anec,weights.bin}
receipts/fixtures/exported/desc-*.json     the four descriptors used
receipts/fixtures/exported/run-export.sh   mac driver (venv python, proven tools dir)
receipts/2026-09-01-ane-exporter.md        this receipt
docs/ane-bundles.md                        status + check-a-bundle + submissions updated
```

Scratch left on 16m1mbp for reproduction: `~/src/ane-export-20260901/`
(exporter copy, descriptors, per-descriptor out-dirs with capture/, hwx/, bundle/,
export.log).
