# ane-export

 This directory contains two host-side paths that produce schema-4 ANE bundles:

- `ane_export.py` retains the macOS reference workflow for small elementwise
  captures. It uses Xcode, ANECompiler, and the private `ANECCompile` entry
  point.
- `h13_package_to_bundle.py` adapts an explicit Linux `mil-hwxc` H13 ANEC
  package without private Apple frameworks.

Both outputs must pass `mlx-omarchy-info --check-bundle` before any device work.
A successful check proves bundle structure and declared H13/driver-ABI
compatibility. It does not prove physical-device or numerical qualification.

## macOS reference workflow

Build the companion compiler tool:

```
xcrun clang++ -std=c++17 -fblocks -framework Foundation \
  -F/System/Library/PrivateFrameworks -framework ANECompiler \
  ane-compile-hwx.mm -o ane-compile-hwx
```

Describe and export one region:

```
printf '%s\n' '{"op":"add","input_shape":[1,512],"const_value":0.25}' > desc.json
python3 ane_export.py desc.json --out-dir out-add-1x512 \
  --tools-dir . --target h13 --source-commit <40-hex commit>
```

The exporter emits `bundle/manifest.json`, `bundle/model.anec`, and
`bundle/weights.bin`. It also retains the MIL capture and HWX conversion input
 outside the bundle. Schema 4 records one program, explicit ordered identity
 return views, exact channel allocations, and driver ABI major 1.

Both producers set `release_asset.model_sha256` to the same compiled-payload
collection identity. They sort payload records by `path`, retain exactly
`role`, `path`, `byte_size`, and `sha256`, serialize with
`json.dumps(records, sort_keys=True, separators=(",", ":"),
ensure_ascii=True)`, and hash the UTF-8 bytes. ANEC and weights records are
included.

Supported descriptors use `add`, `mul`, or `matmul`, positive input shapes, and
fp16. `matmul` also needs `weight_shape`. The retained compiler rejected the
hand-authored matmul forms in the 2026-09-01 receipt. Const tensors use
`weights.bin` through `BLOBFILE`; inline fp16 constants are not accepted by
that reference compiler.

## Explicit Linux package workflow

 The adapter accepts only `mil-hwxc.h13-anec-package.v2`, target `H13`, and
artifact format `anec`. The compiler invocation must use `--format anec`.
Provide the source graph, a repository containing the recorded generation
commit, and the generation receipt. The receipt binds the compiler manifest,
compiler source commit, generation binary digest, graph digest, and every
emitted payload digest. `--compiler-source` only proves that the recorded
commit exists in the supplied repository. It does not verify the historical
binary or require the checkout HEAD to match.

```
python3 h13_package_to_bundle.py package \
  --out-dir bundle \
  --graph-source model.mil \
  --compiler-source /path/to/mil-hwx-compiler \
  --compiler-receipt package/source.json \
  --name graph-region \
  --source-repo owner/repository \
  --source-commit <40-hex commit> \
  --model graph-region
```

 The adapter preserves explicit physical outputs and ordered logical return
 views, including duplicates, reshapes, and overlapping slices. It retains
 program order, bindings, NCHW geometry, and allocations. The manifest digest
 must match the receipt before wiring is read. HWX packages, non-H13 targets,
 old schemas, unknown fields, invalid receipts, and embedded `constantInputs`
 are rejected. The old v1 archive in `ane-compiler.lock` is incompatible with
 this adapter; a new release pin awaits full compiler qualification.

## Validate

```
./.work/build/tools/mlx-omarchy-info/mlx-omarchy-info \
  --check-bundle bundle
```

The check opens no Vulkan or ANE device. A missing or invalid bundle leaves the
region on Vulkan. The runtime-generated cache key and qualification boundaries
are documented in `docs/ane-bundles.md`.
