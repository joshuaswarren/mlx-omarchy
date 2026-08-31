# U6 stop-gate: one-op MIL program through ANECCompile (2026-08-31)

## Verdict: ACCEPTED

A hand-authored, single-operation MIL `program(1.3)` (fp16 elementwise add,
runtime input + weights.bin const) was accepted by the private macOS
`ANECCompile` entry point on 16m1mbp. It produced a real HWX container with a
populated kernel section and I/O spans that match the authored tensor
geometry. A converted Linux `.anec` artifact was produced for two shapes.

One deviation was needed for the conversion step (TD flag-word drift, below).
The repo checkout was not modified.

## Host and toolchain identity

| Item | Value |
|---|---|
| Host | 16m1mbp, MacBook Pro, arm64 |
| macOS | 26.6.2 (25G83) |
| Xcode | 26.6 (17F113) |
| clang | Apple clang 21.0.0 (clang-2100.1.1.101) |
| ANECompiler.framework | 9.509.0 (DTPlatformVersion 26.6.1) |
| coremlcompiler | 3520.5.1 |
| coremltools (probe only) | 9.0 in scratch venv (py 3.12, homebrew) |

`sw_vers` output:

```
ProductName:	macOS
ProductVersion:	26.6.2
BuildVersion:	25G83
```

## Method

1. Read `tools/ane-compile-hwx.mm` and `tools/hwxv2-to-anec.py` (local Linux
   checkout, read-only). The tool takes a capture dir with `model.mil`
   (MIL text) plus `weights.bin`, calls `ANECCompile`, and polls for
   `model.hwx`. No mlmodel/mlpackage stage exists. Prior working invocation:
   `receipts/qwen-hwx-export-convert.log`.
2. scp both tool sources to `~/src/mil-oneop-proof-20260831/` on 16m1mbp.
3. Built the compile tool (exact command):

```
xcrun clang++ -std=c++17 -fblocks -framework Foundation \
  -F/System/Library/PrivateFrameworks -framework ANECompiler \
  ane-compile-hwx.mm -o ane-compile-hwx
```

Result: `BUILD_OK`, binary 52336 bytes.

4. Authored the MIL program by hand. Dialect ground truth: the retained
   coremlc capture `/private/tmp/aneforge-add-cache/c3f45fbb2c54e9fbbad2cc58/model.mil`
   on 16m1mbp (read-only). Shape `[1, 512]` is mine; the captured program used
   `[1, 2048]`. A `[1, 2048]` twin was also compiled as a variant probe.
5. Compiled, converted, hashed.

### coremltools plumbing probe (negative result)

`author_mil.py` built the same add via `mb.program` (MIL Builder, explicit
program syntax). coremltools 9.0 serializes to the new `main[CoreML8](...)`
format for every opset (iOS16/17/18) and cannot emit legacy `program(1.3)`
text. The MIL text was therefore authored directly in the coremlc dialect.
coremltools played no role in the delivered artifacts.

## MIL source (verbatim)

Primary, `capture/model.mil` (438 bytes, sha256 `0417bba9...3151`):

```
program(1.3)
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3520.4.1"}, {"coremlc-version", "3520.5.1"}})]
{
    func main<ios18>(tensor<fp16, [1, 512]> t1) {
        tensor<fp16, [1, 512]> t0 = const()[name = string("t0"), val = tensor<fp16, [1, 512]>(BLOBFILE(path = string("@model_path/weights.bin"), offset = uint64(64)))];
        tensor<fp16, [1, 512]> t2 = add(x = t1, y = t0)[name = string("t2")];
    } -> (t2);
}
```

Variant, `capture-2048/model.mil` (442 bytes): identical with `[1, 2048]`.

The `buildInfo` line is a serializer fingerprint replicated from the retained
capture for dialect fidelity. It was not produced by running coremlc. The
program itself (function signature, const, add op) is hand-authored.

### weights.bin format (reverse-engineered, 1152 bytes for [1, 512])

| Offset | Size | Field | Value |
|---|---|---|---|
| 0x00 | 4 | version | 1 |
| 0x04 | 4 | entry count | 2 |
| 0x08 | 56 | zero pad | 0 |
| 0x40 | 4 | blob magic | 0xDEADBEEF (LE `ef be ad de`) |
| 0x44 | 4 | blob version | 1 |
| 0x48 | 8 | blob byte size | 1024 (= 512 fp16) |
| 0x50 | 8 | reserved | 0 |
| 0x58 | 8 | data absolute offset | 128 (0x80) |
| 0x60 | 32 | zero pad | 0 |
| 0x80 | 1024 | raw fp16 data | 0.25 repeated (`0x3400` LE) |

Values: all 0.25, so the op computes `out = in + 0.25` per element when run.

## Compile commands and output

```
$ cd ~/src/mil-oneop-proof-20260831
$ ./ane-compile-hwx capture hwx-output h13
callback_status=0
ANECCompile=0 callback_status=0        # EXIT=0, model.hwx 32768 bytes
$ ./ane-compile-hwx capture-2048 hwx-output-2048 h13
callback_status=0
ANECCompile=0 callback_status=0        # EXIT=0, model.hwx 32768 bytes
```

Negative probe (inline fp16 const list instead of BLOBFILE):

```
$ ./ane-compile-hwx capture-inline hwx-output-inline h13
callback_status=1
ANECCompile=1 callback_status=1        # EXIT=1, no model.hwx
```

The compiler rejects an inline `tensor<fp16, [1, 512]>([0.25, ...])` const.
The BLOBFILE + weights.bin form is required for this op. The failed attempt
left a 0-byte `model.hwx.additional_weights.bin` sidecar.

## HWX structure (compiled artifacts)

Both HWX files share the container shape of the repo's known-good
`tools/fresh-w4.hwx.sample`: header word 0 = `0xBEEFFACE`, 12 load commands,
`__TEXT` payload at file offset 0x4000.

| Section | [1,512] build | [1,2048] build |
|---|---|---|
| `__FVMLIB,__const` (input) | addr 0x30000000 size 0x400 | same addr, size 0x1000 |
| `__FVMLIB,__data` (output) | addr 0x30004000 size 0x400 | same addr, size 0x1000 |
| `__TEXT,__text` (task stream) | size 0x1F8 | size 0x1F8 |
| `__TEXT,__const` (kernel) | size 0x400, 457 nonzero bytes | size 0x1000, 1993 nonzero bytes |

I/O spans equal the authored geometry exactly (512 or 2048 fp16 elements).
The kernel sections are populated, so the graphs are real compiled work, not
stubs.

### Deviation source: TD flag-word drift

`find_task_offsets` seeds on the exact word `0xF401F800` at task+0x28.
The fresh compiler emits `0x4401F800` there (bits 28 and 30 differ:
`0xF4` vs `0x44` top byte). The TD preamble bytes 0x00-0x27 are identical to
the sample. Both of today's builds and both shapes show the same variant, so
it is deterministic for this graph family, not a botched compile.

Unmodified converter failure (stock tool, run first):

```
$ .venv7/bin/python hwxv2-to-anec.py hwx-output/model.hwx model.anec 512 512
ValueError: expected at least one task descriptor in __TEXT
EXIT=1
```

## Deviations

1. **Converter patch, scratch copy only.** `hwxv2-to-anec-patched.py` adds
   `TD_MAGIC_CURRENT = 0x4401F800` and accepts either TD word. Full diff
   against the copied original:

```diff
10a11
> TD_MAGIC_CURRENT = 0x4401F800  # mil-oneop: macOS 26.6.2 ANECompiler 9.509.0 flag variant
94c95
<     magic = struct.pack("<I", TD_MAGIC)
---
>     td_magics = (TD_MAGIC, TD_MAGIC_CURRENT)
98c99
<         if data[offset:offset + 4] == magic
---
>         if struct.unpack_from("<I", data, offset)[0] in td_magics
233c234
<         == TD_MAGIC
---
>         in (TD_MAGIC, TD_MAGIC_CURRENT)
```

   The repo checkout is untouched. Caveat: the converted `.anec` has not run
   on the Linux ANE; on-device parity is the next gate.

2. **MIL authored directly, not via coremltools.** coremltools 9.0 cannot
   serialize `program(1.3)` text (emits `main[CoreML8]`). The task allowed
   coremltools as plumbing only; plumbing was unnecessary because the tool
   consumes MIL text directly. See negative probe above.

3. **buildInfo line replicated** from the retained capture (see above).

4. **`[1, 512]` primary shape is mine.** A `[1, 2048]` twin matches the
   retained capture's shape and compiled identically well.

5. **venv7 coremltools 7.2 is not importable** under Python 3.12 (no
   `distutils`). It was unused by every delivered step; the venv supplied
   numpy for fp16 encoding and ran the stdlib-only converter.

## Conversion output

```
$ .venv7/bin/python hwxv2-to-anec-patched.py hwx-output/model.hwx model-512.anec 512 512
wrote=model-512.anec content=0x4000 task-stream=0x1f8 td-count=1 td@content+0x0 workspace=0x0 input=0x400 output=0x400 kernel@content+0x200 (0x400B) kdma-enabled=[] ...
EXIT=0

$ .venv7/bin/python hwxv2-to-anec-patched.py hwx-output-2048/model.hwx model-2048.anec 2048 2048
wrote=model-2048.anec content=0x4000 task-stream=0x1f8 td-count=1 td@content+0x0 workspace=0x0 input=0x1000 output=0x1000 kernel@content+0x200 (0x1000B) kdma-enabled=[] ...
EXIT=0
```

Independent header re-parse of both `.anec` files (against `_build_header`'s
`<QIIQQII32I192Q>` layout) confirms: td_size 0x274, td_count 1, task_stream
504, kernel 0x400 / 0x1000, 1 input + 1 output port, NCHW `(1,512,1,1)` /
`(1,2048,1,1)` with strides 64/64, header padding zero, content 0x4000 bytes
copied verbatim from `__TEXT`.

## Artifact table (sha256, macOS `shasum -a 256`, verified identical locally)

| Artifact | Bytes | sha256 |
|---|---|---|
| capture/model.mil | 438 | `0417bba9b253e8ada7ab60357aaa7cd05d0000dde58d2c578e6174c310ad3151` |
| capture/weights.bin | 1152 | `9436a9563e7e7c17767424814e9d418774cad1a507715a75891a15c24c6c6143` |
| hwx-output/model.hwx | 32768 | `1d6ccb51b3dbf15c80269b3ce4af1fe46459cfe2a65fcac9cc6931e0495a6e73` |
| model-512.anec | 20480 | `479a30e8197bce9130f5d78939ab2862b83d015ca27e8623b8855747981f527e` |
| hwx-output-2048/model.hwx | 32768 | `a482cbb5bdbf6a15c91c7bfea6462815568b8b26f66fc1905a6b27506b459555` |
| model-2048.anec | 20480 | `d541a664e9019d6276810fec778402bb67e174bd5d15b4ef5bdba0874b0f7abf` |
| capture-inline/model.mil (rejected) | 3439 | `a044bb52e27c8cfa69893537ac37cd46c7a3bfb8ee5c102958c17cfadb608bcf` |

## Scratch dir on 16m1mbp (left in place, reproduction environment)

```
~/src/mil-oneop-proof-20260831/
  ane-compile-hwx            52336 B  built tool
  ane-compile-hwx.mm          3009 B  copied source
  hwxv2-to-anec.py           15844 B  copied source (unmodified)
  hwxv2-to-anec-patched.py   15981 B  patched scratch copy (deviation 1)
  author_mil.py               1463 B  coremltools probe (negative result)
  gen_oneop.py                1852 B  MIL + weights generator
  capture/                    model.mil + weights.bin          [1, 512]
  capture-2048/               model.mil + weights.bin          [1, 2048]
  capture-inline/             model.mil + empty weights.bin    (compiler-rejected)
  hwx-output/model.hwx        32768 B
  hwx-output-2048/model.hwx   32768 B
  hwx-output-inline/          0-byte sidecar only (failed compile)
  model-512.anec              20480 B
  model-2048.anec             20480 B
  .venv/                      py3.12, coremltools 9.0, numpy
  .venv7/                     py3.12, coremltools 7.2 (unimportable), numpy
```

## Local files written

All under `/home/joshuawarren/src/mlx-omarchy/receipts/` (nothing committed):

```
2026-08-31-mil-oneop-proof.md            this receipt
fixtures/mil-oneop/gen_oneop.py          MIL + weights generator
fixtures/mil-oneop/author_mil.py         coremltools probe
fixtures/mil-oneop/model.mil             authored MIL [1, 512]
fixtures/mil-oneop/model-2048.mil        authored MIL [1, 2048]
fixtures/mil-oneop/model-inline.mil      rejected inline-const MIL
fixtures/mil-oneop/weights.bin           1152 B blob
fixtures/mil-oneop/model.hwx             32768 B  ([1, 512] compile)
fixtures/mil-oneop/model-2048.hwx        32768 B  ([1, 2048] compile)
fixtures/mil-oneop/model-512.anec        20480 B
fixtures/mil-oneop/model-2048.anec       20480 B
```

## Stop-gate conclusion for U6

The compiler does not reject a hand-authored one-op MIL `program(1.3)`.
`ANECCompile` returns 0 with `callback_status=0` and writes a HWX whose task
stream, kernel, and I/O geometry match the authored program. MLX-to-MIL
lowering (KTD6) can proceed on this path.

Two design constraints for the lowering work:

1. **Pin or widen around the TD flag word.** `hwxv2-to-anec.py` seeds task
   discovery on `0xF401F800`; this ANECompiler (9.509.0) emits `0x4401F800`
   for a single-task add graph. Either pin the export host's ANECompiler
   version or widen the TD-magic match as in the scratch patch. The Aug 28
   Qwen exports on this same OS build produced `0xF401F800` TDs, so the flag
   varies with graph family as well as compiler state.
2. **Const weights must ride weights.bin via BLOBFILE.** Inline fp16 const
   tensors are rejected (`ANECCompile=1`). The weights.bin blob layout above
   (0x40-byte file header, 0x40-byte `0xDEADBEEF` blob record, payload at
   0x80) is what the lowering must emit.

Not covered by this gate: on-device execution of the converted `.anec` on the
Linux ANE, and multi-op MLX region lowering.
