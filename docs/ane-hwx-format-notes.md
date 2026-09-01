# HWX format notes from freedomtan/coreml_to_ane_hwx

External reference evaluated against our pipeline on 2026-09-01:

- Guide: <https://github.com/freedomtan/coreml_to_ane_hwx/blob/main/GUIDE_ANE_HWX_FORMAT.md>
- Repo: <https://github.com/freedomtan/coreml_to_ane_hwx> (BSD 3-Clause)
- Web parser: <https://freedomtan.github.io/coreml_to_ane_hwx/hwx_dump_js/>

This doc maps what the guide documents onto what we already proved. Nothing
here is copied verbatim; each point cites the guide section it paraphrases.

## What our pipeline already knows (ground truth)

- MIL compile path: hand-authored `program(1.3)` MIL plus `weights.bin`
  through `ANECCompile`; inline fp16 const tensors are rejected
  (`receipts/2026-08-31-mil-oneop-proof.md`).
- `weights.bin`: 0x40-byte file header, 0x40-byte `0xDEADBEEF` blob record,
  payload at 0x80.
- TD discovery: `hwxv2-to-anec.py` seeds on word `0xF401F800` at task+0x28;
  ANECompiler 9.509.0 emits `0x4401F800` for some graph families (same
  receipt, deviation 1).
- Container: `0xBEEFFACE` Mach-O magic, `__TEXT,__text` task stream,
  `__TEXT,__const` kernel; bundle/manifest contract in `docs/ane-bundles.md`.

## What the guide confirms

1. **Mach-O container.** The guide (section 2) matches our receipts: the
   header magic may be `0xBEEFFACE` or standard `0xFEEDFACF`, segments are
   found via `LC_SEGMENT_64` (`cmd == 0x19`), and the task stream lives in
   `__TEXT/__text`. Our `find_task_offsets` and the converter's section
   handling agree.
2. **`cpusubtype` names the generation.** Section 1 tabulates
   `cputype 0x80` (or ARM64) with `cpusubtype` 1..10 mapping H11..H18. Our
   h13-targeted compiles and the converter's hwx v2 path sit inside this
   table; the converter can read `cpusubtype` to pick encodings instead of
   assuming one generation.
3. **Kernel weights ride the Mach-O.** Section 2 puts weights and scales in
   `__KERN`; our compiled HWX carries the kernel in `__TEXT,__const`. Same
   fact, different section layout per compiler version. The guide's point
   stands: kernel payloads are Mach-O sections, not part of the TD stream.

## What the guide adds (new to us)

1. **16-byte section header before the first task** (section 3). The guide
   says `__TEXT/__text` starts with a 16-byte header (signature word 1,
   three reserved zero words), so tasks begin at section offset 0x10.
   Our converter seeds on the TD magic instead and never needed this, but
   the exporter and any future parser should expect the header.
2. **Zero-size task blocks are padding** (section 3). A task word 0 with
   `task_size == 0` is 16-byte alignment padding; skip 16 bytes and retry.
   This gives the converter a magic-independent task walk: parse
   `task_size` from the header, step to the next 16-byte boundary, and
   treat `0xF401F800`/`0x4401F800` as one field of the header rather than
   a magic seed.
3. **TD header field layout** (section 3). The header is 32 bytes for
   H14/H15 and 36 bytes for H16+ (`dtid`, dependent task ID, appended).
   Fields: `tid`, `task_size:11`, `exe_cycles`, log/exception masks,
   `live_outs`, and control flags (`tsr`, `tde`, `ene`). The guide does not
   name the word at task+0x28, so it does not explain our TD flag drift
   directly. But the layout confirms the drift word is a flags/live-outs
   class field, not a structural offset, which supports widening the
   converter to accept both values.
4. **Instruction stream encodings** (section 4). H13 uses
   `[31:26] count, [25:0] byte address`. H14+ uses two formats keyed on
   bit 31: dense (`[20:15] count`, `[14:0] word address`) and sparse
   (`[30:15] 16-bit mask`, `[14:0] word address`). Addresses are word
   indices. This is the decode rule for the task streams our converter
   currently ships opaque, and it is what a Linux-side validator would need
   to check KDMA/DMA setup before submit.
5. **Stateful register carryover** (section 5). Registers persist across
   tasks; a parser must keep one 8192-word register array for the whole
   file. This explains why single-task graphs (our one-op proof) look
   self-contained while multi-task graphs inherit ChannelCfg and friends.
6. **Register map and defaults** (sections 5 to 7). Block base addresses
   per generation (Common, L2, PE, NE, TileDMA, KernelDMA), the ChannelCfg
   data-format encoding (`0x0` INT8, `0x2` FLOAT16), and architecture
   defaults: H14/H15 omit ChannelCfg and run INT8; H16+ defaults to FP16.
   Our h13-target graphs are FP16 by MIL construction; this table says the
   compiled stream must carry an explicit ChannelCfg there, which is a
   checkable exporter invariant.
7. **PE and NE semantics** (sections 6 and 7). `PE_Config` op field
   (0=add, 1=multiply, 2=max, 3=min), pooling modes, activation encodings
   in `MacCfg`/`PE_Config`, and the `task_type` activity heuristic that
   filters inherited-but-inactive PE state. This maps our compiled add/mul
   graphs onto named hardware operations and gives a way to verify the
   kernel section actually implements the authored op.

## What the guide does not cover

- The exact TD flag words `0xF401F800` / `0x4401F800`. The variance stays
  an empirical finding of `receipts/2026-08-31-mil-oneop-proof.md`.
- The MIL `program(1.3)` dialect, `BLOBFILE` consts, or the `weights.bin`
  blob layout. The repo works at the Espresso/`net.plist` layer upstream of
  us and never needed hand-authored MIL.
- Multi-task dependency execution on Linux, or `dtid` semantics beyond
  "seldom used, usually 0".
- Why the compiler rejects inline const lists; the guide predates and
  skips MIL-level behavior.

## Suggested follow-ups for the exporter

1. Add the magic-independent task walk (section 3 rules: 0x10 section
   header, `task_size`-driven stride, 16-byte padding skip) to
   `hwxv2-to-anec.py` as a fallback when neither TD flag word matches.
2. Add a unit test that converts an H14+ sparse-format task stream; the
   current converter path has only exercised dense-format streams.
3. For multi-op regions, emit per-task `tid`/`dtid` in the export receipt
   so task dependency structure is visible without a full register decode.
4. Cross-check the compiled add graph against the guide's `PE_Config`
   op=0 encoding once the H13 register bases for our compiler version are
   confirmed; a mismatch would catch a kernel-section regression.

## Attribution

All format facts above paraphrase the cited guide sections. The upstream
repo is BSD 3-Clause; quote limits respected, no code copied.
