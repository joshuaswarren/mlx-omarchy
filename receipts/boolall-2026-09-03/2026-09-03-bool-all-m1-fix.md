# 2026-09-03 — bool All wrong past the first word on M1 (root cause + fix)

Agent: BoolAllFix. Worktree off 7942b94, uncommitted (Main commits).

## Verified defect (all numbers below observed this session)

- Machine: jwm1, MacBook Pro (13-inch, M1, 2020), Apple M1 (G13G B1),
  Honeykrisp, Mesa 26.1.7, Vulkan apiVersion 1.4.354, kernel
  7.1.6-1-1-ARCH, Python 3.14.7. One core online; builds -j1 nohup.
- Reference machine: dev box x86_64, llvmpipe (LLVM 22.1.8), Mesa with
  MLX_OMARCHY_ALLOW_NON_APPLE=1. Green on every check, unfixed and fixed.
- Source commit of the measured builds: 7942b94 (unfixed baseline),
  7942b94 + this worktree's shader fix (fixed). Not the wheel: both
  builds were from source, prepared by scripts/prepare-mlx.sh.

### Unfixed size table (source build, probe receipts/boolall-2026-09-03/probe_boolall.py)

Whole-array, 1-D, all-true input, mx.all:

- n=1..4: correct (true).
- n=5..4097: wrong (false) at every size probed:
  5,6,7,8,9,16,17,31,32,33,34,63,64,65,127,128,129,255,256,257,511,512,
  1000,4095,4096,4097. Correctness never returns at any larger size.
  4096/4097 straddle the chunk split (scratch + combine phase); wrong in
  both phases.
- All all-false input: correct (false) everywhere.
- One false inside the first word: correct (false) everywhere.

Mixed inputs, whole-array:

- false@4,false@8 (rest true): mx.all correct-answer false, wrong-reason —
  reads past word 0 return falsy, so these cannot distinguish.
- single true@4 (rest false): mx.any returns false for every n>=5. The
  earlier claim "mx.any is correct in both polarities" only covered
  all-true and all-false inputs.

Axis reduce (2,n) grid, axis 1:

- n=3,4: row 1 wrong (false) — fails EARLIER than whole-array because row
  1 already touches word 1 (for (2,3) row 1 starts at packed element 3).
- n>=5: row 1 wrong at every probed size.

Per-position misread map, unfixed kernel, from mx.any(single true@k):

- n=9: positions 4..8 read falsy (should be true).
- n=33: positions 4..15 and 20..31 read falsy. Positions 0..3 (word 0)
  and 16..19 (word 4) read correctly. Word 4 correct is a 16-byte-aligned
  window — a driver codegen fingerprint, and the reason "the kernel reads
  only the first word" was impossible as stated: word 0 of all-true data
  is nonzero, so an extent-clipped read would return true, not false.

mx.logical_and(ones, ones) at n=33 and n=65: correct on the same buffers,
unfixed kernel. compare_bool.comp reads the same word-packed bool inputs
through BYTE_AT, so input layout and transport are correct; the defect is
in the reduction kernel's byte extraction.

### Root cause

overlay/mlx/backend/omarchy/shaders/reduce_general.comp, load_truthy()
(7942b94 lines 166-167): packed-bool extraction used shift-then-mask with
a data-dependent shift amount,

    uint word = input_data.values[(params.lhs_offset + index) >> 2u];
    return ((word >> (((params.lhs_offset + index) & 3u) * 8u)) & 0xFFu) != 0u;

Honeykrisp miscompiles this construct family (receipt
2026-09-02-masked-scatter-m1-fix, diag68: reproduces with a literal
constant input; data-independent of the value). It is long-standing, not
a 959c7a0 regression: Release031 installed the published v0.3.0 aarch64
wheel on jwm1 in a fresh venv — mx.all wrong at n=5 and n=33 there — and
959c7a0 never touched reduce_general.comp (git log --follow: only 82c01cd
and 6cef85b). llvmpipe compiles the same SPIR-V correctly, which is why
the dev box never saw it.

Host dispatch (primitives.cpp dispatch_reduce_general) is correct: it
passes reduce_size in elements and one packed output word per 4 outputs;
llvmpipe answers correctly with the identical push constants.

### Fix (this worktree)

reduce_general.comp: replaced the dynamic shift-then-mask in load_truthy
with the receipt-proven BYTE_AT macro (four constant shifts picked by a
select chain), the same construct compare_bool.comp has used since
959c7a0. The device-context caveat from select.comp (where BYTE_AT was
wrong) is answered by measurement: the fixed kernel passed every check
below on the M1.

### Fixed verification on the M1 (same source build, only the shader changed)

- probe_boolall.py: 366 checks, 0 failed (was 85 failed). Misread maps
  empty at n=9 and n=33.
- omarchy_reduce_ops_tests: 22/22 cases, 6834/6834 assertions (was 21/22
  cases, 37 failed assertions, all in the new word-boundary test).
- Full battery, 24 binaries, two passes each: pass1 FAIL=0, pass2 FAIL=0.
  Pass-2 totals: 408 test cases, 828679 assertions, 0 failed. Per-binary
  counts in receipts/boolall-2026-09-03/m1-battery-fixed.log.

llvmpipe reference (same probe): 366 checks, 0 failed, unfixed and fixed.

## Files

- overlay/mlx/backend/omarchy/shaders/reduce_general.comp (fix)
- overlay/tests/omarchy/test_reduce_ops.cpp (regression test, see below)
- receipts/boolall-2026-09-03/: probe_boolall.py, jwm1-verify.sh,
  m1-table-unfixed.txt, m1-table-fixed.txt, llvmpipe-table-fixed.txt,
  m1-battery-fixed.log, this file.

## Coverage gap closed

TEST_CASE "bool Any and All stay exact across word and chunk boundaries"
in overlay/tests/omarchy/test_reduce_ops.cpp: sizes
4,5,8,9,32,33,64,65,256,257,4096,4097; all-true, all-false, false inside
the first word, false at first byte of words 1 and 2, single true at
byte 4; whole-array and (2,n) axis reduce; int32 AnyAll control.
Demonstrated failing on the unfixed M1 build (37 assertions) and passing
on llvmpipe and on the fixed M1 build.
