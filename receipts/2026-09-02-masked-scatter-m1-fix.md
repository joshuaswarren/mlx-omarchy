# 2026-09-02 — masked_scatter M1 wrong-value fix (defects 3 and 4)

Agent: FixM1MaskedScatter. Machine producing every on-device number below:
jwm1, Apple M1 (G13G B1), Honeykrisp, Vulkan 1.4.357. Reference numbers:
dev box llvmpipe via the same probe (VK_ICD_FILENAMES=lvp).

## Defects

- Defect 3: "masked_scatter fills true positions from the source in
  order" — got dst 5 where -3 was scattered (write silently skipped).
- Defect 4: "masked_scatter carries the scan across chunks and rows"
  — 598 assertions; scattered negatives landed as 0 across chunks.

## Root cause (ours to trigger, driver miscompile underneath)

NOT the mask transport and NOT the scan barriers. Minimal repro chain,
all standalone Vulkan (no MLX), probe at /tmp/msprobe on jwm1:

1. Mask words arrive clean on the M1: raw SSBO reads show exactly
   0x00010001 (diag5/diag7). The `bit == 1u` vs 0xFF hypothesis died.
2. The scan shape alone is fine on the M1: scan_a (hardcoded bit) gives
   s_scan[255]=128, correct, as do unrolled / no-predicate /
   double-buffered / memoryBarrierShared variants (scan_b..scan_e).
3. The SAME scan fed by a bit produced with a single dynamic shift
   `(word >> ((lane & 3u) * 8u)) & 0xFFu` undercounts on the M1:
   s_scan[255]=32 instead of 128 — exactly a window-64 scan, i.e. later
   Hillis-Steele steps stop propagating (diag42, per-iteration dump in
   diag55). llvmpipe: 128. Reproduces with the shift on a literal
   constant (diag68), through shared memory (diag67), and with the
   shamt hoisted to a private var (diag83) — it is shift-then-mask with
   a data-dependent amount feeding the scanned value.
4. Works on the M1: constant shifts dispatched per lane (diag76
   if/else), byte-table select (diag81), uvec4 swizzle (diag82/84),
   mask-then-shift (diag86). Data-independent: diag68 fails with a
   constant 0x00010001 input.

So: Honeykrisp miscompiles this scan when its input bit comes from a
dynamic shift+mask; every skipped write then leaves the dst value in
place (defect 3's "got 5") and every undercounted carry across chunk
boundaries misroutes or drops writes (defect 4's zeros). The kernel's
barriers are uniform-scope and correct; llvmpipe hides nothing here
because the failure is codegen, not ordering.

## Fix (landed, uncommitted)

overlay/mlx/backend/omarchy/shaders/masked_scatter.comp — replace the
dynamic shift with four constant shifts picked by `flat_pos & 3u`
select chain, and normalize `bit = byte != 0u ? 1u : 0u` (hardening so
a stray non-0/1 true byte still counts; ScatterDeterminism's
row_base/matrix_m append is a separate hunk by them).

## On-device proof (probe, exact overlay shader bytes + push layout)

Defect 4 shape (2x600 → one global row of 1200, every other position
true, value -k), DUMP of out[0..8]:

- verbatim M1:  -0 0 -1 0 0 0 0 0  (and -2 at 16, -3 at 18) — BROKEN
- fixed M1:     -0 0 -1 0 -2 0 -3 0 -4 ... -11 at 22 — CORRECT
- fixed llvmpipe: identical to fixed M1
- fixed M1 repeated runs: 3/3 identical, correct

Defect 3 shape (dst {1..6}, mask {0,1,1,0,1,0}, src -k):

- verbatim M1:  1 -0 -1 4 5 6      (write skipped) — BROKEN
- verbatim llvmpipe: 1 -0 -1 4 -2 6 (reference)
- fixed M1:     1 -0 -1 4 -2 6 — CORRECT, matches reference

Note the probe's defect-3 rank maps to -2, not the suite's -3: the op
layer expands the 2-D mask + 1-D value into one global row, so the
suite's third true position consumes value[2] = -3. The mechanism and
fix are identical; the suite itself is the final word.

## Pending

omarchy_indexing_ops_tests + omarchy_primitive_tests runs on jwm1 are
blocked by the shared-tree build storm (missing compute.h/compute.cpp
enum appends from siblings; Main's stash recovery). jwm1's clone is at
6049968, predating masked_scatter entirely, so it needs an overlay sync
plus a single-core rebuild once the tree is coherent. The probe above
is the same shader bytes, same push-constant layout, same shapes, real
device — but the suite binary run is the remaining acceptance item.
