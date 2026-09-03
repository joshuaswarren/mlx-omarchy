# 2026-09-02 — four M1-red suites: root causes, fixes, on-device proof

Agent: M1RedInvestigation. Machine producing every on-device number below:
jwm1, Apple M1 (G13G B1), Honeykrisp, Vulkan 1.4.354. Reference numbers:
dev box llvmpipe (LLVM 22.1.8) through the same suites with
MLX_OMARCHY_ALLOW_NON_APPLE=1. Baseline worktree: /tmp/m1red-baseline at
clean f449ed2 (own worktree; profiling harness absent).

## Clean-HEAD baseline (the number the harness question needed)

At f449ed2, CPU=OFF, no env flag, 1 run each — counts identical to
PerfProfileBaseline's 008e86b runs with profiling on AND off, case lists
and line numbers byte-identical:

- omarchy_scatter_determinism_tests: 15 of 21 cases red (52 assertions, 4 failed)
- omarchy_select_layout_tests: 3 of 11 red (768 assertions, 13 failed)
- omarchy_eq_math_tests: 1 of 6 red (92 assertions, 15 failed)
- omarchy_primitive_tests: 1 of 86 red (604733 assertions, 28 failed)

Verdict: nothing here is caused by the profiling harness, and the
CPU-backend flip changed nothing on hardware.

## Root cause 1 — VK_EXT_shader_atomic_float never existed on the M1

The M1 Honeykrisp does not advertise VK_EXT_shader_atomic_float or
atomic_float2 at all (vulkaninfo, jwm1: the M1 extension list has neither;
the AtomicFloat feature blocks in vulkaninfo output belong to the
llvmpipe device on the same box — the earlier "measured on both" note read
llvmpipe's list). 12 of the 15 scatter reds were the capability gate
refusing fp32/fp16 float Scatter Sum/Prod and ScatterAxis Sum by name.
Correct refusal, missing fallback.

Fix: FCAS kernel variants. scatter.comp/scatter_axis.comp compile a
second FADD family without `#extension GL_EXT_shader_atomic_float`, whose
op 17 is a compare-exchange fp32 add through the uint view of the fp32
scratch — the exact twin of the op-13 CAS multiply that never needed the
extension. Prod is ungated (op 13/16 unchanged). The host picks HW vs
FCAS per `shader_atomic_float_add` and op 11 vs 17 with it. The op-layer
test at test_scatter_determinism.cpp:478 additionally had an invalid
updates shape (masked on llvmpipe by its early return); rewritten to pin
the fallback: on a device without the feature, float scatter_add must
still compute exact totals.

## Root cause 2 — shift-then-mask byte extraction (receipt-proven class)

The masked_scatter receipt (2026-09-02) proved Honeykrisp miscompiles
`(word >> ((i & 3u) * 8u)) & 0xFFu` with a data-dependent shift amount.
The same construct survived in four more shaders and produced the
remaining wrong values:

- scatter.comp ops 1/6/7 (bool scatter RMW): two of six word-lane writes
  dropped in "scatter bool across word lanes" and "put_along_axis bool
  none across lanes".
- logical_or.comp lines 91/93: 13 of 33 bytes wrong in the 33-element
  LogicalAnd case (primitive suite) and 15 of 33 against a scalar true.
- select.comp float path condition read: 6 + 5 wrong picks in the
  broadcast- and strided-condition cases.

Fix: the receipt's own recipe — four constant shifts picked by a select
chain, as the `BYTE_AT` macro (macro, not helper: the same receipt keeps
per-thread helper calls out of the hot path on this driver). Applied in
scatter.comp, scatter_axis.comp, logical_or.comp, compare_bool.comp
(lhs_byte/rhs_byte), and the FLOAT path of select.comp.

## Finding 4 — the same construct miscompiles in OPPOSITE directions by context

This is the driver-receipt entry, and it means neither byte-extraction
form is safe by default on this hardware; every future byte-extraction
site must be probed per site, not pattern-matched:

- In logical_or.comp, scatter.comp ops 1/6/7, and select.comp's float
  path, the shift-then-mask form miscompiles and the BYTE_AT select-chain
  form is correct (suites went red -> green with proof repeated 3x).
- In select.comp's PACKED bool path, the plain shift-then-mask helper
  `byte_at` is CORRECT and the select-chain macro form miscompiles:
  with the macro, "where selects between packed-bool operands" failed 13
  of 13 checked positions and the dtype-sweep bool leg showed spurious 1s
  at 5/17; restored helper form: 0 wrong. Probe-proven on jwm1 with an
  8-variant probe (verbatim/pre-OR-densified/dense-built/all-true/
  all-false/inverted/dense-control) — all 0 wrong under the helper form.

The packed select.comp section therefore deliberately KEEPS the original
extraction form, with a comment pinning this so nobody "fixes" it into
the macro.

## Root cause 5 — Honeykrisp built-in sin/cos collapse for large arguments

Probe (scalar sweep through our own shader path, jwm1): error 4.2e-6 at
100, 2.8e-5 at 1e3, 3.6e-4 at 5e3, 4.5e-4 at 12345, 1.2e-3 at 2e4,
4.8e-3 at 123457, garbage from 1e6, exact zeros at 1e8 and above.
llvmpipe stays accurate far higher; the W8 receipt's "+/-1 saturation
band" is llvmpipe-shaped, not a contract.

Fix (redesigned twice under review; final design): a host-side gate,
`trig_argument_gate` in primitives.cpp. Every Sin/Cos GPU dispatch
reduces max|x|, reads it back, and REFUSES BY NAME above
`kTrigArgumentLimit = 1e5`. History of the limit, because each
iteration was evidence-driven:

1. First draft: in-shader saturation to +/-1 above 1e4. Withdrawn in
   review: a bounded-but-wrong value is a silent wrong value - ours,
   not the driver's - and strictly worse than the bug.
2. Second draft: named refusal above 1e4. Withdrawn when Main flagged
   the consumer: fast::RoPE is OMARCHY_USE_FALLBACK and composes
   exactly these sin/cos calls with inv_freq[0] = 1.0, so the limit is
   a positional ceiling. CONFIRMED ON HARDWARE: fast::rope at position
   12345 threw the gate's named error; 1e4 would have ended
   Qwen-class 32k-context decode ten thousand tokens in.
3. In-shader Payne-Hanek fallback (the retained W8 helpers, whose bit
   anchors carry an in-source fix comment): probed on the M1 and
   DISCARDED - sin(5e6) returned -7.9e15, sin(1e8) returned 8.2e12:
   the reduction's carry chain rides the dynamic-indexing shapes this
   driver miscompiles, the same family as Findings 2-4.
4. Final: refusal above 1e5. Criterion: the largest round power of ten
   below the measured collapse band with margin - measured built-in
   error 4.5e-4 at 12345, 4.8e-3 at 123457, 1e-2 and worse toward 1e6 -
   covering Qwen-class 32k contexts with 3x positional margin. The
   limit is stated as conservative: the built-in is still recognisably
   a sine at 123457, and positions past 1e5 now refuse rather than
   degrade. RoPE verified on hardware after the fix: fast::rope at
   positions 12345 and 100000 computes matching cos-sin hand
   references; the W8 test pins accuracy bands below 1e5 and the named
   refusal above it. Residual: sin/cos inside COMPILED tapes bypasses
   Sin/Cos::eval_gpu and inherits no gate; the compiled-tape suite
   uses small arguments only.

## On-device proof (all runs on jwm1 unless stated)

Final state, 3 repeated runs per suite per machine, every assertion
compared inside the suite:

- omarchy_scatter_determinism_tests: 21/21 cases, 127/127 assertions, 3/3
- omarchy_select_layout_tests: 11/11 cases, 768/768 assertions, 3/3
- omarchy_eq_math_tests: 6/6 cases, 108/108 assertions, 3/3
- omarchy_primitive_tests: 86/86 cases, 604733/604733 assertions, 3/3

CAS determinism specifically: the suite's own "run twice is bitwise
identical" case (bitwise CHECK_EQ across two dispatches) and the
integer-valued duplicate-total cases passed every run; the fractional
duplicate case (the only order-sensitive one) passed within its
documented tolerance 3/3. CAS-loop ordering nondeterminism affects only
last-bit rounding of non-exact addends — the same class upstream Metal's
native float atomicAdd is in; no ordering contract exists upstream
either.

llvmpipe regression runs (dev box, MLX_OMARCHY_ALLOW_NON_APPLE=1): all
green, 3 repeated runs each — scatter_determinism 21/21 cases 124/124
assertions (3 fewer than M1: case 478 early-returns where the capability
is present, by design), select_layout 11/11 768/768, eq_math 6/6 108/108
(the refusal path itself verified on llvmpipe: device-independent
contract), primitive 86/86 604733/604733. Logs: /tmp/m1red-lvp-*.log.

## v0.3.0 impact (facts for the release decision)

v0.3.0 ships from this tree, so the following M1-only behaviors are in
the published release:

- SILENT WRONG VALUES (raise nothing): bool scatter writes across word
  lanes (scatter_add/put_along_axis bool, >4 elements); packed-bool
  LogicalAnd/LogicalNot outputs crossing a 32-bit word; where() with
  broadcast or strided bool conditions (wrong operand picked); bool
  where over transposed operands; sin/cos of |x| beyond ~2e4 (garbage or
  zero from ~1e6).
- LOUD GAP (raises by name): float32/float16/bfloat16 scatter Sum/Prod
  and float ScatterAxis Sum refuse on the M1 for lack of the extension —
  the capability gate worked; only the fallback was missing.

## Files changed (no commits; Main integrates)

- overlay/mlx/backend/omarchy/shaders/scatter.comp — BYTE_AT, op 17, HW gate
- overlay/mlx/backend/omarchy/shaders/scatter_axis.comp — same
- overlay/mlx/backend/omarchy/shaders/logical_or.comp — BYTE_AT
- overlay/mlx/backend/omarchy/shaders/compare_bool.comp — BYTE_AT
- overlay/mlx/backend/omarchy/shaders/select.comp — float path BYTE_AT;
  packed path deliberately original (see Finding 4 comment)
- overlay/mlx/backend/omarchy/shaders/elementwise.comp — comment-only
- overlay/mlx/backend/omarchy/primitives.cpp — FCAS selection, scatter
  gate -> fallback, trig_argument_gate, Cos/Sin eval_gpu, ops.h include
- overlay/mlx/backend/omarchy/compute.h — FCAS enums, corrected comment
- overlay/mlx/backend/omarchy/compute.cpp — FCAS cases + includes
- overlay/mlx/backend/omarchy/CMakeLists.txt — USE_HW_ATOMIC flips + 7 FCAS variants
- overlay/mlx/backend/omarchy/device.h — corrected capability comment
- overlay/tests/omarchy/test_scatter_determinism.cpp — case 478 rewrite
- overlay/tests/omarchy/test_eq_math.cpp — W8 rewrite + helper
