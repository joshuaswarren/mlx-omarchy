# Wrong-value sweep — 2026-09-11

Branch `wrongvalue/20260911` (pushed), base `bddc061f` (= receipt source
`2a9add42` content; `git diff 2a9add42 bddc061f -- overlay patches` is
empty). Host: x86_64 dev box, llvmpipe/lavapipe, `MLX_OMARCHY_ALLOW_NON_APPLE=1`,
same runner invocation as the 2026-09-11 upstream receipt (`pytest` per file,
`DEVICE=gpu`, `MLX_ENABLE_TF32=0`). M1 verification: not yet run (single GPU,
window queue — see "Pending M1 work").

## Fixes landed (5 of the 6 wrong-value cases + the pooling case)

### (b) `mx.convolve` f32 M=24 N=4 mode=same — FIXED. Cause named: `5facd59a`.

Commit **`2a198ea1`** (revert of `5facd59a` "batch host scalar GPU fills",
which is inside the regression window).

Mechanism: `mx.convolve(mode='same')` with an even kernel pads `(N/2, N/2-1)`
→ the pad's zero fill is a host scalar fill. `5facd59a` removed the
`encoder.synchronize()` that preceded every scalar fill and replaced the
guarantee with an in-buffer dependency barrier before `vkCmdFillBuffer`.
The barrier does not re-establish what the drain provided: with the drain
gone, the fill's write to freshly recycled storage is lost and the buffer
keeps the previous occupant's bytes. Outputs 0, 1, 23 (the only outputs
whose windows touch the padded bytes) then read recycled garbage. Standalone
runs pass only because fresh pages are zeroed; inside any suite the
allocator churns and the subtest fails — exactly the "state-dependent"
signature in the 2026-09-11 receipt.

Evidence:
- Deterministic suite repro on llvmpipe: `pytest test_conv.py` red at HEAD,
  green with the revert, 3× each (`before-llvmpipe/`, `after-llvmpipe/`).
- Isolated probe (`conv_alloc.py` in this receipt): allocate/free churn then
  `mx.convolve(a24, v4, 'same')` → 38/40 failing iterations at HEAD, all
  failures at indices {0,1,23} with values that change per call (recycled
  memory), CPU stream correct in the same process.
- A full ALL_COMMANDS barrier **after** the fill does NOT restore
  correctness (32/40 still failing) — only the pre-fill drain does. The
  batching commit's author missed that the fill's correctness rested on the
  drain, not on command ordering.
- The same mechanism covers **(d) MaxPool1d k2 s2 p1** (`test_pooling`
  green in `after-llvmpipe/test_nn`): pooled padding routes through the
  same scalar fill.

### (d) MaxPool1d k=2 s=2 pad=1 — FIXED by the same revert. See (b).

### (e) compile value_and_grad + optimizer step; dynamic dims — FIXED.

Commits **`78e19770`** + **`c3d5286d`** (cherry-picks of the unmerged
`wave/UpstreamSuiteFix` fixes `c807794b`, `8f717d56`): the fused-chain leaf
shader addressed direct/mod-last/div-last leaves as row-major linear
buffers while `encode_leaf` admitted any contiguous array, so a
column-contiguous (transposed) leaf fused in storage order and returned
permuted values; the narrow fix re-cuts the guard to direct mode and adds
tiled addressing for broadcast mod-last/div-last leaves.

- `test_optimizers::test_compiled_optimizer` passes with
  `MLX_OMARCHY_FUSED_CHAIN=0` at HEAD (fusion bisect, run this session) →
  fused-chain root cause confirmed before the cherry-pick; passes
  unconditionally after (`after-llvmpipe/test_optimizers`: 25 passed).
- `test_compile::test_compile_dynamic_dims` passes after
  (`after-llvmpipe/test_compile`: 64 passed / 3 failed — the 3 remaining
  are the tracked bf16 compiled-tape refusals, see (a) note below).

### (a) subnormal f32/bf16 `astype(bool)` — FIXED in `edf4807f`, M1 gate pending.

Cause: the cast shader computed `static_cast<bool>(x)` as a float compare
(`x != 0.0`). Vulkan grants denorm preservation only opt-in
(`DenormPreserve`); the Honeykrisp **fork** (Mesa 26.3.0-devel.hk6f6afc8,
installed on the M1 between the snapshots per
`receipts/2026-09-08-honeykrisp-package.json`; the 2026-09-06 snapshot ran
stock Mesa 26.1.7 per `receipts/m1-qual-2026-09-06/summary.json`) flushes
subnormal operands, so f32 subnormals — and bf16 subnormals, which ride the
same f32 subnormal encoding — compared `!= 0.0` as zero → False. The f16
path survives because f16 subnormals decode to *normal* f32 values. The
latent hazard sat in both the old cast blob and the completed conversion
matrix; **the driver upgrade is what exposed it**, which is why no overlay
bisect can produce a commit. Author's miss (across both casts reworks):
relying on IEEE denorm behavior Vulkan does not promise.

Fix (`edf4807f`): truth from encoded bits — nonzero mantissa/exponent bits
means true; covers subnormals of any width, NaN stays true, both zeros
stay false. Verified on llvmpipe: subnormals True, ±0 False, NaN True,
±inf True (edge probe in `cast_int.comp` build, plus the new C++ test).
LLVMPIPE LIMITATION: llvmpipe preserves denorms, so llvmpipe cannot show
the pre-fix failure — the fails-before evidence is the 2026-09-11 M1
receipt itself (`test_ops.py::test_subnormal_bool_cast` AssertionError:
`f32_sub.astype(mx.bool_)` False). **The M1 post-fix run is the required
gate and has not happened yet.**

Note on the compile variant (`test_compile::test_compiled_subnormal_bool_cast`):
its 09-11 wrong-value assert was the fused-chain bug (e) and passes the
wrong-value state with the cherry-picks; the case now lands back on the
documented standing bf16 compiled-tape refusal (same bucket as
`test_inf_constant` / `test_compile_nonfinite_constants`, tracked in
`docs/known-defects.md`). Not a wrong value; not in this sweep's fix count.

## Explicitly declined this window (evidence attached)

### (c) `mx.sort` wrong order on 0-strided broadcast input — NOT FIXED; cause narrowed.

Facts established this session:
- The upstream failing subtest is the 0-strides block
  (`np.array([1,0,2,1,3,0,4,0])`, default int64, `broadcast_to((16,8))`,
  axis 0/1) — from the 09-11 receipt junit, line 2885.
- HEAD refuses int64 sort by name (`require_sort_dtype` has no int64
  branch; identical at `2a9add42`) — fresh-process repro refuses on
  llvmpipe (`sort_suite_repro.py`).
- Yet inside the full `test_ops.py` run, the same block fails as an
  **assertion** (`after-llvmpipe/test_ops.log`), i.e. the int64 sort
  executed and returned wrong order — reproduced on llvmpipe, matching the
  M1 receipt. Isolated `test_sort` alone refuses. So a **precursor test in
  `test_ops.py` unlocks the int64 sort past `require_sort_dtype`** (same
  suite-state dependence class as (b); probe instrumentation confirms
  `Sort::eval_gpu` is entered with `dtype.val()==8` (int64) and no throw).
- Backend instrumented probes are NOT in the pushed branch (staged
  experiments only; `git checkout`ed before commit). The unlock mechanism
  (which precursor, why the check passes) is the open question; the
  dispatch switch has no int64 case, so any unlocked int64 sort rides a
  float kernel on raw bits and orders garbage — consistent with the wrong
  order.
Next window plan: bisect `test_ops.py` prefixes before `test_sort`
(unittest loader prefix halving, ~7 runs) to name the unlocking test, then
trace which path skips `require_sort_dtype`.

## Per-file before/after counts (llvmpipe, this branch's base vs head)

Same runner shape as the receipt (per-file pytest, same module build path):
`before-llvmpipe/` captured at `bddc061f`, `after-llvmpipe/` at `05e15242`.

| file | before | after |
|---|---|---|
| test_conv.py | 25 passed / 1 failed (`test_numpy_conv`) | 10 passed, 10 skipped, 16 subtests passed — 0 failed |
| test_nn.py | 71 passed / 1 failed (`test_pooling`) | 72 passed, 1 skipped, 3 subtests — 0 failed |
| test_optimizers.py | 24 passed / 1 failed (`test_compiled_optimizer`) | 25 passed, 1 skipped — 0 failed |
| test_compile.py | 63 passed / 4 failed | 64 passed / 3 failed (all 3 = tracked bf16 tape refusals; dynamic_dims + compiled_subnormal wrong-value states gone) |
| test_ops.py | not run before (see note) | 158 passed / 2 failed (`test_sin` standing refusal; `test_sort` declined above) |

test_ops.py before-count note: the first before-run did not include
test_ops.py; the M1 09-11 receipt is the authoritative before state for it
(`test_subnormal_bool_cast` + `test_sort` failing there; the cast now
passes on llvmpipe and its M1 post-fix gate is pending).

## Regression tests

`overlay/tests/omarchy/test_wrong_value_sweep.cpp`
(`omarchy_wrong_value_sweep_tests`, registered in CMakeLists):
- float→bool cast bit-truth table (f32/bf16 subnormals, NaN, ±0, ±inf,
  f16 control) — fails-before evidence: the M1 receipt (llvmpipe preserves
  denorms and cannot show it).
- same-mode convolve + padded max-pool under allocator churn —
  **fails-before proven by rebuild**: with `5facd59a` re-applied the
  convolve case fails; with the revert it passes (measured both ways this
  session). 32/32 doctest assertions green at head.

## Pending M1 work (single GPU window, ~20 min, batched)

1. Warmup, then: run the exact `test_subnormal_bool_cast` snippet on the
   fork driver against the pre-fix wheel (fails-before, expected False),
   then install a wheel from this branch and re-run (expected True) —
   cast gate.
2. Suite re-runs: test_conv / test_nn / test_compile / test_optimizers /
   test_ops on the branch wheel; expect the llvmpipe after-table to hold.
3. Digest gates: the drain restore (`2a198ea1`) touches a shared fill
   path — run the six canonical Q4 digests + BF16 pins before landing.
4. `test_sort` unlock bisect (see declined section) if window time
   remains — else next window.
