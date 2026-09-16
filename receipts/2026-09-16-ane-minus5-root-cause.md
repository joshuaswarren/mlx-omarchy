# ANE tm -5 root cause: the fused fold indexed bias out of bounds and shipped
# NaN island inputs; pair-based bias index fixes it — pin and 104/104 hold
# (2026-09-16)

Verdict: **LAND.** The fused bias(+silu) fold's kernels indexed the bias
buffer with the flattened output word (`bias[word * 2u]`, `word =
row*pairs + pair`) instead of the in-row pair (`bias[pair * 2u]`). Every
output row past the first therefore read out of bounds past the 1024-element
bias buffer: with the fix's one-line index change the fused output matches
the stock dispatch stream bit-exactly, the previously deterministically
wedged island pass completes clean, `encoder_hidden` = pin
`38c73261f29230276ed76f1fc017b76b024156d79218bd5f1347fdc7e7d43ec7`, and the
full Parakeet E2E holds 104/104. Landed as the fold ported onto current
main's fp32-partials linear path (3aa4f348 lineage), kill-switch
`MLX_OMARCHY_CHAIN_FUSION=0` preserved, default on.

## The exact difference (logged fields)

The trigger was data, not timing and not buffer identity — the briefing's
"inputs are byte-correct" premise was self-consistent (fused == fused), not
stock-equal. Shim-rig logging (stub libane, real worker, zero ANE submits,
fresh boot `e52e0f1c`):

- Shipped island payloads, stock vs fusion-on: 96 of 216 submits differed
  (q_v / q_scaled / k_headsT of all 24 island-A submits + the 24 pv-island
  rows). L00-A q_v already differed before any island round: 379844/384000
  elements mismatch (98.9 %), maxabs 2.34; L01-A inputs carried 15360 NaNs
  (stock: 0); L02-A 125952. `pos_kT` (constant) identical everywhere.
  Shapes, dtypes, nbytes, alignment identical throughout; `k_headsT` is a
  non-contiguous `swapaxes` view in BOTH runners — no view/alias/offset
  difference between runners (hypothesis 1 out).
- Worker side: BO identity identical stock vs fused (same handles, offsets,
  tile counts, e.g. bdx=0 tiles=11 size=0x2c000 offset=0x100000000 handle=1;
  bdx=4 tiles=282 …). Payload sizes are validated against the manifest
  before every send, and the -5 came from the submit ioctl after successful
  sends (hypothesis 2 out).
- GPU stream: 6762 vs 5890 primitive dispatches; unsynced ops at the L01-A
  ship 142 (stock) vs 139 (fused) — real but subordinate: the settle lane
  already showed pacing the doorbell does not help, and the data fix below
  removes the wedge without touching submit timing (hypothesis 3
  unnecessary).

Mechanism chain: OOB bias reads → wrong biases on every row > 0 (maxabs
6.0, 92 % of elements) → NaN cascade from L01 → the ANE island submits run
on NaN-laden inputs (L00-A/B/C execs) → the tm event FIFO floods → at the
L01-A submit the driver's ENTRY `ane_tm_collect_events` reads
`TM_IRQ_EVTC(line) > 64` and returns -EIO: `tm completion failed: -5,
finish lines=0` — finish lines 0 because the wedged submit never ran
(omarchy-ane `6fa243a`, `ane/src/ane_tm.c`).

## Isolation evidence (jwm1, `/var/tmp/ane-minus5`)

- Stub libane (`libane-shim/ane_shim.c`): eiln-ane `ane.c` with the submit
  ioctl replaced by a fake completion — the full encoder pass runs with the
  real worker, real BO allocation, zero doorbells. Both shim legs complete
  all 72 rounds.
- First divergent statement: stmt 123 `linear` (`linear_0_cast_fp16`,
  [1,375,1024]); its producer chain traces into the fused linear path.
- Operand repro on the real dumped x/weight/bias: stock path sha
  `1ccb0705151f76b8` == in-situ stock; fused path mismatch 355138/384000,
  maxabs 6.0, deterministic (5/5 same sha — not a race; shapes verified
  correct in-shader via a shape-probe kernel).
- Zero-bias isolation: fused kernel with zero bias matches the chain
  reference everywhere except 24 elements (the OOB reads) — the entire
  delta is the bias term.
- Fix (index `bias[pair * 2u]`/`[pair * 2u + 1u]` in both fused kernels):
  fused vs stock mismatch 0/384000 on the same operands.
- Full pass after fix: 216/216 shipped-payload shas identical stock vs
  fusion-on.

## Hardware gates (one real ANE leg, as mandated)

`flock -w 900 /tmp/m1-gpu.lock`, `--deadline-ms 20000`, real
`libane-strict.so`, fusion on: rc=0, 72 rounds, 0 timeouts, dmesg tm
failures 0, `encoder_hidden` sha256 `38c73261f29230276ed76f1fc017b76b…`
(pin). E2E (`fused_e2e.py`, fixed runner): status `match`, emissions
104/104, matching_prefix 104, tokens match, transcript sha `db501a8c…`,
mel `5b54f4a9a2ba3434…`, `control: gpu-loop`, `tdt_fallback_reason: null`,
cpu_tensor_events 0, total_pipeline 10560.5 ms. Reports:
`/var/tmp/ane-minus5/logs/real-out`, `…/e2e-out`.

## What landed

`overlay/tools/coreml/vulkan_encoder.py` at `3aa4f348` (current main had
reverted the coopmat f16-partials path, so the fold is ported onto the
fp32-partials batched matmul + `_leftover_chain_kernel`): the two fused
kernels (`_leftover_chain_bias_kernel`,
`_leftover_chain_bias_silu_kernel`, fp32-partials variants, pair-based
bias index), the `linear_silu`/`silu_done` coupling, the alias of the silu
statement's name to the fused linear output, and the
`MLX_OMARCHY_CHAIN_FUSION` kill-switch (default on). Port verification:
fusion-on vs fusion-off shim legs 0/216 payload sha diffs; fusion-off vs
the 69fd-lineage stock 0/216 (cross-lineage byte identity).

Evidence classes, stated plainly: the hardware leg + E2E validated the
fold+fix on the coopmat lineage (the runner the -5 repro lived in); the
port to the fp32 lineage is shim-byte-proven against current stock, whose
dispatch stream itself is hardware-proven clean (6/6 E2E, same boot era).
Run the port through the routine matrix+E2E battery in the next scheduled
jwm1 window.

## Box state

jwm1 healthy, no wedge, no reboot used (boot `e52e0f1c` throughout), ane
`6fa243a` loaded refcnt 0, lock free. Evidence home `/var/tmp/ane-minus5`
(shim + logs + runners + reports). Box handed to AneTmRecoveryBisect; their
T8103 window should re-test TQ-clear/0x3000 suspects against a workload
that no longer ships corrupt inputs. `63c1d3cf` not merged, not touched.

The landed rebase of 7da42928 onto origin/main (296c4352, auto-merged onto coopmat+pointwise) differs from the validated 7da42928 tree and was reverted: CHAIN_FUSION=1 silently corrupts encoder_hidden (dadd090d, 0/104 emissions); CHAIN_FUSION=0 is clean 104/104.
