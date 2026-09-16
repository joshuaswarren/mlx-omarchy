# Encoder fold reland on the coopmat lineage: bias index fix + silu-round
# fix, bit-exact and pin-identical on both laptops (2026-09-16)

Verdict: **LAND.** The bias(+silu) leftover-chain fold is re-ported by hand
onto current main's fp16-partials coopmat lineage, wired into the coopmat
branch that actually runs, with two real defects found and fixed by the
gates: the 296c4352 apply-site miss (silent corruption) and a compiler-
elided fp16 rounding in the fused silu kernel (14% element mismatch). All
gates pass on both hosts: lincheck ALL-EXACT, `encoder_hidden` = pin
`38c73261f29230276ed76f1fc017b76b024156d79218bd5f1347fdc7e7d43ec7`, E2E
6/6 no flags = 104/104, transcript `db501a8c…`. encoder_ane median r2–r6:
jwm1 7215.3 → **7036.2 ms**, jw16 5083.0 → **4872.3 ms**. Kill-switch
`MLX_OMARCHY_CHAIN_FUSION=0` reproduces the stock bytes (pin exact on
jwm1). Landed as the fold commit `e2f5c0c7` (this receipt at tip
`516890e3`), fast-forwarded onto origin/main from 22c6b139.

## Why 296c4352 corrupted (and this port does not)

`296c4352` (reverted by `616b5b89`) applied the fp32-base fold text
verbatim; the diff of the two fold commits is byte-identical apart from
hunk offsets. The apply-site hunk's context —
`out = mx.reshape(out, tuple(x.shape[:-1]) + (weight.shape[0],))` —
matches only the fp32-partials **fallback** branch, because the coopmat
branch reshapes with `(n_out,)`. So on the coopmat lineage the fused
kernels were wired into a branch that never runs, while `_index_fusions`
(whose hunks applied cleanly) still recorded the feed-forward silu in
`silu_done`; `execute()` skipped the silu statement and the runner aliased
its name to the unfused biased linear output. The FF silu never executed:
deterministic `encoder_hidden` = `dadd090d…` with a healthy island pass —
the silent corruption the both-hosts receipt caught.

This port:
- wires the fused chain+bias(+silu) dispatch into the **coopmat branch**
  itself (fp16 partials), replacing chain + astype-bias (+ silu) with one
  dispatch, and leaves the validated text on the fp32 fallback branch;
- keeps the pair-based bias index (`bias[pair * 2u]`, in-row pair — the
  minus5 root-cause fix; never the flattened word);
- keeps the `linear_silu`/`silu_done` coupling with the apply-time
  `silu_done.discard` escape when the fused path is not taken;
- keeps `MLX_OMARCHY_CHAIN_FUSION` (default on; `0`/`false`/`no` = stock
  dispatch stream).

The fused kernels read `half(partials[…])` per block — identity on the
coopmat path's fp16 partials, the same single RNE the stock chain applies
on the fallback's fp32 partials — and accumulate in fp16 ascending,
byte-identical to `_leftover_chain_kernel` on both dtypes (lincheck
proves it per arm below).

## Second defect the gates caught: elided half() round in the silu tail

First lincheck on jwm1: bias arm 0 mismatches on all four shapes, but
bias+silu mismatched 12–19% of elements. Isolation (probes archived in
this receipt dir):

- tail-only (1-block partials, zero bias) vs `_silu_kernel`: **0**
  mismatches — the silu math itself is exact;
- the mismatch count tracks the "silu applied to the **unrounded** f32
  bias sum" model — this compiler elides the in-register `half` round of
  `half w0 = half(float(a0) + float(bias[...]))` when `w0` only feeds
  `float()` reads (the bias kernel's identical store-side round is
  observable, hence exact);
- `volatile` does not force the round; `as_type` is not in the MSL subset
  the omarchy backend accepts.

Fix: the fused kernel stores the rounded bias sum to `reduced[col]/
[col+1u]` and reads it back before the silu math — a memory round-trip
pins the rounding. Each thread owns its two columns, so the write-back
has no cross-thread hazard. Lincheck after the fix: **ALL-EXACT**, three
consecutive runs, both arms, on the four program shapes
(375,512,640) (375,1024,1024) (375,1024,4096) (375,4096,1024), u16-view
comparison vs the current stock chain (chain kernel + astype-f32 bias +
silu kernel) on coopmat fp16 partials.

## Gates (hardware, both laptops, before landing)

Environment identity per host identical to the both-hosts baseline
receipt: worker `f171a61e…`, libane-strict `56b46234…`, bundles tree
`18e3b7ea…`, harness `fused_e2e.py` `0e38e7b1…`, model `b650695c…`,
non-diag wheel site at f43ab71c, `ANE_ISLAND_MODE=resident-batch`, lock
`/tmp/m1-gpu.lock` `flock -w 900` never stolen never unlinked, no flags.

| Gate | jwm1 (T8103) | jw16 (T6001) |
| --- | --- | --- |
| lincheck fold (4 shapes × 2 arms × 3 runs) | ALL-EXACT | ALL-EXACT |
| standalone encoder `encoder_hidden` | `38c73261…ec7` | `38c73261…ec7` |
| standalone ANE | 1 submit / 72 rounds / 1 worker start / 0 timeouts, exec 2637 ms | 1 submit / 72 rounds / 1 worker start / 0 timeouts, exec 2653 ms |
| kill-switch `CHAIN_FUSION=0` pin | `38c73261…ec7` | n/a (switch is dtype/host-independent) |
| E2E 6 runs no flags | 6/6 `match`, 104/104, bounds pass, gpu-loop, fallback null, cpu_tensor_events 0, 1 submit / 0 timeouts | 6/6 `match`, 104/104, bounds pass, gpu-loop, fallback null, cpu_tensor_events 0, 1 submit / 0 timeouts |
| `encoder_hidden` per run | `38c73261…ec7` ×6 | `38c73261…ec7` ×6 |
| transcript | `db501a8c080380ea…0790` | `db501a8c080380ea…0790` |

## Encoder ms before/after (median r2–r6, same-day same-boot-era baseline)

| Stage | jwm1 before | jwm1 after | jw16 before | jw16 after |
| --- | ---: | ---: | ---: | ---: |
| encoder_ane | 7215.3 | **7036.2** (−179.1) | 5083.0 | **4872.3** (−210.7) |
| total_pipeline | 8541.8 | 8436.0 | 6418.5 | 6213.9 |
| encoder vk_compute_dispatches | 3709 | **2885** (−824) | 3709 | **2885** (−824) |
| encoder gpu_primitive_dispatches | 6762 | 5696 | 6762 | 5696 |

Before numbers: `receipts/2026-09-16-parakeet-e2e-both-hosts` battery
re-collected from jwm1 `/var/tmp/E2EREV-battery` (same dirs, same boot
era). The −179 ms is smaller than the fp32-base fold's win because the
coopmat + fused-pointwise lineage (69fd5397, 83de2255) already cheapened
the removed dispatches; the fold still removes 824 compute dispatches
(24 FF bias+silu dispatch groups) worth 179.1 ms on jwm1.

## jw16 window

llm-inference.service (Main-cleared, not a router leg) stopped
09:10:38–05:00, restarted 09:11:36–0500 (active confirmed); window
**58 seconds** — well inside the 30-min cap. A first attempt of this
script placed the lincheck before the service stop and would have
dead-waited the flock: llama-server holds `/tmp/m1-gpu.lock` for its
whole lifetime (the unit wraps it in flock), so the gate script was
killed before the window opened and re-run with the lincheck inside the
window. Zero contention on every recorded run: each held the lock via
`flock -w 900`, and the lock freed only after the stop. Zero contention: the service's llama-server holds
`/tmp/m1-gpu.lock` while running; the lock freed only after the stop, and
every recorded run held the lock via `flock -w 900`.

## Artifacts

`receipts/2026-09-16-encoder-fold-reland/`: `lincheck_fold.py` (bit
harness), `probe_silu.py`, `probe2.py`, `probe3.py`, `probe4.py`
(isolation + fix probes), `gate-enc.sh`, `run-battery-jwm1.sh`,
`gate-jw16.sh`, `collect-fold.py`. Host side: jwm1 `/var/tmp/enc-fold-reland/`
(runner `57fb19c5…`, battery out-r1..6), jw16 `/var/tmp/enc-fold-reland/`
(same runner sha, battery out-r1..6). Runner staged on both hosts has
sha256 `57fb19c5996ca00fc198d50b5da875faa3c1d2fc6958894f32a994e2f91ca379`
= worktree `overlay/tools/coreml/vulkan_encoder.py` at `<land-sha>`.

## Not claimed

- No claim on other fixtures, other SoCs, macOS, or cold-SPIR-V numbers
  (r1 quoted separately in the collected logs).
- The fallback fp32-partials branch is guard code for non-coopmat shapes
  on this program; its fold site carries the validated 7da42928 text and
  is exercised only off the coopmat guard.
- `63c1d3cf` not merged, not touched.
