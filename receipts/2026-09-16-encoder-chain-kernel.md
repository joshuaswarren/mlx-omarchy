# Encoder chain kernel: fused bias(+silu) epilogue is byte-identical and
# GPU-faster in isolation but deterministically trips the ANE tm -5 wedge —
# nothing landed (2026-09-16)

Verdict: **NO-LAND.** The fused chain kernels are bit-exact on all four
program shapes and remove 2 dispatches + one full f16 round-trip per biased
linear, but with the fusion active the encoder's first ANE island-A submit
fails with `tm completion failed: -5` and wedges the engine — 3/3 fusion-on
attempts across 3 boots, 5/5 clean for the stock stream. Landing is blocked
on the tm race (AneTmRecoveryT8103's lane). The kill-switched runner
(`MLX_OMARCHY_CHAIN_FUSION=0`) is proven inert and preserved as the landing
candidate once the tm bug is fixed.

## Evidence home (jwm1)

`/var/tmp/enc-chainkernel` — runner
`runner/vulkan_encoder.py` sha256 `77b4ee5f74ac85249745405f860d21f0e04aff06af99d6a3ce5ae40eb510f27b`
(origin/main `69fd5397` runner sha256 `eedb7c4802ca678183deb3a55c1347f...`
+ gated fusion hunks only), `chainbench.py` (identity+microbench),
`matrix.sh` (discriminator), `run4.log`/`run5.log` (microbench), and the
`mx-*-out`/`mx-*-scratch` matrix legs. Island bundle programs unchanged:
`island-attn-a-kt/program-0.anec` `cf0ecac2...`, `program-1.anec`
`b801f621...`, `island-pv/program-0.anec` `3ae36f21...`.

## Attack (a) — the chain kernel itself: no win, not landed

The chain kernel after `69fd5397` reads fp16 partials `[K/16, 375, N]`
(the f32 partial round-trip is already gone). Pairing adjacent columns and
reading them as one uint32 word (`partials.view(mx.uint32)`,
`unpackHalf2x16`), with per-output ascending fp16 accumulation reproduced
by exact-f32-add + single `packHalf2x16` rounding (the
`fused_chain.comp` integer-packing lesson), is bit-identical but **not
faster**: (375,1024,1024) v0=1.10 ms vs v2=1.08 ms; (375,1024,4096)
3.65 vs 3.73; (375,4096,1024) 3.65 vs 3.68; (375,512,640) 0.53 vs 0.51.
The kernel is bandwidth-bound and already streams the minimum bytes.

## Attack (b) — fold bias (+ silu) into the chain's final store

Two kernels added (`_leftover_chain_bias_kernel`,
`_leftover_chain_bias_silu_kernel`): the chain half is byte-identical to
the landed chain; the epilogue reproduces the removed dispatches' roundings
exactly — bias sum rounds once to fp16 (equal to stored-f16 chain output
widened + f32 bias + round), and the silu fold rounds the bias sum to fp16
*first* (the stored f16 the separate silu dispatch read), then silu in f32,
one final rounding. Unit identity on the four program shapes
(375,512,640)/(375,1024,1024)/(375,4096,1024)/(375,1024,4096), fp16 bits
compared: **ALL-EXACT, 0 mismatches** (`diag3.py`, seed-0 data).

Microbench (release wheel site, 30-iter medians, jwm1):

| shape (M,K,N) | current chain+bias | fused | current +bias+silu | fused+silu |
| --- | ---: | ---: | ---: | ---: |
| 375,1024,1024 | 1.72 ms | 1.26 | 1.90 | 1.26 |
| 375,1024,4096 | 6.07 | 3.79 | 6.35 | 3.79 |
| 375,4096,1024 | 4.37 | 3.69 | 4.45 | 3.69 |
| 375,512,640 | 1.29 | 0.60 | 1.72 | 0.68 |

## Attack (c) — attention MatmulF32 → fp16 coopmat: not reached

Blocked by the same investigation; the remaining MatmulF32 dispatches are
non-island `matmul` statements. Not attempted after (b)'s finding.

## Why NO-LAND: the discriminator matrix (boot #4, after smoke PASS)

| leg | runner | dispatches (prim/sub) | encoder_hidden | result |
| --- | --- | --- | --- | --- |
| main | `69fd5397` as-is | 6762 / 244 | pin `38c73261...` | PASS |
| fusionoff | mine, `MLX_OMARCHY_CHAIN_FUSION=0` | **6762 / 244 (identical)** | pin `38c73261...` | PASS |
| fusionon | mine, fusion active | — | none | **ANE tm -5, wedged** |

`tm completion failed: -5, finish lines=0; preserving resources until
reboot` at 70.57 s, first `island-attn-a-kt` L01-A resident submit
(`ane_exec failed for program 1`). Identical pattern on boots #1 (pre-reboot
matrix, 04:22-04:23) and #2 (129 s). The stock stream never missed:
3 baseline legs + 5 main-runner legs clean across boots. The fusion changes
only the GPU dispatch stream feeding the first island submit (one fused
dispatch replaces chain + f32-add + cast per biased linear), so the tm -5
is sensitive to GPU stream shape/timing around the first resident submit —
evidence for AneTmRecoveryT8103.

## Boot ledger (all reboots authorized by Main)

- reboot #1: cleared boot-0 wedge (module hand-loaded `b52064c`, -5 twice,
  `tm not idle after reset: 0x0` → preserve; loads `6fa243a` after boot).
- reboot #2: cleared fusionon wedge #2 (129 s).
- reboot #3: cleared fusionon wedge #3 (matrix leg 3, 70.57 s). Box handed
  to AneTmRecoveryT8103 after the matrix. `/tmp/m1-gpu.lock` inode changed
  at every boot (reboot, not theft).

## Landing path

1. AneTmRecoveryT8103 fixes/proves the tm -5 (their repro window).
2. Re-run `matrix.sh`: fusionon must pass 3 legs with pin `38c73261...`
   and no `-5` in dmesg.
3. Then E2E 6×: 104/104, transcript `db501a8c...`, encoder wall
   improvement expected ≈0.4-0.9 s from the fold (145+48 biased linears,
   microbench deltas).
4. Land with the `MLX_OMARCHY_CHAIN_FUSION` kill-switch default-on.
