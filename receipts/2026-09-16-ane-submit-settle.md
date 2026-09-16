# ANE submit settle: doorbell-pacing guard applied and measured — does NOT
# fix the fused runner tm -5; hypothesis disproven, nothing landed (2026-09-16)

Verdict: **NO-LAND.** The worker-side settle knob works exactly as specified
(banner-proven applied in the worker child, quantitatively proven to pace
every doorbell, stock pin `38c73261…` unchanged under it), and it does not
prevent the wedge: fusion-on deterministically fails at the same
`island-attn-a-kt` L01-A `ane_exec failed for program 1` with a 2 ms guard
*and* with a 100 ms guard immediately before the failing doorbell. The
"fast-follow submit lands in the previous task's hardware retirement tail"
hypothesis is disproven at any plausible retirement-tail scale (µs..ms).
Neither acceptance condition (fusion 104/104 ×3) can be met, so per the
mandate nothing is landed: no worker commit, no fold runner.

## Legs (jwm1, every submit under `flock -w 900 /tmp/m1-gpu.lock`, deadline 20 s)

| leg | settle_us | stream | result |
| --- | ---: | --- | --- |
| fus-s2000 | 2000 | fusionon | **tm -5**, `island-attn-a-kt` L01-A program 1, round elapsed 24 ms, ANE-DIRTY |
| stock-plumb | 100000 | stock (wt-main runner) | clean, encoder_hidden `38c73261…` (pin holds), rc=0 |
| fus-s100000 | 100000 | fusionon | **tm -5**, same L01-A program 1, round elapsed 228 ms (the failing doorbell had its 100 ms guard), ANE-DIRTY |

Knob application proof, two independent kinds:

1. Worker child stderr banner (added to `resident_child_loop`):
   `[omarchy-ane] resident child settle_us=100000` in both 100000 legs.
2. Stock encoder wall with settle=100000: **18129 ms vs 7994 ms baseline**
   (`/var/tmp/ane-settle/out-stock-plumb/run-report.json` vs the chain-kernel
   `mx-main` report) = +10.1 s ≈ 96 doorbells × 100 ms, first doorbell free.
   exec_ns 12125 ms vs ~4943 ms baseline. Pin still `38c73261f29230276…` —
   the guard is byte-inert on the stock stream.

Both fusion wedges produced the identical kernel signature:
`tm completion failed: -5, finish lines=0; preserving resources until reboot`
(fus-s2000 at 1723 s uptime on boot `44b8807e`; fus-s100000 at 483 s on boot
`97c2dbb3`).

## Box state (handed back honestly)

- **jwm1 is wedged-soft right now** (module says "preserving resources until
  reboot"), module `6fa243a` / `3D83D6E0B587342305BE5C0` loaded, refcnt 0,
  boot `97c2dbb3`. The next ANE work on jwm1 needs a reboot first; this lane's
  one-reboot budget is spent (spent clearing wedge #1, announced by ledger).
- `/var/tmp/mlx-main-strict` restored pristine at `3aa4f348` (git status
  empty; settle source change reverted from both `overlay/` and `.work/mlx`),
  stock worker rebuilt bit-exact `f171a61ecfc89942b…` (pre-experiment sha).

## What the settle change was (preserved, not landed)

In `overlay/mlx/backend/omarchy/ane/worker.cpp` `execute_plan`, before every
`device.exec` after the process's first: `std::this_thread::sleep_for`
`MLX_OMARCHY_ANE_SETTLE_US` (default 2000, `0` disables), gated on a
process-wide `previous_doorbell_issued` flag (device.exec blocks on
completion, so the previous completion event is always observed first).
Preserved for the next lane:

- `/var/tmp/ane-settle/worker-settle.patch` — the exact 82-line diff vs
  `3aa4f348` (includes the settle banner line).
- `/var/tmp/ane-settle/worker-settle-f0b017d0` — the built worker with the
  knob (sha `f0b017d07a7c6651…`).
- `/var/tmp/ane-settle/leg.sh` — parameterized leg runner (enc/e2e modes).
- Logs: `log-fus-s2000.txt`, `log-stock-plumb.txt`, `log-fus-s100000.txt`,
  `scratch-*/resident-worker.stderr`, out dirs with run reports.

## Consequence for the -5 anatomy

Pacing the doorbell is ruled out. The fused GPU dispatch stream triggers the
L01-A `-EIO` by some channel other than submit timing — candidates the next
window should consider: what the fused kernels change about GPU memory
traffic/buffer addressing visible to the ANE BOs at that submit, or TM FIFO
state carried across the L00-C → L01-A boundary that is sensitive to the GPU
stream *shape* rather than its *speed*. Note the failing round itself
completed in 24 ms (fused) — well inside stock per-round means — so it is not
a deadline effect.
