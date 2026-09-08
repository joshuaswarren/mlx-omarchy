# Paired ANE vs GPU measurement on jwm1-linux (2026-09-08) -- ANE candidate stopped

Paired, same host, same window discipline (one flock on /tmp/m1-gpu.lock per
process tree, quiet CPU, never unlink). GPU numbers from the integrated
baseline release wheel mlx_omarchy-0.32.2.dev202609081618+c254867
(sha256 255c2f93...7e02); ANE numbers from the lifecycle6 ABI1 stack
(module f2a3e5e+lifecycle6, sha256
83819510cae37e5c748ff7586bf009d038866e5bfb657372646533a07b0ce0f0) via the
proven benchmark-packages.py harness. Raw JSON in this directory:
gpu-bench-c254-release.json, b1-gemv-k896-n4864.json,
b2-gemv-k4864-n896.json (+ device y.fp16 outputs), plus
ane-locked-run.log (the receipted ANE run) and
ane-unlocked-run-REJECTED.log (a window-contract violation, excluded as
evidence, disclosed below).

## Head-to-head (warm, 3 warmups + 30 samples, median, transfer->readback)

| projection | ANE warm e2e | GPU fp16 | GPU 4-bit qmm g64 (the real model op) | ANE vs GPU qmm4 |
|---|---:|---:|---:|---:|
| gate/up [1,896] x W[4864,896]^T | 200.95 ms | 1.021 ms | 0.439 ms | ~458x slower |
| down [1,4864] x W[896,4864]^T | 80.98 ms | 1.061 ms | 0.387 ms | ~209x slower |

Every iteration of these two packages on these tested inputs matched exact
fp16 outputs. No broader numerical-correctness claim is made.

Per-token arithmetic from the measured medians, if all 72 decode
projections (24 layers x gate+up+down) ran at these rates:
24 x (2 x 200.95 + 80.98) = ~11.6 s/token, versus the baseline's measured
4-bit whole-model decode of 56 tok/s = 17.9 ms/token. This is arithmetic
on this stack's measured numbers, not a claim about ANE hardware limits.

## Disclosure: window-contract violation (excluded run)

An intermediate ANE run at ~2026-09-08T17:23:35Z-17:24:02.839851Z UTC
executed without acquiring /tmp/m1-gpu.lock. It is a violation of the
window contract. Its outputs (512.87 / 100.62 ms) are excluded as
evidence and are not cited anywhere; the raw log is committed as
ane-unlocked-run-REJECTED.log. The cause of its difference from the
locked run is unknown -- no thermal, power, or occupancy instrumentation
was attached, and no cause is claimed. Exact timeline of all ANE/GPU
activity in this session (UTC):
- 17:22:08.525053Z paired window start, lock acquired (GPU bench + failed
  ANE preflight)
- 17:22:10.423963Z paired window end (ANE half aborted on a wrong
  device-node check; module left loaded)
- 17:23:15.223145Z manual rmmod, module VERIFIED_ABSENT
- ~17:23:35Z (approx) unlocked ANE run started (VIOLATION)
- 17:24:02.839851Z unlocked run unload VERIFIED_ABSENT
- ~17:25:07Z (approx, 16.26 s window) locked ANE run started
- 17:25:23.234637Z locked run unload VERIFIED_ABSENT
- 17:25:23.237210Z locked window end
Overlap confirmation requested from Main, ParityBaseline, and
DecodeParity for the 17:22:08-17:25:24Z span.

## Measured rejection

For the tested decode-projection candidates (gate/up, down; batch 1,
model shapes, warm, weights resident on device), the ANE path on this
stack measured 200.95 ms and 80.98 ms per projection against the GPU's
0.439 ms and 0.387 ms for the real 4-bit model op. That alone is
sufficient to stop this candidate: no MLX connected-graph integration is
warranted for these regions. Other model regions (attention, norms,
softmax, lm_head, prefill tiles) were not benchmarked on ANE in this
session and no claim is made about them. Two structural facts observed
while preparing candidates, for the record: the H13 backend lowered each
of these matmuls into 96/146 per-op program slices, and M in
{2,8,16,32,64} GEMM at K=896 N=4864 was refused by the compiler
("H13 intermediate physical writes must not overlap").

## Higher-value direction (recommendation, not a measured result)

ParityBaseline's diagnostic profile (intrusive instrumentation, diag
wheel 5e201380) recorded GPU busy 19.8/22.6/9.1%, intra-submission gap
p50 42.0/37.5/34.9 us, and dispatches 7045/76075/19915, with
ElementwiseF16 31-35% and QmmVecQ4WordSubgroupF16 21-23% of kernel busy
share. These are instrumented values; uninstrumented idle share is not
established by this session (DecodeParity is validating). If that
validation holds, reducing GPU dispatch count and submission gaps --
e.g. fusing elementwise chains (RoPE/RMSNorm/SiLU-mul/residual adds) in
mlx-omarchy's graph layer -- is the recommended next direction to
evaluate, as it would benefit both the 4-bit and bf16 models.

## Provenance chain

- GPU: release wheel mlx_omarchy-0.32.2.dev202609081618+c254867
  (255c2f93...7e02), script /tmp/gpu_bench.py (qmm on default device, no
  forced CPU), window 17:22:08-17:22:10Z UTC under flock.
- ANE (receipted): benchmark-packages.py (lifecycle6 harness, requires
  /sys/module/ane/version == f2a3e5e+lifecycle6), window ending
  17:25:23.237210Z UTC under flock; sudo insmod pre-load hash recorded;
  sudo rmmod + VERIFIED_ABSENT captured at 2026-09-08T17:25:23.234637Z
  (CLEANUP_OK, raw log committed); /dev/accel node present while loaded,
  absent after unload.
- GPU-side earlier smoke on the stale Sept-3 18f59e3 wheel is REJECTED
  and recorded in receipts/2026-09-08-stale-wheel-smoke-rejected.md
  (window 16:33:02-11Z, overlapped only ParityBaseline's pip install;
  no timed baseline activity existed in that window).
