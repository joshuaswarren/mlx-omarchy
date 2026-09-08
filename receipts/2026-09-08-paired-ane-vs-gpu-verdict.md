# Paired ANE vs GPU measurement on jwm1-linux (2026-09-08) -- ANE candidate stopped

Paired, same host, same window discipline (one flock on /tmp/m1-gpu.lock per
process tree, quiet CPU, never unlink). GPU numbers from the integrated
baseline release wheel mlx_omarchy-0.32.2.dev202609081618+c254867
(sha256 255c2f93...7e02); ANE numbers from the lifecycle6 ABI1 stack
(module f2a3e5e+lifecycle6, sha256
83819510cae37e5c748ff7586bf009d038866e5bfb657372646533a07b0ce0f0) via the
proven benchmark-packages.py harness. Raw JSON in this directory:
gpu-bench-c254-release.json, b1-gemv-k896-n4864.json,
b2-gemv-k4864-n896.json (+ device y.fp16 outputs).

## Head-to-head (warm, 3 warmups + 30 samples, median, transfer->readback)

| projection | ANE warm e2e | GPU fp16 | GPU 4-bit qmm g64 (the real model op) | ANE vs GPU qmm4 |
|---|---:|---:|---:|---:|
| gate/up [1,896] x W[4864,896]^T | 200.95 ms | 1.021 ms | 0.439 ms | ~458x slower |
| down [1,4864] x W[896,4864]^T | 80.98 ms | 1.061 ms | 0.387 ms | ~209x slower |

Every ANE iteration matched exact fp16 outputs -- the stack is numerically
correct, it is only slow. An earlier unlocked ANE run the same hour
measured 512.87 / 100.62 ms for the same packages (run-to-run power/thermal
variance); the receipted locked-run numbers above are the evidence. Both
runs agree on the verdict by two orders of magnitude.

## Why the ANE loses: program-slice dispatch, not compute

The H13 backend lowers one large matmul into 96 (gate/up) / 146 (down)
per-op program slices; the ABI1 driver serializes submissions. At the
measured fixed floor of ~0.15-0.4 ms per program dispatch (lifecycle6
MLP slices, add-relu fused single program 0.160 ms), a decode GEMV cannot
beat the GPU's single-kernel 0.39-0.44 ms. Even hypothetical Apple-style
whole-chain fusion (one program, ~96 tasks at the measured ~20-40 us/task
from the multi-task softmax program) lands at 2-4 ms per projection --
still 5-10x slower than the GPU qmm4 kernel, and that fusion does not
exist in this fork's backend today (eltwise chain fusion only; M>1 GEMM
is refused outright with "H13 intermediate physical writes must not
overlap", so prefill is compiler-blocked at model shapes).

Per-token ceiling if ANE ran all 72 decode projections (24 layers x
gate+up+down): 24 x (2 x 200.95 + 80.98) = ~11.6 s/token, versus the
baseline's measured 4-bit whole-model decode of 56 tok/s = 17.9 ms/token
-- ~650x slower on projections alone, before attention, norms, or lm_head.

## Verdict

The ANE acceleration candidate for Qwen2.5-0.5B on this stack is STOPPED
with paired measured evidence. No MLX connected-graph integration is
warranted: there is no hot static region where this ANE path wins warm
end-to-end, and the small-op shapes where ANE did win (add/softmax/matvec
at <=512 elements, ~1.2-4x) are noise-level contributors to per-token
time (transfer-dominated, exactly the class the assignment excludes).

## Higher-value cross-layer route (from the baseline's own profile)

ParityBaseline's c2548675 profile shows the GPU dispatch-bound, not
compute-bound: GPU busy 19.8/22.6/9.1%, intra-submission gap p50
42.0/37.5/34.9 us, dispatches 7045/76075/19915 for the three workloads.
Kernel busy share: ElementwiseF16 31-35%, QmmVecQ4WordSubgroupF16 21-23%,
CopyGeneralF16 10.7-11.7%, FastRopeF16 9.7-11.3%, FastRmsNormF16 7.0-10.7%.
The lever that moves end-to-end throughput is cutting GPU dispatch count
and submission gaps (fusing elementwise chains -- RoPE/RMSNorm/SiLU-mul/
residual adds -- into fewer kernels and fewer submissions), not ANE
offload. That work lives in mlx-omarchy's graph/scheduler layer and
directly attacks the 77-91% idle GPU time; it benefits both the 4-bit and
bf16 models.

## Provenance chain

- GPU: release wheel mlx_omarchy-0.32.2.dev202609081618+c254867
  (255c2f93...7e02), script /tmp/gpu_bench.py (qmm on default device, no
  forced CPU), window 17:22:08-17:22:10Z UTC under flock.
- ANE: benchmark-packages.py (lifecycle6 harness, requires
  /sys/module/ane/version == f2a3e5e+lifecycle6), window 17:25:07-17:25:23Z
  UTC under flock; sudo insmod pre-load hash recorded; sudo rmmod +
  VERIFIED_ABSENT captured at 2026-09-08T17:25:23.234637Z (CLEANUP_OK);
  /dev/accel node confirmed present while loaded, absent after unload.
- GPU-side earlier smoke on the stale Sept-3 18f59e3 wheel is REJECTED
  and recorded in receipts/2026-09-08-stale-wheel-smoke-rejected.md
  (window 16:33:02-11Z, overlapped only ParityBaseline's pip install;
  no timed baseline activity existed in that window).
