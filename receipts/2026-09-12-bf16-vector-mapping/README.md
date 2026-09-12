# BF16 vector mapping: numerical rejection, unresolved performance cause

The proposed per-lane load widening is not bit-preserving. It remains a
rejected software candidate; no changed kernel or driver ships. The retained
hardware measurements characterize the existing implementation, not a speedup.

This review withdraws the earlier claim that copy throughput proves no kernel
headroom remains. It also withdraws the 1.42 ms recoverable-time bound, the
8.4 us dispatch attribution, and the claim that dispatch structure exclusively
explains the native-performance gap. Those conclusions exceed the measurements.

## Candidate and numerical check

The shipped `matmul_vec.comp` BF16 path assigns four consecutive k-elements
per lane, advancing by 128 across a 32-lane subgroup. Each subgroup owns four
output columns. Its shuffle-down reduction uses offsets 16, 8, 4, 2, 1.

The proposed mapping assigns eight consecutive k-elements per lane and advances
by 256. It reduces load-instruction count but changes accumulation grouping.
The retained `model/counterexample_K16.npz` produces packed BF16 `0x0000` with
the shipped mapping and `0x4000` with the candidate mapping.

The parent independently checked that fixture with the system `libm fmaf`
operation substituted for the model's FMA. Both the original K=16 fixture and
its zero-padded K=896 version gave the same disagreement. Inputs were checked
for finiteness. This is software evidence against a bit-preserving claim, not
a modified-kernel GPU qualification or a statement about all load widening.

The prior random-data model check found 0 differing columns among 96 columns.
That does not override a counterexample. The existing digest policy remains
unchanged; no claim is made that this candidate converges to native outputs.

## Byte accounting

`model/byte_accounting.py` counts the projection weights once per generated
token for the pinned Qwen2.5-0.5B BF16 model, including all 24 layers:

| Class | Projections | Bytes/token |
|---|---|---:|
| Vec | wq, wo, gate, up, lm_head | 767,721,472 |
| Tile | wk, wv, down | 220,200,960 |
| Total | 169 dispatches in the recorded census | 987,922,432 |

The earlier attribution receipt's roughly 370 MB estimate omitted the
24-layer multiplier from part of the total. The integer accounting corrects
that error. It does not measure DRAM transactions, cache reuse, or elapsed time.

Dividing these bytes by the historical 18.31 ms GEMV ablation marginal gives
54.0 GB/s. An ablation marginal is a wall-time difference between interacting
workloads, not an isolated kernel duration. This quotient cannot bound the
in-model tile rate or identify where the remaining time is spent.

## Historical hardware window

The retained window ran on jwm1-linux, Apple M1 G13G B1, with
`mesa-honeykrisp-omarchy 26.3.0.devel.hk6f6afc8-1` and wheel
`mlx_omarchy-0.32.2.dev202609121038+a2e38c3`.
The full wheel hash is in `verdict.json`. Raw driver information remains in
`window/drivers.txt`.

One `/tmp/m1-gpu.lock` window lasted 53 seconds. The wrapper passed its quiet
gate, discarded five warmups, and measured whole-submit CLOCK_MONOTONIC
brackets. Medians use 24 rounds, except eight for the misaligned arm.

| Arm | Median ms | Derived throughput |
|---|---:|---|
| lm_head, n=151936 k=896, vec route | 4.730 | 57.57 GB/s read |
| Same shape, misaligned lhs offset, tile route | 22.357 | 12.18 GB/s read |
| Half-sized lm_head, n=75968 | 2.546 | 53.47 GB/s read |
| Contiguous BF16 copy | 10.452 | 52.10 GB/s aggregate read + write |

The aligned and misaligned arms distinguish the routes in isolation. Their
ratio is not an in-model gain. Read-only GEMV and aggregate copy throughput
are different workloads; neither establishes a hard bandwidth ceiling for the
other. The half-size result is consistent with bandwidth sensitivity but does
not exclude other costs. The A/A interleaved medians differed by 0.24%; this
single spread is not a confidence interval.

The earlier isolated 9.3 ms lm_head figure is superseded by the correctly
scoped 4.730 ms whole-submit measurement. The earlier span screen's 4.687 ms
is within 0.9% of that measurement. All three canonical short-leg control runs
retained `f26175202f3dabe9`, at 31.88, 31.88, and 31.95 tokens/s.

## What remains unresolved

The attribution experiment's all-ablated workload retains loop walks, output
stores, launch grids, and submission behavior. Its residual cannot be called
pure dispatch or launch time. The later barrier-deferral experiment reduced
barrier counts without a repeatable decode gain; it also did not isolate the
whole remaining gap. See `../2026-09-12-bf16-encoder-barrier-pair/`.

A useful next timing experiment must distinguish GPU loop/store work from
launch, dependency, and submission costs. The old residual divided by dispatch
count is not that experiment. Native performance parity remains unmet.

No new hardware performance run accompanies this correction. Raw measurements,
SPIR-V, the numerical model, and the fixture remain under `window/`, `spirv/`,
and `model/`. Only the unsupported analysis and derived bound output were
removed; `model/byte_accounting.out` now contains integer byte counts only.
