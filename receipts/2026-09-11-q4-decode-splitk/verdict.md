# Q4 decode split-K: the last unmeasured high-parallelism structure loses at every split count — receipt-only negative

- schema: mlx-omarchy/q4-decode-splitk/1 (2026-09-11)
- agent: Q4DecodeSplitK; branch wave/Q4DecodeSplitK (off origin/main c9881be4), receipt + env-gated measurement variant only. Never merged; main and the integration worktree untouched.
- host: jwm1-linux (<m1-host>), Apple M1 (G13G B1), linux-asahi 7.1.6.asahi1-1, installed fork driver mesa-honeykrisp-omarchy 26.3.0.devel.hk6f6afc8-1, verified by `pacman -Q` inside every window.
- timing: host wall clock around whole submits only. No device-timestamp number is used (known ~2.07x undercount on this driver).
- windows: two single top-level flocks on /tmp/m1-gpu.lock — 13:15:23Z (aborted after 3 s, bench loader bug from this receipt's own edit, fixed, rerun) and 13:19:06–13:20:19Z (complete). loadavg 0.00 before each; each announced on hub before the flock.

## What was measured and why

The memory-roof receipt left the decode GEMV at 41 GB/s against a 60 GB/s
layout roof and named "insufficient memory-level parallelism inside the
kernel" as the remaining deficit. Every order-preserving lever was measured
and lost. The one structure left unmeasured was split-K: give each output's
k range to S workgroups-per-row-block and reduce the partials in a second
dispatch. It deliberately changes the accumulation order, so it was a
measurement and a policy input, never a landing candidate.

The variant ships as `-DQMM_VEC_Q4_SPLITK={2,4,8}` builds of the fused
multi-weight qmm_vec shader plus a `qmm_vec_splitk_reduce` pass, selected
only by `MLX_OMARCHY_QMM_VEC_Q4_SPLITK=2/4/8` at dispatch (f16, subgroup,
multi-weight path — the path the four real decode dispatches take).
Default binaries are byte-identical: all 24 default variants compile to the
same SPIR-V from the variant source. Each split keeps the production
one-word-per-lane map over its contiguous word range, applies the same
per-group scale/bias fma, reduces its 32 lanes exactly like production, and
writes one raw f32 partial per (split, output column); the reduce pass sums
the S partials in split order, rounds once, and applies the Add epilogue in
the production round-then-add order.

## 1. In-model chain result (the decisive datum)

Gap-mode four-dispatch layer chain (the instrument behind the 41 GB/s
figure), RAW-chained like the model, pass 2 of two independent legs, wall
medians. Split arms record split+reduce per shape — eight dispatches per
layer — and their byte count credits them with the full partial round trip.

| arm | layer ns | bytes | GB/s | time vs base |
|---|---|---|---|---|
| base widedep | 205,877 | 8,427,008 | **40.93** | 1.000 |
| splitk2 | 314,852 | 8,629,760 | 27.41 | **1.529** |
| splitk4 | 365,249 | 8,832,512 | 24.18 | **1.774** |
| splitk8 | 597,255 | 9,238,016 | 15.47 | **2.901** |

Split-K does not raise memory-level parallelism into the roof; it lowers
achieved throughput with every added split, and that is before counting the
reduce dispatch as anything more than its bytes. Instrument health: base
reproduces the known 41 GB/s, and the wg128 candidate arm measured
-4.4%/-5.2% here, independently reproducing the Q4RowsPerLane receipt's
-4.25%/-4.42% same-day finding.

## 2. Isolated per-shape walls (two legs, wall ratio vs base)

| shape (grid) | S=2 | S=4 | S=8 |
|---|---|---|---|
| qkv (144) | 0.99–1.08 | 1.02–1.12 | 1.19–1.40 |
| o (112) | 0.98–0.99 | 0.99–1.03 | 1.14–1.37 |
| gate_up (1216) | 1.11–1.16 | 1.40–1.46 | 2.15–2.47 |
| down (112) | 1.01–1.05 | 1.07–1.10 | 1.17–1.24 |

No shape at any split count shows a gain outside noise. The already-widest
shape (gate_up, grid 1216) degrades worst and monotonically: extra
workgroups buy eviction, not parallelism. The full wall tables are in
verdict.json `tables.iso_per_shape_wall_ns`.

## 3. The reduction pass is not cheap enough — and cannot be

Reduce recorded alone (split dispatch removed): 61.1/64.1/65.0/37.4 µs wall
per dispatch for qkv/o/gate_up/down (medians across both legs; the ~21–23 µs
submit bracket is inside those numbers). Split-K adds one fixed-cost
dispatch per GEMV per layer at every split count, on a chain whose whole
budget is ~830 µs/token of GEMV. Even a zero-time reduce would leave the
split kernels themselves slower (table 2).

## 4. Engagement and accuracy

- Engagement is proven by output change plus scaling: down (K=4864) moves
  exactly 1 of 896 f16 outputs by 1 ulp at every S; split walls scale with S
  (gate_up 2.15–2.47x at S=8). A non-engaged variant would compare and time
  identical to base.
- Measured precision change vs the current production kernel: max 1 f16 ulp
  on 1 of 12,672 layer-chain outputs, 0 elsewhere, at every split count.
  The reordered f32 sums round to the same f16 bits almost everywhere — the
  order change is real but lands under the f16 rounding on these shapes.
- The f64-RNE model-level reference comparison from the original assignment
  was NOT run (it needs the wheel; cut with the legs by the receipt-only
  redirect). The current kernel's own f64 distance is documented in
  receipts/2026-09-11-q4-decode-pair-map.

## Why no model-level legs

Parent redirect on the isolated result: the best split count is +52.9% per
layer, which fails the >=3% model-level gain bar by orders of magnitude, and
any split count moves the pinned canonical digests by construction (the
order change is the point of the experiment), so the digest axis could not
compensate. A wheel build and a 3x{fork,stock}x{base,splitk}x3-legs matrix
would measure nothing not already decided. The harness is committed and
ready (run_splitk_legs.py, run_accuracy_arm.py, compare_accuracy.py,
window_legs.sh) if the isolated picture ever changes.

## Decision

Rejected. The pin stays with the single-pass one-word-per-lane kernel and
the 60 GB/s roof stays unreachable from this direction: the deficit is not
buyable with workgroup-level k splitting on this device — the chain was
already latency/occupancy-bound, and more concurrent row-blocks mostly buy
extra x-row reads, partial traffic, and a serialized reduce dispatch. The
env-gated variant and shaders stay on this branch as instrumentation only.
Nothing merged; no test added (nothing landed).

## Provenance

- branch wave/Q4DecodeSplitK: 73bb1e4 (env-gated variant; ran the window),
  plus this receipt commit (bench loader fix: restore
  LOAD_DEV(CreateDescriptorPool) clobbered by the ResetDescriptorPool
  addition; frozen shader copies refreshed; reduce shader copy added to the
  bench).
- shader copies: qmm_vec_base.comp == qmm_vec_splitk.comp ==
  2b9b521a6c79b4a6563f2f87c4245948fcae41ac672b3980c1959e8ee7feaaf1
  (current production shader), qmm_vec_splitk_reduce.comp ==
  0760653634b8522b7f2a2878de345dab6d6e3777d791260f521c8a4a542b441c.
- logs: m1-logs/iso-{a,b}-20260911T131906Z.ndjson,
  m1-logs/gap-{a,b}-20260911T131906Z.ndjson.
- wheels: none built (redirect).
