# Staging attribution: what the 179-instruction phase actually costs (2026-09-12)

Question, from receipts/2026-09-12-agx-qmm-codegen: the shipped
`shaders/qmm_coopmat.comp` step body issues 229 instructions per 16-k step
(16 matrix fmadd, zero issue waste, no spills), yet runs 1034-1047 GFLOP/s
against a 1440-1459 GFLOP/s zero-traffic ceiling and a 2148 GFLOP/s isolated
matrix rate. Two theories about the staging phase are already measured dead
(exec-mask drains: de-divergence was -13 to -26 percent; TILE_M64 and
STEP_K32 also dead). This receipt enumerates the remaining candidate
explanations as testable hypotheses, names each instrument, reports each
number, and ranks.

Machine: jwm1 (Apple M1 G13G B1, aarch64), Omarchy Linux, fork driver
`mesa-honeykrisp-omarchy 26.3.0.devel.hk6f6afc8-1`, AGX_SIMDMAT=1.
Wheel: see `drivers.txt` (built from this branch at ca94dc0d).
Probe harness: `~/benchq/qmm-coop-bench/qmm_coop_bench_probe.py` (sha256 in
drivers.txt), kernel-isolated mx.quantized_matmul at the real Qwen shapes;
median of 30 after 4 warmups; f16 output digest per cell. One top-level
flock `/tmp/m1-gpu.lock` per window; quiet gate inside the lock.

## The reference frame (measured, prior receipts)

| quantity | value | source |
|---|---|---|
| shipped kernel, gate_up 1053x896x9728 | 1032-1047 GFLOP/s, digest 5179630cd4a7c3f9 | qmm-codegen dump run + ilp-chains arm0 |
| zero-traffic pure-MulAdd, same dispatch geometry | 1440.4 GFLOP/s | coop-datapath arm4 |
| chains ladder in-geometry (1/2/4/8 chains) | 1013.9 / 1359.9 / 1359.6 / 1458.7 | coopmat-ilp-chains |
| isolated matrix rate, 4 chains, flat dispatch | 2148.2 GFLOP/s | fma-ceiling |
| shared-load+MMA, no staging writes, no barriers | 746.4 GFLOP/s | coop-datapath arm3 |
| chunk-staged (8 barriers/chunk, 16 KiB shared) | 336.9 GFLOP/s | coop-datapath arm1 |
| padded shared strides (16 KiB) | 269.0 GFLOP/s | coop-datapath arm2 |
| barrier-free scalar-FMA kernel | worse than coopmat | parity-status narrative |
| source de-divergence of staging loads | -13 to -26 pct on large shapes | parity-status narrative |

Gap to attribute: 494 clk/step measured vs 240 clk matrix-busy (16 ops x
14.9 clk) = ~254 clk/step of matrix-pipe idleness.

## Hypotheses and instruments

### H0 - redundant lane-invariant address arithmetic (LEADING, static leg done)

The step body recomputes ~94 of its 107 integer ops every step although they
depend only on lid/subgroup/tile/push constants. Full classification:
`analysis/isa-classification.md`. Only 13 int ops/step are step-varying.

Instruments: (a) static dataflow classification of the committed ISA dump
(done, above); (b) probe `hoist` — source hoist of the ~50 source-removable
ops, bit-preserving by construction (identical integer values, identical
guards), digest-gated, ISA-diffed, then paired-measured. (c) Driver-side
root cause named with file:line — see H6.

Result: MEASURED - FALSIFIED. Structure-pinned paired measurement
(window 3): unrollbase 818.7 vs hoist 772.3 GFLOP/s at the dominant
cell - hoisting the redundant address ops costs 5.7 percent rather
than gaining anything, while the instrument simultaneously detects the
dequant effect (point H1), so the null is real. See Measurement and
Ranked conclusion 2.

### H1 - dequantization arithmetic is the cost

24 float ops (8 ffma dequant + 8 u32_to_f + 8 fadd widen) plus ~17 int
extract ops (11 and + 10 shr incl. shared + 8 bfeil) run per step.
Instrument: probe `nodequant` — the eight dequant stores become raw
`float(packed)` casts, removing extract/cvt/ffma work (~23 ops/step) while
keeping loads, stores, guards, barriers, and the math phase unchanged.
ATTRIBUTION PROBE: values are wrong, digests differ; never a landing
candidate; global traffic unchanged (same packed word loads).

Result: MEASURED. Removing the dequant chain buys ~+9 percent within
the unrolled regime (nodequant 892.7 vs unrollbase 818.7) but changes
values (digest 36fc742484148d3b); attribution only, never a landing
candidate. See Measurement and Ranked conclusion 3.

### H2 - shared-memory bank conflicts on staging writes / fragment reads

No bank-conflict model exists anywhere in the AGX compiler (grep across
src/asahi/compiler: none), so nothing in codegen responds to conflicts.
The only measured probe is prior: arm2 padded strides (66/34) at 16 KiB
shared = 269 GFLOP/s, confounded by the 4x shared growth (arm1 shows 16 KiB
alone collapses to 337). Clock model bound: 40 shared accesses per step
cannot produce 254 idle clk unless each access cost +6 clk, i.e. conflict
factors that the shipped consecutive-word store pattern does not exhibit
per-lane-group. Verdict: not dominant, bounded small by arithmetic + arm1/2
confounds; a bit-preserving w_s layout-swap probe ([16][32] -> transpose)
is designed but NOT run (residual gap only if H0+H1 leave >100 clk/step
unexplained).

### H3 - matrix throughput at THIS fragment shape/layout below 2148

Already answered by prior receipts: the fma-ceiling 2148 figure is a flat
128-lane SPIR-V loop; inside the real dispatch geometry the same ladder
tops at 1458.7 (8 chains), and coop-datapath arm4 (fragments loaded once)
measures 1440.4 at this exact cell. The shipped kernel already sits at the
1-chain rung of that ladder (1043 vs 1014). This receipt adds no new
measurement; the 1459-vs-2148 geometry gap (0.68) remains an open
microarchitectural question (64-lane workgroups / barrier domains /
16 KiB skeleton are the differences) but it bounds NOTHING about staging:
the staging-phase target is the 1043 -> 1459 share.

### H4 - occupancy limited by shared memory, nothing hides latency

Static: the kernel uses 4 KiB shared (x_s 2 KiB + w_s 2 KiB) and 64
lanes/workgroup; HK_MAX_SHARED_SIZE is 32 KiB (hk_private.h:35). Residency
is not shared-limited at 4 KiB; the prior arms prove the sensitivity
direction (16 KiB -> 3x collapse), i.e. MORE shared hurts, and the shipped
kernel is at the small end. Subgroup count per core (2 workgroups x 2
subgroups) is fixed by the 32x32 tile geometry and identical in the arm4
ceiling measurement that reached 1440 — occupancy cannot explain the
1043-vs-1440 staging gap because the ceiling was measured at the SAME
occupancy. Verdict: eliminated by identity of geometry with the ceiling
probe.

### H5 - instruction-cache or constant-fetch pressure at this kernel size

Static: the whole shader is ~6.7 KiB of packed ISA (final dump), loop body
1666 B. M1 G13 instruction fetch operates on far smaller footprints in the
fma-ceiling scalar arms without effect, and the arm4 ceiling loop (same
kernel, staging deleted) would suffer identical fetch conditions while
measuring 1440. Verdict: eliminated by the same identity argument as H4 —
the ceiling probe shares everything except the staging phase.

### H6 - structural driver fault: no LICM anywhere in the AGX backend (NEW, file:line)

`agx_optimize_loop_nir` (mesa fork src/asahi/compiler/agx_compile.c:2762)
runs copy_prop / remove_phis / dce / dead_cf / cse / peephole_select /
phi_precision / algebraic / constant_folding / undef / loop_unroll and
NEVER calls `nir_opt_licm`; grep for nir_opt_licm across
src/asahi/compiler returns nothing while the pass exists in NIR core
(src/compiler/nir/nir_opt_licm.c). `agx_nir_lower_simdmat` runs in
`agx_preprocess_nir` (agx_compile.c:3519) BEFORE this loop, so the
coopmat-lowering placement is not the issue — the absence of hoisting is.
Consequence: every rolled loop in every AGX shader recomputes all
loop-invariant scalars per iteration; for this kernel that is the ~94
lane-invariant int ops/step, of which ~40 (the coopMatLoad fragment
addresses) are unreachable from GLSL and only a driver LICM can lift.
This also answers the uniform-hoisting question raised for the
wait-batching work: there is no uniform-register hoisting path to be
"missing" — there is no hoisting at all.

Instrument: static (file:line + pass-list evidence + the ISA classification
it predicts). Fix candidate handed to the driver workstream: add
`NIR_PASS(_, nir, nir_opt_licm);` to the AGX optimize loop, prove-fires per
discipline, ISA-diff against the committed classification.

## Ranked conclusion

1. **The staging phase is structure-bound, not work-bound.** The single
   largest term is the rolled step loop itself: unrolling with
   bit-identical arithmetic costs -21.3 percent at the dominant cell
   (unrollbase 818.7 vs base 1040.3). The shipped kernel's compact
   229-instruction rolled body is the best measured form; every
   restructure measured across this and prior receipts (TILE_M64,
   STEP_K32, de-divergence -13..-26 percent, arms 1-3, both probe
   families) is slower. This also retroactively explains the
   de-divergence result: body-size perturbations flip
   scheduling/unroll behavior worth more than the work they touch.
2. **H0 (lane-invariant address-op redundancy) is FALSIFIED as a
   performance cost.** In the structure-pinned comparison the
   instrument is demonstrably sensitive (it sees the dequant effect,
   point 3), and it reports that hoisting the ~40-50 redundant address
   ops per step makes things ~4-5 percent WORSE, not better
   (hoist 772.3 vs unrollbase 818.7). The redundant address work hides
   under other constraints; it is not stealing issue slots from the
   matrix pipe. Consequence for H6: a driver-side LICM pass would
   remove the ~94 invariant ops but the measured expectation is ~zero
   performance gain (possibly negative); LICM remains a code-size/
   generality improvement for the AGX backend, not a prefill lever.
   The window-2 timing of the probes was already dominated by the
   unroll pathology, consistent across windows 1-3.
3. **Dequant arithmetic is a real second-order cost (~+9 percent at
   large shapes) with no bit-preserving removal** - the extract/cvt/
   ffma chain matters (nodequant 892.7 vs unrollbase 818.7), but its
   removal changes values, so it is only reachable through an
   arithmetic change under docs/parity-id-policy.md (e.g. an
   f16-operand kernel; see the known coopmat datapath receipt).
4. **The residual 1040 -> 1459 gap to the zero-traffic ceiling is
   phase organization** (barrier cadence + serialized-load latency the
   wait-batching work targets, measured ~+1.5 percent unpaired), not
   staging instruction count: neither removing address work (point 2),
   removing dequant work (point 3, +9 percent, arithmetic-changing),
   nor adding ILP (coopmat-ilp-chains ladder) closes it, and the
   zero-traffic arm4 probe runs the same geometry with NO staging at
   all.


## Measurement (window 3, commit e21179eb, 2026-09-12)

Structure pinning took three windows. Windows 1-2 (commits ca94dc0d,
22b9c35e) are SUPERSEDED: both showed that tiny body-size deltas flip
`nir_opt_loop_unroll` into restructured k regions (step loop unrolled,
phase counts changed) that move the number -14 to -26 percent on their
own, swamping the attribution; a min()-opaque trip count did not stop
it and SPIR-V DontUnroll loop control is not honored end to end on this
stack. Window 3 pins the structure in the SOURCE: all three probe paths
expand the four 16-k steps via one macro (QMM_STEP_BODY), so they share
identical structure by construction; `unrollbase` (shipped arithmetic
verbatim, unrolled) is the structure control. Digest gate: hoist AND
unrollbase bit-identical to base on all 8 cells x 3 rounds (PASS);
nodequant value-wrong by design (digest 36fc742484148d3b).

gate_up 1053x896x9728, median of 3 interleaved rounds (GFLOP/s):

| arm | value | vs base |
|---|---|---|
| base (shipped, rolled step loop) | 1040.3 | - |
| unrollbase (shipped arithmetic, unrolled) | 818.7 | **-21.3%** |
| hoist (address ops hoisted, unrolled) | 772.3 | -25.8% (-5.7% vs unrollbase) |
| nodequant (dequant arithmetic removed, unrolled) | 892.7 | -14.2% (+9.0% vs unrollbase) |

All eight cells (base | hoist | nodequant | unrollbase, GFLOP/s; hoist%
nodeq% unrb% vs base):

| shape | base | hoist | nodequant | unrollbase | deltas |
|---|---|---|---|---|---|
| 1053x896x9728 | 1040.3 | 772.3 | 892.7 | 818.7 | -25.8 -14.2 -21.3 |
| 1053x4864x896 | 981.0 | 738.8 | 841.1 | 778.2 | -24.7 -14.3 -20.7 |
| 1053x896x896 | 848.0 | 672.3 | 752.4 | 711.7 | -20.7 -11.3 -16.1 |
| 262x896x9728 | 900.2 | 677.7 | 776.6 | 724.9 | -24.7 -13.7 -19.5 |
| 262x4864x896 | 841.1 | 580.9 | 615.2 | 579.8 | -30.9 -26.9 -31.1 |
| 262x896x896 | 525.4 | 481.9 | 497.1 | 479.4 | -8.3 -5.4 -8.8 |
| 1053x896x128 | 345.2 | 412.9 | 431.6 | 367.8 | +19.6 +25.0 +6.5 |
| 262x896x128 | 86.2 | 109.5 | 90.5 | 93.1 | +27.0 +5.0 +8.0 |

The large-shape pattern is uniform: unrolling the shipped rolled step
loop costs 20-31 percent with IDENTICAL arithmetic; within the unrolled
regime, removing dequant arithmetic buys back ~7-9 percent while
hoisting the lane-invariant address ops LOSES a further ~4-5 percent.
Only the tiny 128-wide-K cells flip sign (short k regions like
unrolling).

## Artifacts

- Probe logs + digest gates: `logs/probe-{base,hoist,nodequant,
  unrollbase}-{warmup,r1,r2,r3}.log` (JSON lines), `drivers.txt`
  (wheel c9a1b621 -> e21179eb builds; window 3 wheel sha256 in
  drivers.txt, source commit e21179eb).
- Raw AGX shader dumps (about 1 MB each, kept on jwm1 at
  `~/src/mlx-omarchy-qmmattr/receipts/2026-09-12-qmm-staging-attribution/logs/dump-*.log`,
  sha256 in `dump-sha256.txt`): Mesa dumps every compiled variant per
  run, so per-kernel extraction needs the section filter in
  `analysis/extract2.py`; static counts quoted here are qualitative
  (base = rolled step loop matching the committed 2026-09-12-agx-qmm-
  codegen classification; probe paths = source-expanded straight-line
  k region, structure identical across paths by construction).
- Aggregator: `analysis/aggregate.py`; window script:
  `analysis/qmm_dump_one.py` + `qmmattr_window.sh` (repo root of the
  jwm1 tree).
- Static classification (unchanged by measurement):
  `analysis/isa-classification.md`.

Probes vs candidates, restated: `hoist` and `unrollbase` are
bit-preserving and REJECTED as performance candidates by measurement;
`nodequant` is value-wrong and was never a candidate. No source change
is proposed by this receipt; the shipped kernel stands as the best
measured form and the structural levers that remain are
arithmetic-changing (parity-id-policy territory) or driver-side
(wait-batching, matrix lowering width).


## Probes vs candidates — explicitly

- `hoist` is bit-preserving and, if it wins, is a legitimate landing
  candidate AFTER the full parity-id-policy gates (six canonical legs x
  three reps, digest pins).
- `nodequant` is an attribution probe only. Its outputs are wrong by
  construction; it is never a landing candidate under any outcome.
- H2's layout-swap probe is designed but unrun; H3/H4/H5 carry no new
  measurement by prior-receipt identity arguments.
