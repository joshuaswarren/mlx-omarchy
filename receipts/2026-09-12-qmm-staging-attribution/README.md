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

Result: MEASURED [fill: probe numbers]

### H1 - dequantization arithmetic is the cost

24 float ops (8 ffma dequant + 8 u32_to_f + 8 fadd widen) plus ~17 int
extract ops (11 and + 10 shr incl. shared + 8 bfeil) run per step.
Instrument: probe `nodequant` — the eight dequant stores become raw
`float(packed)` casts, removing extract/cvt/ffma work (~23 ops/step) while
keeping loads, stores, guards, barriers, and the math phase unchanged.
ATTRIBUTION PROBE: values are wrong, digests differ; never a landing
candidate; global traffic unchanged (same packed word loads).

Result: MEASURED [fill]

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

[fill after measurement]

## Probes vs candidates — explicitly

- `hoist` is bit-preserving and, if it wins, is a legitimate landing
  candidate AFTER the full parity-id-policy gates (six canonical legs x
  three reps, digest pins).
- `nodequant` is an attribution probe only. Its outputs are wrong by
  construction; it is never a landing candidate under any outcome.
- H2's layout-swap probe is designed but unrun; H3/H4/H5 carry no new
  measurement by prior-receipt identity arguments.
