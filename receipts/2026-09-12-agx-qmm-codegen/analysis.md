# AGX codegen analysis: shipped qmm coopmat kernel (2026-09-12)

Kernel: `shaders/qmm_coopmat.comp`, compiled by the installed driver
(mesa-honeykrisp-omarchy 26.3.0.devel.hk6f6afc8-1, git-6f6afc8968).
Dump: `dumps/qmm-coopmat-full-dump.log.gz` (AGX_MESA_DEBUG=shaders,
MESA_SHADER_CACHE_DISABLE=true, AGX_SIMDMAT=1), captured 2026-09-12 on
jwm1 during a live run of the same kernel at 1053x896x9728 that measured
**17.5371 ms = 1046.7 GFLOP/s, f16 digest 5179630cd4a7c3f9** - the same
digest and within noise of receipts/2026-09-11-coopmat-ilp-chains arm0
(1043.0 GFLOP/s), so the dump and the number describe the same code.

## Per-16-k-step instruction accounting (final packed ISA)

The steady-state loop body is 229 issued instructions per lane
(`while $r36 < 4` per 64-k chunk; one step body, looped). Addresses
0x1a4-0x65e are the staging phase (179 instr), 0x660-0x830 the math phase
(50 instr).

| class | count | detail |
|---|---|---|
| `simd_matrix_fmadd32` | 16 | exactly the 16 the algorithm needs (2 ksub x 2 A-frag x 4 B-frag); no redundant issue |
| `lload` (shared) | 24 | scalar 32-bit; each 8x8 fragment = 2 per lane at adjacent addresses (r38, r38+4) - pairable to 12 vector loads |
| `lstore` (shared) | 16 | 8 x_s + 8 w_s, scalar |
| `load` (global) | 5 | 4 x words + 1 w word, each inside its own divergent guard |
| `wait` (scoreboard) | 5 | **one per global load, within 2 instructions of it** |
| `barrier` | 2 | mandatory (single-buffered staging) |
| integer addr/bit math | 107 | 65 iadd + 11 and + 10 shr + 8 imadd + 8 bfeil + 5 shl; ~90 recomputed from lane-constant values every step |
| float/cvt | 24 | 8 ffma (dequant) + 8 u32_to_f + 8 fadd (f16->f32 widen via `fadd x, -0.0`) |
| control flow/mov/ldimm | 32 | 5 if/else/pop_exec guard triplets + loop bookkeeping |

Register pressure: **max r49 of 256**, 50 distinct registers, **zero**
stack/spill operations in the whole shader. The 8 accumulator fragments
(r18-r33, 16 f32) live in registers across the entire k loop including
the staging phase.

## Where the time goes

The matrix pipe needs 16 ops x ~14.9 clk = **240 clk/step** when
sustained (the receipts/2026-09-11-fma-ceiling 4-chain rate, 2148
GFLOP/s). The measured 1046 GFLOP/s implies **~494 clk/step**, so the
matrix pipe idles about half of every step. The 229 issue slots (~229
clk) alone cannot explain the gap; the missing ~250 clk are memory
latency and barrier path, and the ISA shows exactly where:

1. **Serialized global loads (primary, driver fault).** The GLSL loads
   all four x words in one loop and unpacks them in a second loop
   (qmm_coopmat.comp:126-143) - the source already batches. The driver
   re-serialized them: each load sits in its own guarded then-arm, and
   `agx_insert_waits_local` drained ALL pending scoreboard messages at
   EVERY block exit - including exec-mask transitions that are not real
   branches - and only tracked 2 slots. The ISA shows
   load -> wait -> else -> join four times over (0x1fc/0x204,
   0x25e/0x266, 0x2c0/0x2c8, 0x322/0x32a, plus 0x466/0x46e for the w
   word): four exposed memory latencies per step instead of one.

2. **Loop-invariant address math recomputed per step (secondary,
   driver).** ~90 of the 107 integer ops per step rebuild lane-constant
   values: `lid/8`, `lid%8`, the x_s/w_s store addresses, w_base, and
   the 12 fragment load addresses derived from
   `thread_index_in_simdgroup`. NIR CSE in the AGX pipeline is
   block-local; nothing lifts them across the step-loop boundary.

3. **Scalar fragment loads (tertiary, driver).** 24 scalar `lload`
   where 12 vector loads would do; the two elements per lane per
   fragment are adjacent (8 B) in shared memory.

Inside the math phase the code is already good: the ks=1 fragment loads
and address updates are issued between the two 8-fmadd bursts
(0x6b8-0x750), so shared-memory latency hides under matrix work, and
the bursts have dependency spacing 8. This matches the
coopmat-ilp-chains finding that bursts are not the problem - the phase
structure between them is.

## What is NOT the cause

- Matrix issue count: 16/16, zero waste.
- Accumulator spills: none (0 stack ops; fragments persist in r18-r33).
- Register pressure: 50/256 registers.
- MulAdd issue order: spacing 8 inside bursts, already interleaved.
- Barrier count: 2/step is the kernel's single-buffered staging
  structure; the driver cannot legally merge those barriers or move
  memory across them.

## Driver patch

`hk/agx-wait-batching` @ 84fcd220de1 (mesa fork, base 6f6afc89684 =
installed package): `agx_insert_waits.c` now carries pending scoreboard
state across exec-mask fallthrough edges of divergent if/else regions,
seeds the merge with the union of both arms' pending writes, inserts
waits at first use, uses 4 slots (2-bit packed immediate), and falls
back to the old conservative per-block drain whenever the CFG shape
does not match the tracked if/else structure (real branches, back
edges, and barriers still drain). Built locally for A/B:
`~/src/mesa-wt-waitbatch` (patched) and `~/src/mesa-wt-waitbatch-base`
(unpatched base), selected by the ICD JSONs in
`~/benchq/qmm-coop-bench/icd-waitbatch-{base,patched}.json` - the
system driver is never overwritten.

A/B timings and the six-canonical-digest screen are appended to
verdict.json after the measurement windows.
