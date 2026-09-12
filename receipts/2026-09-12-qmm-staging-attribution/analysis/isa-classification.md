# ISA classification: every instruction of the 16-k step body (2026-09-12)

Source dump: `../2026-09-12-agx-qmm-codegen/dumps/qmm-coopmat-full-dump.log.gz`
(final packed ISA), step-loop body 0x1b0..0x832, classified by dataflow on
2026-09-12. Registers: `r36` = step counter (0x196 ldimm, 0x828 increment),
`r34` = chunk counter (outer loop, updated at 0x83e), `r1` = lid,
`r2` = subgroup id, `r3/r4` = lid%32 / lid/32, `r14` = tile row base.
An instruction is STEP-VARYING iff its def transitively depends on `r36`
(otherwise INVARIANT). The full body re-derives all of these every step;
nothing is carried across iterations.

Opcode histogram (226 issued in body window; 229 with while/jmp edges):

| op | n | op | n | op | n |
|---|---|---|---|---|---|
| iadd | 65 | lload | 24 | simd_matrix_fmadd32 | 16 |
| lstore | 16 | and | 11 | shr | 10 |
| imadd | 8 | fadd | 8 | u32_to_f | 8 |
| ffma | 8 | bfeil | 8 | mov | 7 |
| if/else | 5+5 | load | 5 | wait | 5 |
| ldimm | 5 | pop_exec | 5 | barrier | 2 |
| shl | 5 | | | | |

## Step-varying integer ops: 13 of 107

| site | op | role |
|---|---|---|
| 0x1b8 | iadd r37 += r36<<4 | k_base = chunk*64 + step*16 |
| 0x1d6, 0x238, 0x29a, 0x2fc | shr ×4 | k_base/2 for the four x words |
| 0x1f4, 0x256, 0x2b8, 0x31a | iadd ×4 | x address += k_base/2 |
| 0x446 | shr | k_base/8 for the w word |
| 0x44e, 0x45e | iadd, imadd | w address += k_base/8 |
| 0x828 | iadd | step counter++ |

(0x1b0 `shl r37 = r34<<6` is chunk-varying but step-invariant — still
recomputed 4x per chunk.) A hand-written lowering needs ~13 of these.

## Lane-invariant recomputed ops: 94 of 107

Every one of these depends only on lid, subgroup id, tile indices, and
push constants — constant across all 56 steps of a 1053x9728x896 tile
pass (14 chunks x 4 steps):

| block | sites | ops | source construct |
|---|---|---|---|
| x global address invariant parts | 0x1c0-0x31a | ~20 | `(lid+64j)/8`, `row*x_words_per_row`, `(lid+64j)%8` per load |
| x_s store addresses | 0x340-0x438 | ~20 | `local_row*STEP_K + 2*(i%8)` per store, recomputed per element |
| w_s base + 8 store offsets | 0x484-0x5a0 | 10 | `8*w_part*TILE_N + w_column_local`, `base + q*TILE_N` |
| w word invariant parts | 0x456 | 1 | `w_row + w_part` |
| fragment addresses (coopMatLoad lowering) | 0x5c6-0x658 + math phase 0x6b8-0x7f0 | ~40 | `thread_index_in_simdgroup` extraction chain + 28 per-fragment `uXX + base` offsets (12 pre-barrier + 16 between madd bursts) |
| chunk term | 0x1b0 | 1 | `chunk*CHUNK_K` inside the step loop |
| dequant w_base reuse | 0x49c-0x5b6 | ~3 | address iadds feeding u32_to_f/ffma/lstore chains |

## Why the optimizer never hoisted them

1. **Composite lane+loop expressions.** The GLSL writes each address as
   one expression per use: `i = lid + 64*j; local_row = i/8; ...
   + (i%8)`. Hoisting `lid/8` and `lid%8` out of the step loop requires
   reassociating `(lid + 64j)/8 == lid/8 + 8j` and `(lid + 64j)%8 ==
   lid%8`. NIR CSE is block-local here and no reassociation pass runs
   (the AGX pipeline's LICM sees only whole defs; the defs themselves
   depend on the loop-carried `step` through `+64j` folded constants —
   except the constants are loop-invariant, so even plain LICM should
   lift the `lid`-only subexpressions if they existed as separate SSA
   values — they never do).
2. **coopMatLoad lowering emits per-use address arithmetic.** The ~40
   fragment-address ops derive from `thread_index_in_simdgroup`
   (0x5c6) and per-site uniform offsets, recomputed at every coopMatLoad
   expansion inside the step loop. This is the AGX cooperative-matrix
   lowering running after the scalar-optimization window (post-LICM), so
   no NIR pass can lift them. Source-level restructuring cannot touch
   this block: GL_KHR_cooperative_matrix exposes no per-lane element
   access (already named in receipts/2026-09-11-qmm-coop-datapath,
   finding 5).

## Consequence for the attribution probes

- Probe `hoist` (QMM_COOP_PROBE_PATH=1) removes the source-side share:
  x-address invariant parts (~20), x_s addresses (~20), w_s base (2),
  w word invariant part (1), chunk shl (1) ≈ 40-50 ops/step, leaving
  the per-step body at ~185-190 instructions vs 229. Bit-preserving by
  construction (identical integer values, identical guards).
- The remaining ~40-50 lane-invariant ops/step are
  backend/lowering-generated (fragment addresses, per-store offset
  iadds that survive as `base + const`). Removing them requires a
  driver-side change: hoist or CSE the address arithmetic at/after
  cooperative-matrix lowering, and/or fold constant displacements into
  the scalar shared lload/lstore displacement field (all 24 lloads and
  16 lstores currently use displacement 0; the 24 scalar lloads are
  also pairable to 12 vector loads — separate line item from
  2026-09-12-agx-qmm-codegen).
