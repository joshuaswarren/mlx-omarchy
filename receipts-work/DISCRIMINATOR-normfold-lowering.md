# DISCRIMINATOR — norm-fold "lowering wall" (t6001 NORM_PROLOGUE vs standalone fast_norm)

Date: 2026-09-24 · Box: jw16 (jw16mbp1-linux, Apple M1 Max G13C) · Lane: Jw16GpuSubmit.SpirvDisc
Driver: honeykrisp `/usr/local/lib/libvulkan_asahi.so.d3fa18e` (mesa-1 d3fa18e8dd0, tip-1)

## Verdict

**The hypothesized lowering wall does not exist.** Across all four compilation
levels (SPIR-V, NIR×2 stages, AGX IR, final binary disassembly), the
`fast_rms_norm_bf16` pipeline and the `qmm_vec_q4_multi_subgroup_bf16_np`
prologue reduce the norm with **numerically identical AGX code**. Exactly one
lowering difference exists — FMA contraction of the square accumulation — and
it is **provably value-identical for bf16 inputs** (v·v is exact in fp32, so
`ffma(v,v,sq)` ≡ `fmul`+`fadd` in every rounding step). Combined with 12/12
byte-identical micro dispatches, the 44/320 gate flips **cannot originate in
shader lowering**. The discriminator refutes the mode-0-class wall and
redirects the lane to the host/tape domain (scheduling or a non-shader value
difference).

## Method

- Provenance: `fast_rms_norm_bf16.spv` and `qmm_vec_q4_multi_subgroup_bf16_np.spv`
  from the gate build tree recompiled with the exact CMake glslc args
  (`glslc -O --target-env=vulkan1.3 -DUSE_BF16=1 [-DUSE_SUBGROUP=1 -DQMM_VEC_Q4_WORD=1
  -DQMM_VEC_MULTI=1 -DNORM_PROLOGUE=1]`) → **sha256 byte-identical**
  (`985093d1…`, `35c4009b…`). Shaders unchanged bfbf7581..HEAD (empty diff), so
  these are the gate SPIR-Vs.
- Dumps: `vkdump.c` harness — vkCreateInstance → vkCreateDevice (Apple pd) →
  vkCreateComputePipelines **only**; zero queue submissions, zero dispatches.
  `AGX_MESA_DEBUG=shaders MESA_SHADER_CACHE_DISABLE=true` captures per-pipeline
  NIR + AGX IR + final disassembly to stdout. Plain variant dumped identically.

## Findings by level

1. **SPIR-V** (confirms prior lane): identical op sequences for the reduction
   (FMul+FAdd square accumulate, FAdd tree, exact-division expansion,
   `InverseSqrt`). Decoration asymmetry: `np.spvasm` carries **212
   NoContraction**, `base.spvasm` (fast_norm) **0**, `plain.spvasm` (qmm
   without prologue) 200. Origin: qmm_vec.comp declares `precise float`
   (dot/input_sum/accumulator/bias_term — receipts/2026-09-09-q4-gemv-order);
   fast_norm.comp has none. The `precise` regime marks the whole function's
   float ops, including the prologue copy.
2. **NIR (post-import, pre-AGX)**: standalone contracts
   `%22 = ffma %21, %21, %13` (sq += v·v, one rounding). np keeps
   `%57 = fmul %56, %56 // exact` + `%58 = fadd %35, %57 // exact`
   (NoContraction forbids contraction). Everything else — 8-level shared tree
   (128→1), exact-division expansion (fmul + 2×ffma NR + NaN cmpsels from
   preamble uniforms), `frsq`, eps add — same op shapes.
3. **AGX IR / final disasm**: reduction tails op-identical
   (`fmul $r6,$r5,u15; ffma $r5,u14,$r6,r5; ffma $r5,r5,u15,$r6;
   fadd $r5,$r5,u5; rsqrt r5,r5` vs the same at u22/u23/u24 in np).
   y-store loops: `shl 16; fmul; fmul` (commutative-exact order), identical
   RNE `bf16_store` bit math. Plain-vs-np normalized full-disasm diff: **zero
   float-op differences outside the prologue**; dot-loop diffs are
   addressing-only (shared `normed_row` vs global x word loads, load-slot
   letters).

## Why the one difference is neutral

ffma computes RN(v² + sq); mul+add computes RN(RN(v²) + sq). bf16 v has an
8-bit significand → v² has ≤16 bits, exact in fp32 (24-bit) for all normal
results, so RN(v²) = v² and both paths round identically. Denormal v²
(|v| < ~3.4e-19) is the only theoretical divergence window — unreachable for
LLM residual-stream activations, and the gate flips are large structured
argmax losses, not denormal noise.

## Empirical confirmation

`micro_fold.py` (candidate wheel bfbf7581, /var/tmp/normfold-venv),
MLX_OMARCHY_FUSED_GEMV_NORM=1 vs =0, fold-on/off output hashes:
**K ∈ {2048, 4096, 8192} × seeds {0,1,2,3} = 12/12 byte-identical**
(p1, p2, out). Includes the gate model's K=2048.

## Consequences for the lane

- Do not spend further iterations on shader-level exactness knobs for the
  prologue (explicit `fma()` splits, precise modifiers, reorders): the code
  already lowers to result-identical arithmetic. Deliverable B is void by
  evidence, not skipped.
- The flip source is dynamic or host-side: candidates in order —
  (a) early-fired group ordering vs producer writeback (stable-timing race;
  probe: delay the fused group one turn, or event-fence xb before dispatch),
  (b) a runtime value difference in the fused dispatch's push-constant block
  beyond alpha/rhs_offset/lhs_offset (all three verified correctly wired in
  `dispatch_quantized_gemv_group`, primitives.cpp:7441-7445),
  (c) tape-level interaction of the merged group with GDN state updates
  (kv-direct already refuted by the byte-identical signature under bfbf7581).
- Sharpest next probe: dump the prologue's shared `normed_row` bytes and the
  standalone norm's output row bytes for the same x in the model tape
  (debug-store knob), at the first diverging token (p1 step 14). Byte-equal ⇒
  definitively scheduling; byte-different ⇒ the x or weight the prologue sees
  differs from the standalone's input.

## Artifacts

- jw16:/var/tmp/sdv-discriminator/ — vkdump.c, standalone.dump (1562 l),
  np.dump (12735 l), plain.dump (8608 l), normalized disasms, rsqrt windows,
  standalone.err/np.err/plain.err (pipeline result 0, device lines).
- Regen:
  `AGX_MESA_DEBUG=shaders MESA_SHADER_CACHE_DISABLE=true ./vkdump <spv> > out.dump 2> err`
- Branch copy: receipts-work/DISCRIMINATOR-normfold-lowering.md (this file),
  receipts-work/spirv-discriminator/{vkdump.c, np-rsqrt.win, sa-rsqrt.win}.
