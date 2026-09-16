# 2026-09-15 TDT GPU step — fused decoder dispatch (agent/tdt-gpu-step)

Status: **exactness proven end-to-end on both arms; performance regressed —
default decode path NOT swapped. NO-LAND on perf.**

The whole per-emission TDT decoder step (embedding gather, both BNNS-contract
LSTM layers, projector, joint head, both argmax reductions) runs as ONE
`mx.fast.metal_kernel` dispatch per decoder call, with the TDT joint result
carried for the matching `run_joint` callback, so decoder state never leaves
the GPU between calls. Gate: token-exact must hold on both arms. It holds.
The wall-time gate does not: `tdt_decode` went 2880.49 ms → 6793.59 ms.

## Receipt numbers (jwm1-linux, flock -x /tmp/m1-gpu.lock, never stolen)

| arm | metric | before (origin/main overlay) | after (fused step) |
| --- | --- | --- | --- |
| native encoder (diagnose_decision_142) | free_decode first_divergence | null | **null** |
| native encoder | decision 142 token / gap | 7892 / 0.0 | **7892 / 0.0** |
| native encoder | decoder/joint calls | 145 / 146 | **145 / 146** |
| native encoder | joint_on_native traces | 16/16 | **16/16** |
| ANE encoder (parakeet_e2e) | status | match | **match** |
| ANE encoder | emissions | 104/104 | **104/104** |
| ANE encoder | transcript sha | db501a8c080380ea… | **db501a8c080380ea…** |
| ANE encoder | tdt_decode | **2880.49 ms** | **6793.59 ms** |

Before number: receipts/2026-09-15-encoder-fused-leftover/e2e-report.json
(same overlay state, same invocation). After:
`/var/tmp/TdtGpuStep/e2e-after/e2e-report.json` (copied to
`derivation/e2e-report.json`). decoder_mean 16.2 → 28.3 ms, joint_mean
3.6 → 18.4 ms.

## What is now proven (the reusable part)

1. **fp16 semantics of the omarchy Vulkan backend on G13G B1** (probe
   `derivation/fma16_probe.py`, 262144 cases): fp16 add, fp16 mul, bit-indexed
   `float16_t` LUT reads, `packHalf2x16` bit extraction and the fp32→fp16
   convert are exact. Hardware `fma()` on fp16 is **double-rounded through
   fp32** (48490/262144 mismatched) and cannot express the BNNS single-rounded
   fp16 FMA.
2. **exact_fma16**: `rnd16(acc + x·y)` built from a TwoSum split in fp32 (fp16
   products are exact in fp32) plus an explicit tie correction (round bit =
   mantissa bit 12, visible sticky = bits 11..0, unsigned half-ulp16 bump
   directed by the residual's sign). 0/262144 mismatches vs the fp64 golden,
   including adversarial ties and extreme exponent spreads. Requires the GLSL
   `precise` qualifier on every exactness-critical fp32 op — without it the
   compiler reassociates and the TwoSum collapses to zero (this silently
   produced plausible-but-wrong values; the mel kernels' `precise` usage is
   the existing convention this follows).
3. **mx.matmul contract** for fp16 `[1,K]@[K,N]`: fp32 ascending accumulation
   rounded once to fp16, then a separate fp16 bias add. The fused projector
   and joint reproduce it bit-for-bit (probe `derivation/matmul_contract_probe.py`,
   0/640 × 8 trials), which is what makes the joint fusion contract-legal.
4. **The fused kernel is bit-exact against the landed numpy BNNS contract and
   run_joint** on real pinned weights: `derivation/validate_step.py` —
   VALIDATION PASS 0/24 trials, all stages (next_hidden, next_cell,
   decoder_hidden, token/duration logits, argmax decisions) exact, including
   recurrent chaining; mode-1 joint-only dispatch vs run_joint 12/12
   (`derivation/mode1_check.py`).
5. Both acceptance arms reproduce with the fused path (numbers above).

## Why it is slower, and the follow-up

The exact_fma16 emulation costs ~18 fp32 ops per k-term, and the kernel runs
ONE threadgroup of 640 threads (single dispatch), so ~1/8 of the AGX sits
busy behind a 1280-deep serial dependency chain per lane while the landed CPU
contract uses every core through numpy fp64. 28.3 ms/call vs 16.2 ms.

Follow-up (exactness-preserving, laid out in the kernel docstring): go
lane-parallel with 1024+ threads (2560 lanes, preact to shared, gates/cell
pairing per thread), then split k into per-thread segment chains — the BNNS
128-block fold structure lets partial block sums combine exactly in fixed
order, so parallel k does not break the contract. Joint+argmax can move to a
second dispatch sized 8198 lanes.

## Artefacts

- `overlay/tools/coreml/vulkan_decoder_step.py` — the kernel + host packing
  (`pack_step_weights`, `run_step`), mode 0 full step / mode 1 joint-only /
  mode 2 debug dump. LUT `fused_lut_2026-09-15.npz` sha 5a7591b8… unchanged;
  pinned decoder untouched (637f077e…).
- `derivation/`: fused_e2e.py (patched harness callbacks), fused_diagnose.py,
  validate_step.py, mode1_check.py, fma16_probe.py, matmul_contract_probe.py,
  preact_iso.py, layer0_iso.py, e2e-report.json (after run).
- jwm1 stages: `/var/tmp/TdtGpuStep/pkg` (origin/main overlay + the new
  module), `/var/tmp/TdtGpuStep/e2e-after/`.
- Branch `agent/tdt-gpu-step` (worktree ~/src/tdt-gpu-step-wt), based on
  origin/main 108fd4b5. NOT merged; commit 63c1d3cf untouched.

Resolved model: zai/glm-5.3-flash
