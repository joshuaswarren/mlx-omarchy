# Requalification: native-order dense BF16 decode GEMV against the parity contract

Date: 2026-09-10. Agent: Bf16NativeOrderRequalify.
Branch under test: `wave/Bf16DecodeGemvRebased` — commit 6f1612c2 ("perf: match
native BF16 decode GEMV order") rebased onto origin/main 7aa8cf0f.
Status: IN PROGRESS — this file is finalized at the end of the window.

## Question

The candidate was set aside on a float64-purity argument. Requalified against
the actual project parity contract (`docs/parity-id-policy.md`): a candidate
may move a Linux-divergent leg only TO the native macOS digest, may not move
any already-native leg, and both drivers must agree. Does this kernel land?

Decision inputs being measured:

1. Three BF16 generated-id digests vs native macOS `7fc0f968789b1882`
   (short), `407b7624ed1b3b29` (long), `ff502900d2a179a5` (1K ctx) on the
   cooperative-matrix fork AND stock Mesa.
2. Six Q4 canonical digests untouched.
3. Paired decode/prefill medians at 30/262/1053 legs, both drivers, as a
   fraction of the committed base-M1 native baseline (ea09eefa).
4. Worst-case and typical bf16 ULP error vs RNE(f64) for the current
   sequential kernel and the candidate, including the bias-cancellation
   elements.
5. omarchy_matmul_family_tests + omarchy_runtime_tests on llvmpipe (x64) and
   on the M1.

NOT landing in this ticket either way. The receipt exists so the land/no-land
decision is made on evidence.
