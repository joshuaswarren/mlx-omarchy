# Q4 long-context scaling: prefill 0.61 / decode 0.69 of native

Receipt for 2026-09-11, agent Q4LongCtxScaling. Host: jwm1-linux, Apple M1
(G13G B1), Honeykrisp fork driver. Base: origin/main 63c9a8d8 + branch
q4/longctx-scaling (d89c4e9f). Wheel:
mlx_omarchy-0.32.2.dev202609112326+d89c4e9f-cp314-cp314-linux_aarch64.whl.

## The question

The Q4 short legs sit at or past native (prefill 1.13, decode 0.74) but the
1K-context legs fall to prefill 0.61 / decode 0.69. What does CONTEXT do to
the Q4 path, and can the scaling part be closed bit-preservingly?

## Decode: the deficit growth is one kernel, and it is fixed

Per-token decode walls (committed denominators, receipts/
native-baseline-2026-09-06 vs fork):

| leg (k range)   | native ms/tok | fork ms/tok (b6d662a) | fork-native |
|-----------------|---------------|-----------------------|-------------|
| short (30-61)   | 6.64          | 8.89                  | +2.25       |
| 262 (262-389)   | 6.81          | 9.16                  | +2.35       |
| 1K (1053-1084)  | 7.12          | 10.28                 | +3.16       |

Fork's deficit grows +0.91 ms/token from short to 1K while native's whole
wall grows only +0.48 ms over the same k range. The only decode component
whose work grows with k is attention: the 24-layer sdpa chain microbench
(receipts/2026-09-10-decode-attribution) grows 0.80 -> 2.12 ms from k=16 to
k=1024, i.e. +1.3 ms — the entire fork growth. That chain streams
12.6 MB more KV per token at ~10 GB/s effective, against the 68 GB/s part
roof and the ~41 GB/s the Q4 GEMV already achieves. The mechanism inside
SdpaDecodeNativeF16: one 1024-thread workgroup owns one query head; each
subgroup holds its online-softmax state and issues only ~4 scalar 2-byte
loads per key iteration, with the per-key subgroupAdd fencing the AGX
scheduler from hoisting later loads across iterations. Against ~700 ns DRAM
latency that is ~a few KB in flight per core -> single-digit GB/s. Native's
sdpa_vector pays the same algorithm at ~26 GB/s effective (its 0.48 ms
growth), i.e. native hides latency better but is itself not at the roof.

### The fix (bit-preserving, landed on q4/longctx-scaling)

aux_size==1 selects packed 32-bit loads of adjacent f16 pairs through
aliasing word views (bindings 4/5/6) and 8-key batched load software
pipelining in both the one-pass and two-pass loops: 32 loads in flight per
subgroup instead of 4. The primitive selects it only when every indexed row
base and row stride is even, so the packed loads address exactly the values
the scalar loads read. The per-key fma/exp/subgroup order, the
key-to-subgroup mapping, and both merge epilogues are the frozen legacy
order verbatim; the legacy scalar loops remain in the kernel and remain
reachable via MLX_OMARCHY_SDPA_DECODE_SCALAR=1, which is the paired A/B
arm. No arithmetic changes, so the rule-2 pinned Q4 short and 1K-context
digests cannot move by construction. SPIR-V evidence: opcode histogram vs
the released kernel shows additions only in loads/control-flow/index math;
no new arithmetic or rounding op kinds.

Window evidence (this receipt): 17-k_len bit-identity sweep (one-pass, 64-
and 128-block two-pass regimes, strided capacity-4096 cache slices) packed
vs scalar uint16-exact; paired sdpa chain micro at k=30/262/1053; three Q4
legs x {packed, scalar} x 2 reps, every run asserting the canonical
generated-id pins on both arms. Results in verdict.json.

## Prefill: receipt-only negative — the mechanism does not scale, it saturates

Three-rung qmm decomposition (fork, component-isolated probes; m=1053 and
m=30 from receipts/2026-09-10-qmm-prefill-tile, m=262 from
qmm_m262_probe.py in this receipt):

| rung | fork leg | qmm ms/model | qmm share | fork qmm GFLOP/s | native leg (implied native qmm) |
|------|----------|--------------|-----------|------------------|---------------------------------|
| m=30   | 90.6 ms | ~40-45  | ~45% | ~480 (latency-bound) | 102 ms |
| m=262  | 270 ms  | 219.8   | 81%  | ~806                 | 216 ms |
| m=1053 | 946 ms  | 725.7   | 77%  | ~982                 | 572 ms |

Named mechanism: the deficit does NOT come from anything context-specific
(mask handling, KV traffic, recomputation — all measured or bounded below
16% of the leg). It is the qmm coopmat kernel's asymptote: ~1.0 TFLOP/s
versus native's implied ~1.66 TFLOP/s, multiplied by a qmm share that
grows linearly with context. The absolute deficit goes -11 ms (m=30, fork
already ahead) -> +54 ms (m=262) -> +374 ms (m=1053), exactly tracking the
qmm share.

Why the kernel saturates, and why no bit-preserving lever exists:

1. GL_KHR_cooperative_matrix can only build a B matrix via coopMatLoad from
   memory; there is no per-lane register construction. Every dequantized
   f32 weight value therefore does a shared-memory round trip per 16-k
   step, with two workgroup barriers per step. The AGX simdgroup-matrix MMA
   itself sustains 93% of the 2303 GFLOP/s scalar ceiling in isolation
   (receipts/2026-09-11-fma-ceiling); the staging chain, not the MMA, caps
   the composed kernel.
2. Schedule variants are measured dead (receipts/2026-09-10-qmm-prefill-tile):
   TILE_M 64 -2.7% on the dominant shape; STEP_K 32 -40..-50% plus a
   digest-changing Honeykrisp coopmat anomaly. Both rejected there.
3. Every arithmetic restructure (f16 staging, reordered accumulation) moves
   the rule-2 pinned Q4 short and 1K prefill digests; the policy path
   requires proving native's exact intermediate arithmetic, which needs
   native captures of the Metal qmm kernel's dequant order. Not available
   tonight; not attempted.

Path named for a future candidate (a new design, not a schedule tweak):
dequantize into registers and feed simdgroup-matrix FMA directly, outside
GLSL cooperative matrix (explicit AGX simdgroup-matrix SPIR-V, the
fma-ceiling harness already emits such instructions), reproducing native
Metal's accumulation order and qualifying it against the oracle captures.
Alternatively natively-captured intermediates could unlock the f16-staging
route under policy rule 1.

## Verdict

- Decode: bit-preserving packed/batched KV loads land with the sweep,
  paired timings, and unchanged canonical digests on all legs (verdict.json).
- Prefill: receipt-only negative on the kernel; decomposition above.
