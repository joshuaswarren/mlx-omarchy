# Q4 long-context scaling: receipt-only negative for decode, decomposition for prefill

Receipt for 2026-09-11/12, agent Q4LongCtxScaling. Host: jwm1-linux, Apple M1
(G13G B1), Honeykrisp fork driver. Base measured: origin/main 63c9a8d8 +
branch q4/longctx-scaling (d89c4e9f, shader rewrite) — wheel
mlx_omarchy-0.32.2.dev202609112326+d89c4e9f-cp314-cp314-linux_aarch64.whl,
installed libmlx sha256 71308bd0e8b61620... == wheel member (provenance
recorded in every leg JSON as `match`).

## Verdict up front

**The packed-load/prefetch decode shader is NOT landing.** The falsified
theory, stated so it cannot mislead the next reader:

> The Q4 decode deficit's growth with context (+0.9 ms/token from 30 to
> 1053 keys) coincides with the decode-SDPA kernel's growth, and the
> kernel's incremental KV streaming rate (~10 GB/s vs native's ~26) looks
> like a memory-latency problem. IT IS NOT. Batched loads + packed 32-bit
> pair loads — a 4x increase in loads in flight — measured ZERO on
> production legs at 1K (-0.2%) and NEUTRAL-TO-WORSE on the isolated
> kernel (-27% at k=511, -17% at k=1023). The ~10 GB/s figure does not
> mean load timing is the constraint; do not re-run load-timing or
> prefetch levers on this kernel.

What remains true and is the durable lead: the kernel's cost is ~54-69
cycles PER KEY, serial, in every stream — one subgroupAdd + exp + fma
update per key in an online-softmax chain whose each step depends on the
previous. Native pays ~20 ns/key for the same algorithm (~2.7x less).
Closing that requires breaking the serial per-key dependency (chunked /
blocked online softmax with per-chunk local maxima), which reorders
accumulation and MOVES the rule-2 pinned Q4 short and 1K-context digests.
That path needs native intermediate captures under policy rule 1 and is a
new design, not a tweak. Nothing of it is in this branch.

## What was measured (all artifacts in this directory)

1. **Bit-identity gate: PASS, 17/17.** `sdpa_equiv_sweep.py` /
   `equiv.json`: packed (aux_size==1) vs frozen scalar
   (MLX_OMARCHY_SDPA_DECODE_SCALAR=1) uint16-exact at k = 1, 2, 31, 32,
   33, 61, 262, 320, 511, 1023, 1024, 1025, 1053, 1084, 1500, 2048, 2049
   — covering one-pass, 64-block two-pass, 128-block two-pass, strided
   capacity-4096 cache slices. SPIR-V opcode histogram vs the released
   kernel: additions only in loads/control-flow/index math.
2. **Production legs, paired, digest-gated: 12/12 canonical pins hold on
   BOTH arms** (`leg-*.json`; short 7fd25a869ff21678, 262 4cc08910089477fd,
   1K 7da83f06ec9f001d; provenance `match` in every run). Medians:
   packed 111.42/108.01/96.03 tok/s vs scalar 110.20/107.35/96.19 —
   +1.1% / +0.6% / **-0.2%**. The 1K leg, the leg the whole lever aimed
   at, shows ZERO gain.
3. **Kernel-isolated paired chain micro** (`sdpa_chain.json`,
   harness mirroring receipts/2026-09-10-decode-attribution:
   inputs built once and memoized, serial 24-call chain, per-rep
   perturbation, 25 reps): scalar 0.671/0.752/1.018/2.082/1.956/1.629 ms
   vs packed 0.688/0.756/1.294/2.433/2.162/1.606 ms at
   k=30/262/511/1023/1053/1084. Packed is up to 27% WORSE mid-range.
   The batch arrays cost registers; the driver's own scheduling was
   already covering the loads.
4. Prefill qmm at the missing rung (`qmm_m262_probe.py` /
   `qmm_m262.json`, quiet machine load 0.50): **233.3 ms/model of the
   270 ms fork leg = 86% share** at m=262 (gate_up 892.7 GFLOP/s).

## Prefill: receipt-only negative, three-rung decomposition

Fork qmm share and rate vs native (m=1053 and m=30 from
receipts/2026-09-10-qmm-prefill-tile; m=262 from this receipt):

| rung | fork leg | qmm ms/model | qmm share | fork qmm GFLOP/s | native leg |
|------|----------|--------------|-----------|------------------|------------|
| m=30   | 90.6 ms | ~40-45 | ~45% | ~480 (latency-bound) | 102 ms |
| m=262  | 270 ms  | 233.3  | 86%  | ~760                 | 216 ms |
| m=1053 | 946 ms  | 725.7  | 77%  | ~982                 | 572 ms |

Named mechanism: the growing prefill deficit is not anything
context-specific (mask, KV traffic, recomputation are all under 16% of
the leg). It is the qmm coopmat kernel asymptoting at ~1.0 TFLOP/s
against native's implied ~1.66, times a qmm share that grows linearly
with context. Deficit: -11 ms (m=30, fork ahead) -> +54 (262) -> +374
(1053). Why no bit-preserving lever exists: GL_KHR_cooperative_matrix
can only build a B matrix via coopMatLoad from memory — every
dequantized f32 weight does a shared-memory round trip with two barriers
per 16-k step; TILE_M 64 / STEP_K 32 are measured dead
(receipts/2026-09-10-qmm-prefill-tile); every arithmetic restructure
moves the rule-2 pinned digests. Future path: native-order kernel that
dequantizes to registers and feeds simdgroup-matrix FMA directly outside
GLSL coopmat (the fma-ceiling harness already emits such SPIR-V),
qualified against the oracle captures.

## Branch status

q4/longctx-scaling (d89c4e9f) carries the shader rewrite and is the
measured artifact for this receipt. It is DO NOT MERGE: no measured gain.
FusedDecodeAdjudicate should continue from MAIN's sdpa_decode_native.comp
(not this branch's). The sweep harness (`sdpa_equiv_sweep.py`) and chain
harness (`sdpa_chain_micro.py`) are generic and stand alone from the
shader — reuse them.

## Verdict

- Decode: receipt-only negative. Bit-identity evidence intact so nobody
  re-runs packed loads or prefetch on SdpaDecodeNativeF16.
- Prefill: receipt-only negative + three-rung decomposition above.
- Evidence integrity: summarize.py gate PASS (bit-identity 17/17, all 12
  leg digest pins hold both arms).
