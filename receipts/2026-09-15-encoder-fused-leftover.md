# Encoder fused leftover: layer_norm / silu / GLU / softmax in standalone kernels, pin-exact (2026-09-15)

Verdict: the per-layer leftover elementwise/norm chains now run as standalone
`mx.fast.metal_kernel` dispatches. `encoder_hidden` is **bit-identical to the
pin** `38c73261f29230276ed76f1fc017b76b024156d79218bd5f1347fdc7e7d43ec7`, the
ANE E2E stays **104/104 with transcript sha `db501a8c`**, the frozen contract
bounds pass, and the token stream is byte-identical to the landed baseline.
Encoder stage wall drops **19386.0 → 19080.0 ms** E2E (19484 → 19194 ms on the
standalone-leg basis) and **vk compute dispatches drop 6690 → 5202 (−22.2%)**.

## Change

`overlay/tools/coreml/vulkan_encoder.py` (branch tip staged copy sha256
`6c175adf1bf5a39717496924b1ef4579bb1f8eb5ff42064b081b0caf037a6f0a`, on jwm1 at
`/var/tmp/ParakeetE2EFusedLeftover-stage/vulkan_encoder.py`), on top of the
landed batched-linear runner `0c68c1ff`:

1. **layer_norm** (120 per run): `_ln_cast_kernel` + the two unchanged
   `mx.mean` ReduceF32 dispatches + `_ln_sq_kernel` + `_ln_tail_kernel` —
   the cast/center/square/rsqrt/affine elementwise chain around the
   reductions fuses 11 dispatches into 5.
2. **silu** (72): `_silu_kernel` — sigmoid + mul + cast chain in one
   dispatch, fp32 math, single boundary rounding.
3. **conv-module GLU** (24 sigmoid + 24 mul): `_glu_kernel` — one dispatch
   for `a * sigmoid(b)` on the two channel-axis split halves.
4. **softmax** (24): `_sm_exp_kernel` + `_sm_div_kernel` around the unchanged
   reductions; the row max becomes ReduceF16 over the fp16 input (max is
   order-insensitive and exact, so widening its fp16 result is the same fp32
   value the cast-then-ReduceF32 arm produced).

`_index_fusions` pattern-matches the GLU (sigmoid consumed by exactly one mul
whose other operand is a sibling channel-axis split half); the sigmoid
statement never executes and its split sibling's release point moves to the
mul. Non-pinned envelope forms (LN not `(N,1024)` with axes `[-1]`, softmax
not `(1,8,375,375)` axis `-1`) raise instead of silently falling back.

## Bit-identity: three hazards found by A/B probes, not by taste

Standalone A/B on jwm1 (`receipts/2026-09-15-encoder-fused-leftover/
fused_ab_smoke.py`, output `smoke-output.txt` sha256 `27e98112…`):
**ALL-IDENTICAL** across silu (4096-wide), GLU on real split views, layer_norm
at 1-row and 375-row envelopes, and softmax with all-masked `-inf` rows
(NaN agreement checked separately).

1. **Parameter-name macro collision.** The MSL→GLSL translator macros every
   buffer name (`#define x _b0.data`); an input named `x` eats the `.x`
   swizzle of `thread_position_in_grid` and the shader fails to compile
   (`'_b0' : unknown swizzle selection`). Kernel buffers are named
   `src`/`dst`/`mu`/`va`/`ga`/`be`/`ep`/`lhs`/`rhs`/`rmax`/`rsum`/`expd`.
2. **fp16 locals do not round.** `half gate = half(...)` keeps fp32
   precision in the following fp32 arithmetic, so the GLU gate skipped the
   fp16 rounding the two-dispatch chain performed (26% of outputs off by one
   fp16 ulp). The gate rounds through
   `unpackHalf2x16(packHalf2x16(vec2(s, 0.0))).x` — the same
   integer-packing recipe fused_chain.comp records for M1 equality.
3. **Driver FMA contraction.** The LN tail's `t * rstd * gamma + beta`
   contracted mul+add across statements (42/384000 one-ulp diffs in every
   plain variant). Marking the product `precise float` restores the
   dispatch-chain rounding.

Probes (`fused_probe_rounding.py` + the stage-dir probe scripts) also
confirmed: `inversesqrt` ≡ `mx.rsqrt`, `1.0/(1.0+exp(-x))` ≡ `mx.sigmoid`,
plain fp32 mul and the final `half()` cast are bit-exact against the mx
dispatches, and fp16 max ≡ fp32 max of the widened input.

## Real run (jwm1-linux, islands ABC, same command as the batched-linear land)

Lock `/tmp/m1-gpu.lock`, `flock -w 900`, never steal, never unlink; smoke
A/B and both legs ran under the same lock discipline.

| quantity | value |
| --- | --- |
| status | **match** |
| emissions actual / native | 104 / 104, matching_prefix 104 |
| token stream | byte-identical to the landed baseline `token_ids.json` |
| transcript_match | true, sha256 `db501a8c080380ea027ffa50a4b4956c39df77cb692c4fb78e556311a11a0790` |
| encoder bounds | pass (`encoder_bounds_pass` true), mel bit-exact |
| nan / inf check | pass |
| encoder_hidden sha256 | **`38c73261…` (pin, unchanged)** in the leg and the E2E |
| ANE | islands ABC, 72 submits, 72 worker starts, 0 timeouts, exec 5034 ms |
| encoder wall (E2E stage) | **19080.0 ms**, was 19386.0 ms (−306.0 ms, −1.6%) |
| encoder wall (standalone leg) | **19194.3 ms**, was 19484 ms |
| vk compute dispatches | **5202**, was 6690 (−1488, −22.2%) |
| MIL gpu ops | 1254, was 1278 (24 sigmoid statements never execute) |

Identical encoder bytes ⇒ identical decode: the decode stages measured
2351.8 ms decoder + 525.3 ms joint, and the pipeline total is not comparable
across runs (cold mel in this run, same caveat as the batched-linear
receipt).

## Identity

- fused runner sha256 `6c175adf1bf5a39717496924b1ef4579bb1f8eb5ff42064b081b0caf037a6f0a`
  (= this commit's `overlay/tools/coreml/vulkan_encoder.py` = the derivation copy)
- baseline runner `0c68c1ff…` (landed batched linear, f735a675)
- e2e runner `/var/tmp/ParakeetE2E/parakeet_e2e.py` `16237078…` (unchanged)
- MIL `ac8e9526…`, audio `30885601…`, worker `762dd1de…`, libane `1ab9d95d…`
- wheel `0.32.2.dev202609141626+05015a76` (unchanged); no backend code changed —
  every fusion lives in the runner via `mx.fast.metal_kernel`
- host jwm1-linux, aarch64, Python 3.14.7 (venv-agxgen + MelFrontendPerf venv-cache overlay)

## Artifacts

`receipts/2026-09-15-encoder-fused-leftover/`:
`vulkan_encoder.py` (`6c175adf…`),
`run-report-fused.json` (`b6f168e7…`),
`e2e-report.json` (`3a0a70ca…`),
`fused_ab_smoke.py` (`bba2db03…`) + `smoke-output.txt` (`27e98112…`),
`fused_probe_rounding.py`.

## Not claimed

- The wall win is real but small (−1.6%): the leftover is dominated by the
  ConvF32/GEMM convs, island submits and inter-submit gaps, not by the fused
  elementwise chains; the dispatch reduction (−22%) is the durable part.
- No decode, no other fixture, no macOS, no backend-shader change.
- `63c1d3cf` not merged.

resolved_model: this session (`zai/glm-5.3-flash`).
