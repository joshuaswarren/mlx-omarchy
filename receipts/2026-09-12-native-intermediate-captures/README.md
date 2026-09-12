# Native intermediate captures — 2026-09-12

Fixed-input captures of macOS MLX 0.32.2's own values for (a) the decode
attention path and (b) the quantized matmul prefill path, at the shapes
the canonical legs exercise. Produced on 16m1mbp (Apple M1 Max,
`applegpu_g13s`, macOS 26.6.2) while it still runs macOS; the capture
session is reproducible from `harness/`.

## Provenance

- PyPI `mlx==0.32.2`, `mlx-lm==0.31.3`, venv
  `~/src/mlx-bench-20260901/venv` on 16m1mbp (same venv as the committed
  native baselines). `MLX_DISABLE_COMPILE=1` — see the root-cause
  finding below; it is load-bearing.
- Models pinned exactly as `scripts/bench_matrix.json`:
  `mlx-community/Qwen2.5-0.5B-Instruct-4bit` @ `a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3`,
  `mlx-community/Qwen2.5-0.5B-Instruct-bf16` @ `56d07e766edd7159fbe12ed12d9cf114bf38bf1e`.
- Prompts byte-identical to the committed expansion
  (`~/src/mlx-bench-20260911/prompts.json`); prompt token counts
  asserted 30/262/1053 before any capture.
- Greedy temp 0, seed 0, EOS suppressed, 4 warmup tokens then the
  measured generation — the `bench_decode.py` protocol verbatim, with
  pure observers around `mx.fast.scaled_dot_product_attention` and
  `mx.quantized_matmul` (they convert captured arrays to numpy and
  change no computation).

## Digest oracle reproduction (root-cause finding — read this first)

**Finding (instrument artifact, not a kernel difference):** a harness
that omits `MLX_DISABLE_COMPILE=1` produces a DIFFERENT generated-id
stream on the BF16 long leg (`6ce9a6a22d0fa591`) while the committed
native oracle digest is `407b7624ed1b3b29`. With the env var set —
which the committed bench protocol has mandated all along — the same
harness reproduces every committed native digest exactly. The pin was
never in danger; an unset env var silently routes mlx through its
compile path, which selects different kernels for this shape. Any
future capture harness must set it or its digests are not comparable
to the oracle. Confirmed by a clean unpatched `bench_decode.py` run on
the same day: `407b7624ed1b3b29`.

All six canonical legs reproduce under the fixed capture harness:

| leg (model) | captured digest | committed native oracle |
|---|---|---|
| Q4 short (30/32) | `7fd25a869ff21678` | `7fd25a869ff21678` |
| Q4 long (262/128) | `254d73fd93164b98` | `254d73fd93164b98` |
| Q4 1K ctx (1053/32) | `7da83f06ec9f001d` | `7da83f06ec9f001d` |
| BF16 short (30/32) | `7fc0f968789b1882` | `7fc0f968789b1882` |
| BF16 long (262/128) | `407b7624ed1b3b29` | `407b7624ed1b3b29` |
| BF16 1K ctx (1053/32) | `ff502900d2a179a5` | `ff502900d2a179a5` |

## What was captured (`data/`)

Layout: `data/<model>/<leg>/...` plus `data/attention/` (synthetic
fixed-input captures) and `data/manifest_model.json` (per-call
metadata, sha256-16 of every sdpa/qmm input and output, full generated
id lists).

All f16/bf16 tensors are saved as exact float32 widenings (lossless);
`*_w_packed_u32.npy` files are raw packed weight bits; `*_dequant*`
files are the dequantized operand values the kernel actually consumes.

Per leg (both models):

- `sdpa/step{0,1[,2]}_L<layer>_{q,k,v,out}.npy` — decode attention
  inputs/outputs for all 24 layers at decode steps 0,1 (and 2 on the
  short leg: KV 30, 31, 32 — the sub-block boundaries), bitwise.
- `sdpa/prefill0_L0_*.npy` — the first prefill attention call (layer
  0, causal).
- `qmm/{q,k,v,o,gate,up,down}_{x,y}.npy` — layer-0 prefill quantized
  matmul inputs and native outputs, bitwise, for all seven linears.
- `qmm/{...}_w_packed_u32.npy`, `_scales.npy`, `_biases.npy` — the
  real packed weights, scales and biases for those calls (the exact
  kernel operands; group 64, 4-bit).
- `qmm/{q,down}_w_dequant_f32.npy` (+ gate/up for q4) —
  `mx.dequantize` of those weights = the dequantized operand values.
- `manifest_model.json` additionally records shape/dtype/group/bits
  and checksums for EVERY sdpa and qmm call in every leg (including
  the M=1 qmv decode calls and the 151936-wide lm_head call), so any
  single call can be re-derived and gated later.

Synthetic fixed-input captures (`data/attention/`), seeded
`np.random.default_rng(20260912)`:

- `dispatch/<dtype>_kv<K>_{q,k,v,out}.npy` — real dispatch captures at
  KV ∈ {8,16,30,31,32,33,64,262,263,1023,1024,1053,1054} for
  f16/bf16/f32, q (1,14,1,64), k/v (1,2,KV,64), scale 0.125. On this
  die KV≥1024 routes to `sdpa_vector_2pass` (dispatch condition
  `devc=='s' && KV>=1024`, blocks=128 for N≤8192); KV<1024 routes to
  single-pass `sdpa_vector`. Both paths are therefore captured.
- `exp_inputs_f32.npy` + `fast_exp_outputs.npy` — `metal::fast::exp`
  I/O table on 2M seeded inputs (the transcendental the online-softmax
  chain needs); `fast_exp2_log2e_outputs.npy` tests
  `fast::exp(x) == fast::exp2(x*log2e)` for GLSL equivalence reuse.
- `simd_sum_probe_*` — simd_sum/simd_max on 4096 seeded 32-value
  groups. **Result: none of left-deep, pairwise-tree, or either
  xor-butterfly order matches** — the reduction order is its own thing
  and must be derived (`simd_order_triples.*` banks all 4960
  three-nonzero-lane isolation probes for that derivation).

## Derived accumulation order (from the pinned MLX source)

Source of truth: MLX `1f8e74e3f12f31365464a6867c6579f0e9b29d85`,
`mlx/backend/metal/kernels/sdpa_vector.h`, `quantized.h`,
`steel/gemm/mma.h` (pinned copy under `.work/mlx` after
`scripts/prepare-mlx.sh`).

Decode attention, single-pass `sdpa_vector` (D=V=64, T ∈ {f16,bf16},
U=f32, no mask/sinks — the mlx-lm decode configuration), per query
head h (kv head = h/7, gqa 7):

1. 32 key-streams; stream s ∈ [0,32) owns keys i ≡ s (mod 32),
   ascending. Each (stream, dim-lane d) thread holds dims 2d, 2d+1.
2. q is pre-scaled once: `q[j] = float(scale) * float(q_in[j])`
   (separate rounding, applied to q, NOT to the score).
3. Per owned key i: `score = 0; score += q[2d]*k[2d];
   score += q[2d+1]*k[2d+1]` per lane (2-deep FMA chain), then
   `score = simd_sum(score)` over the 32 dim-lanes. simd_sum's exact
   combine order is captured-but-unsolved (see probes above); it is
   the one remaining order unknown for a bit-exact CPU model.
4. Online update per key: `new_max = max(max_score, score)`;
   `factor = fast::exp(max_score - new_max)`;
   `exp_score = fast::exp(score - new_max)`;
   `max_score = new_max`;
   `sum = sum * factor + exp_score`;
   per owned dim j: `o[j] = o[j]*factor + exp_score*v[j]`
   (single contraction; which operand fuses is settled by the
   bit-exact validation, not by reading).
   NOTE: initial `max_score = Limits<float>::finite_min` =
   `numeric_limits<float>::min()` = the smallest POSITIVE normal
   (1.1754944e-38), not -inf, not -FLT_MAX.
5. After the key loop, cross-stream combine: per-stream
   `factor_s = fast::exp(max_s - new_max)` with
   `new_max = simd_max(max_s)`;
   `denom = simd_sum(sum_s * factor_s)`; for each dim,
   `out[dim] = simd_sum(o_s[dim] * factor_s) / denom`
   (division skipped only when denom == 0), then RNE cast to T.

`sdpa_vector_2pass` (this die, KV ≥ 1024): pass 1 runs the same
per-stream chain over blocks ≡ block_idx (mod 128) and writes
per-(head, block) partials **cast to T (the activation dtype)** plus
f32 `sums`/`maxs`; pass 2 reduces: per-lane max over blocks
lane+32b, `simd_max`; per-lane `Σ_b fast::exp(maxs[lane+32b]-gmax) *
sums[lane+32b]`, `simd_sum`; then `o[dim] += fast::exp(maxs[32b+gid]
- gmax) * partial[32b+gid][dim]` sequentially over b, transpose
combine via `simd_sum`, division by the pass-2 denominator, RNE cast.

Quantized matmul prefill (`qmm_t`, and `qmm_t_splitk` when
`512/(ceil(M/32)*ceil(N/32))` splits K ≥ 2 with K divisible by
`split_k*64`): BM=BN=BK=32, WM=WN=2 (4 simdgroups, 16×16 output tiles
each); weights dequantized to **T (activation dtype)** in threadgroup
memory as `sc[nibble] * int + bias` per `quantized.h:dequantize`
(high nibble uses raw `w & 0xf0` against `s/16` — bit-identical to
`s*nibble` since s/16 is exact, but reproduce the literal form);
per-BK-step `simdgroup_multiply_accumulate` on float 8×8 frags in
kk = 0,8,…,K-8 order; the intra-instruction order of the 8 products
is hardware-defined — `data/qmm/` + the direct-probe captures are the
fixed inputs for deriving it (probe v1: none of 5 candidate orders
matched; element-mapping check needed before the next sweep).
split-K partials are cast to T per partition and summed by the
generic strided reduce.

## Coverage statement

- Decode attention captures cover: all six canonical legs' real decode
  steps (KV 30/31/32, 262/263, 1053/1054) bitwise for both models and
  all 24 layers, plus synthetic boundary shapes KV 8…1054 (f16/bf16/
  f32) including the 1023/1024 dispatch flip between single-pass and
  2pass on this die.
- QMM prefill captures cover: layer-0 x/y/weights for all seven
  linear shapes (K=896: N=128/896/4864; K=4864: N=896) at M=30/262/
  1053 — which spans the split-K selection ladder (e.g. q_proj M=30
  → split_k 14, M=262 → 2, M=1053 → plain qmm) — plus per-call
  checksums for every qmm in every leg, for both models.
- NOT covered yet: per-key score/(m,l) trail dumps and the 2pass
  partials trail from a verbatim dump kernel (three attempts, last
  blocked by an `mx.fast.metal_kernel` scalar-binding defect —
  `mx.array(3)` as a 0-d input reads back as 0 inside the kernel;
  int32-buffer workaround written, validation pending), and the MMA
  intra-instruction order. These are nice-to-have for the BF16 decode
  arm per the current integration plan; the model-run captures above
  are the rule-1 evidence base and are committed now while 16m1mbp
  still runs macOS.

## Harness

`harness/capture_model_internals.py` — the model-run capture (env-gated
observer features for bisection; defaults capture everything).
`harness/probe_attention.py` — exp/simd_sum/dump/dispatch probes.
`harness/probe_2pass.py` — 2pass dump kernels. `harness/probe_mma.py`,
`harness/probe_simd_order.py` — hardware-order derivation probes.
Run with `~/src/mlx-bench-20260901/venv/bin/python` on 16m1mbp;
`--out` selects the data root. Host placeholder `16m1mbp` per the
public-repo discipline; no addresses or service names.

## Bulk arrays are referenced, not committed

The large capture arrays (274 files, 469 MB: the qmm operand and dequantized-weight
dumps and the long-context y matrices) are deliberately absent from this commit. A
public repository would carry them in every clone forever, and their value is as a
verifiable oracle rather than as repository content. `BULK-MANIFEST.json` lists every
one with its size and sha256, so any copy can be proven to be the captured bytes.
The arrays themselves are retained on the Linux M1 at `~/src/native-captures-20260912`
and on the `native-intermediate-captures` branch at `feb631fe`, which was pushed before
this pruning. Everything needed to USE the oracle - the harness, the per-call checksums,
the manifests, the small attention captures, and the fast::exp equivalence data - is
committed here.
