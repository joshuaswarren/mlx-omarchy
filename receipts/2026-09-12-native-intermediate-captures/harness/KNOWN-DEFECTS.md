# Known defects and strong negatives from the capture session

Findings that cost time here, written down so the next person does not
repeat them. All observed on 16m1mbp, macOS 26.6.2, PyPI mlx==0.32.2,
venv `~/src/mlx-bench-20260901/venv`.

## 1. `mx.fast.metal_kernel` 0-d scalar input binds as zero

`mx.fast.metal_kernel(..., input_names=["i"])` called with
`inputs=[mx.array(3)]` (0-d int64 array) reads `i` as 0 inside the
kernel. Repro (same venv):

```python
import mlx.core as mx, numpy as np
src = "if (i % 32 == 0) { o[thread_position_in_grid.x] = 1.0f*i + simd_sum(1.0f); }"
k = mx.fast.metal_kernel(name="repro", input_names=["i"],
                         output_names=["o"], source=src,
                         ensure_row_contiguous=False)
for v in (0, 3, 32):
    r, = k(inputs=[mx.array(v)], grid=(64,1,1), threadgroup=(32,1,1),
           output_shapes=[(64,)], output_dtypes=[mx.float32])
    print(v, np.array(r)[0])   # observed: 32, 32 (expected 35), 64
```

i=0 and i=32 read correctly, i=3 reads as 0 — small non-multiples of
the element size lose the value, so the failure looks like a
conditional that "never fires", not a binding error. Workaround: pass
int32 values through a real buffer (`mx.array(np.full(n, v,
np.int32))`) and index it in-kernel. Never pass 0-d scalars.

Related: the runtime kernel builder caps threadgroup sizes (requested
1024 launched with `threads_per_threadgroup` reporting 24; 512 and
below launch as requested). Any kernel structured like upstream
`sdpa_vector` (32 simdgroups x 32 lanes = 1024 threads, cross-simdgroup
threadgroup exchange) cannot be recompiled through the public API as
is — decompose into 32-thread launches with host-chained state, or
build MLX from source.

## 2. simd_sum combine order: textbook candidates are all wrong

4096 seeded 32-value groups + all 4960 three-nonzero-lane isolation
probes (`simd_order_triples.*`): left-deep sequential, pairwise tree
by low bits, and both xor-butterfly orderings match at chance rates
(≤128/4096). Do not model `simd_sum` with any of these; derive the
tree from the triple data or treat the score as a captured oracle
value. This blocks naive bit-exact CPU models of any kernel that
reduces across a simdgroup (`sdpa_vector` scores, final combines).

## 3. MMA 8-deep intra-instruction order: five candidates rejected

`simdgroup_multiply_accumulate(float 8x8)` on M1 Max, 2048 trials x 64
elements: sequential-fma, sequential-mul-add, product-tree+/-C, and
fma-pair candidates all match at chance rates (best 4096/131072 =
1/32, exactly coincidence level) — `qmm/mma_order.json`. Before the
next sweep, verify the thread-element placement mapping
(`BaseMMAFrag::get_coord` assumption) with a one-hot A/B probe; an
incorrect mapping reproduces exactly this all-candidates-fail
signature.

## 4. Omitting MLX_DISABLE_COMPILE=1 silently changes digests

Documented in README; repeated here because it is the class of
instrument artifact that misdirects a whole day: a capture or bench
harness without `MLX_DISABLE_COMPILE=1` produced `6ce9a6a22d0fa591`
on BF16 long while the committed native oracle is `407b7624ed1b3b29`.
Same machine, same pinned snapshot, same prompt, same day. Clean
unpatched bench with the env var set reproduced the oracle. Always set
it; treat any digest comparison across harnesses without it as void.
