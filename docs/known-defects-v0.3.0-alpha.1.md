# Known silent wrong-value defects in v0.3.0-alpha.1

Release [v0.3.0-alpha.1](https://github.com/joshuaswarren/mlx-omarchy/releases/tag/v0.3.0-alpha.1) was cut on 2026-09-01. The wheels on its release page ship the defects below. Each one returns confidently wrong numbers and raises nothing. A user cannot detect them by watching for errors.

Upstream MLX's own test suites exposed them on 2026-09-02. The receipt records twelve distinct root causes, numbered W1 to W12. It also records a related cumulative-scan case and one unconfirmed suspicion. Per-case evidence and exact commands sit in [receipts/2026-09-02-upstream-suite-coverage.md](../receipts/2026-09-02-upstream-suite-coverage.md). Fixes are in progress on `feat/vulkan-primitives`.

This project's contract is to refuse by name rather than return a wrong number. These defects break that contract. That is why they outrank every coverage gap.

## Does this hit me?

- Transformer inference: `mx.fast.scaled_dot_product_attention` is wrong whenever the key and value head count differs from the query head count. That is grouped-query attention, which current Llama, Qwen, and Mistral models all use. `nn.gelu_approx` produced values up to 1.47e13 under test conditions.
- Training: the `mx.fast.layer_norm` weight gradient is wrong when the normalized dimension exceeds 512 columns, returning NaN at 1024 and worse above.
- NaN-handling code: first-axis reductions drop NaN. `cummax` and `cummin` stop carrying NaN. `array_equal(equal_nan=True)` returns False for equal arrays.
- Integer multi-axis sums: `mx.sum` over three or more axes of a rank-4 int array returns wrong totals with sign flips.
- Indexing, filling, and math: negative-axis `take` fills zeros. Dtype-converting `full_like` fills one element. `mx.sin` saturates at about 1e9. `log10` is one ulp low everywhere.

If none of these matches your use, this page names no defect for it. Everything else fails loudly with a named `[omarchy]` error.

## Attention and transformer inference

### `mx.fast.scaled_dot_product_attention` with grouped-query attention

Corrected on 2026-09-02. This was first recorded as a head-dimension defect, because upstream's `test_sdpa_head_dim_72` and `test_sdpa_head_dim_96` were the failing tests. Root-causing it found a broader and more serious cause: the defect is **grouped-query attention**, not head dimension. When the key and value heads are fewer than the query heads, the backend produces a 5-D result and then installs those 5-D strides on a 4-D output array. Head dim 64 and 128 appeared safe only because the tests that exercise them use equal head counts.

This is worse than the original description. Grouped-query attention is standard in current language models - Llama, Qwen, and Mistral families all use it - so this affects mainstream inference, not an unusual head size. Errors reached 8.5e5, `inf`, and 2.30e+31. Every mask form is affected: additive, bool, causal, and None. All of float16, bfloat16, and float32 are affected. One failing shape is B=1, D=72, 8 query heads, 2 key heads. Upstream Metal passes the same tests.

Do not trust SDPA in this release for any model whose key and value head count differs from its query head count. Equal-head-count attention computes correctly at every head dim tested, including 72 and 96.

### The fast SDPA vector path disagrees with its own decomposition

`mx.allclose(ref, out, atol=1e-4, rtol=1e-4)` fails for several masks. The case is `test_fast_sdpa.py::test_sdpa_vector` at line 314. `ref` is `mlx_primitives_sdpa` on the same inputs. The fast path contradicts the backend's own primitive decomposition. The wrong value is backend-internal.

The receipt records one more suspicion in this family. `test_sdpa_fully_masked` expects a sentinel such as `-inf`; our result does not match. No standalone repro exists yet. The receipt holds it as a follow-up, not a confirmed defect.

### `nn.gelu_approx` produced values up to 1.47e13 under test conditions

Upstream's `test_nn.py::TestLayers::test_gelu` measured these gaps at lines 1086 to 1087:

- `max|gelu - gelu_approx| = 1.46602e+13`, against a limit of 0.0005.
- `max|gelu - gelu_fast_approx| = 5.87973`, against a limit of 0.025.

Both values are far outside the leg shape of any GELU approximation. The same primitives pass in a fresh process: 0.000470 and 0.0203. Under pytest process context they return values up to 1.47e13. The receipt classifies this as state-dependent. It names a process-order or allocator-reuse race, not yet pinned. Two of two in-suite attempts were wrong.

A fresh-process smoke test does not clear this defect. Do not trust `gelu_approx` or `gelu_fast_approx` in this release.

## Training and gradients

### `mx.fast.layer_norm`'s weight gradient is wrong above 512 columns

Upstream's `test_fast.py::TestFast::test_layer_norm_grad` pins this at line 720. The fast gradient must match the composed gradient within 5e-5. The `mx.fast.layer_norm` VJP path returned `nan`.

Root-caused on 2026-09-02, with a scope worth stating precisely: the weight-gradient kernel did not reset its accumulator per 256-column tile, so it summed the wrong columns. Rows of 512 columns or fewer are correct. At 1024 columns the result is `nan`. At 8192 columns the relative error reached 4.94. That threshold is why the kernel's own unit tests passed - they used 32 columns.

Do not train through `mx.fast.layer_norm` in this release when the normalized dimension exceeds 512. A silent NaN gradient reaches every parameter it touches, and a loss curve keeps looking healthy while the model stops learning.

### One vjp path is one ulp off

Upstream's `test_autograd.py::TestAutograd::test_eval_in_grad` got `vjp = 12.000000953674316`. The exact value is `12.0`. This is one ulp of float32 accumulation. Upstream Metal is exact here. Severity is low. The receipt groups it with the `log10` case as a precision defect.

## NaN-handling code

### Float reductions over the first axis drop NaN

`mx.max(x, axis=0)` over an array containing NaN returns the non-NaN maximum. numpy and Metal return NaN. Repro, copied from the receipt:

```python
import mlx.core as mx, numpy as np
x = mx.array(np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]], np.float32))
print(np.array(mx.max(x, axis=0)))   # got [4. 5. 6.]   want [ 4. nan  6.]
```

Float32 and float16 are affected. Axis 0, the non-suffix axis, drops NaN. The suffix axis and whole-array reductions propagate NaN correctly. Integer reductions are unaffected.

Do not trust float reductions over a non-suffix axis of NaN-containing data. The returned values look plausible. They stay inside the legal output range.

### `cummax` and `cummin` stop carrying NaN after the first one

Once a NaN enters a running max, the running max should stay NaN. It does not. Repro from the receipt:

```python
import mlx.core as mx, numpy as np
a = mx.array([1.0, 3.0, float('nan'), 5.0, 4.0])
np.array(mx.cummax(a, reverse=False, inclusive=True))
# got array([ 1.,  3., nan,  5.,  5.])   want [ 1.,  3., nan, nan, nan]
```

The receipt classifies this as the cumulative-scan analogue of the reduction defect above. It sits in the Scan family, not the Sum family. Severity is high.

### `array_equal(..., equal_nan=True)` returns False for equal arrays

Upstream builds an `Equal` primitive with the `equal_nan` flag embedded. The site is `mlx/ops.cpp:2015-2025`. Our `Equal` kernel ignores the flag. NaN compares unequal to NaN, and the trailing `all()` collapses to `False`. Nothing raises.

```python
import mlx.core as mx
a = mx.array([0.0, 1.0, float('nan')])
mx.array_equal(a, a, equal_nan=True).item()   # got False   want True
```

Do not trust any NaN-tolerant equality check in this release. This function is also the comparison harness upstream's suite uses for float NaN cases.

## Integer multi-axis sums

### `mx.sum` over three or more axes of a rank-4 int array returns wrong totals

Deterministic repro from the receipt:

```python
import mlx.core as mx, numpy as np
np.random.seed(0)
x = mx.array((np.random.randn(65,65,1,65) * 128).astype(np.int32))
z = np.asarray(mx.sum(x, axis=(0,1,3))); w = np.sum(np.array(x), axis=(0,1,3))
print(z, w)
# got [-13684]  want [48876]    (max-diff 18 280, sign flips)
```

On the tested shape, single-axis sums are correct. The two-axis pairs (0,1), (0,3), and (1,3) are correct. Three-axis sums, full reduces, and `axis=None` over the same subset are silently wrong. That is 48 of 60 combinations. Rank-4 int inputs are the broken class.

Rank-5 shapes refuse cleanly with `Sum rank above 4`. Do not trust int multi-axis sums over three or more axes in this release. Errors reach 1e4 in magnitude and flip sign.

## Indexing and array construction

### `take` with a negative axis index fills zeros

Negative indices along an axis are treated as out of bounds. The gather's bounds check then writes zeros. From the receipt: `take(int32 [[1,2],[3,4]], array(-1), 0)` returns `[0, 0]` instead of `[3, 4]`.

A scalar `-1` at axis 0 returns zeros. `array([-1])` returns `[[0, 0]]`. `array([0, -1])` zeroes the second row. Floats and ints are both affected.

Flat `take` with negative indices and no axis wraps correctly. For example, `take(a, array({1, -1}))` returns `[2, 4]`. Axis values other than 0 refuse by name.

Do not use negative indices on an axis-based `take` in this release. You get silent zero rows where wrap-around indexing was requested.

### `full_like` with an array scalar and a dtype conversion fills only the first element

Repro from the receipt:

```python
import mlx.core as mx, numpy as np
base = mx.array(np.array([1,2,3], np.int16))
np.array(mx.full_like(base, mx.array(7.5), dtype=mx.float16))
# got array([7.5, 0. , 0. ], dtype=float16)   want [7.5, 7.5, 7.5]
```

The first element converts correctly, and the rest are zero. float16 and int32 conversions are both affected.

Same-dtype `full_like` fills correctly. So do `mx.full` with an array scalar and any Python scalar value.

Do not trust `full_like` with an array scalar plus a dtype conversion in this release.

## Large-argument trig and log precision

### `mx.sin` saturates to plus or minus 1 for arguments of about 1e9 and above

Repro from the receipt:

```python
big = np.array([1e8, 1e9, 1e10, 1e20, 1e30], np.float32)
for o, w, v in zip(np.array(mx.sin(mx.array(big))), np.sin(big), big):
    print(f"sin({v:g}): got {o:.7f}  numpy {w:.7f}  diff {abs(o-w):.3g}")
# sin(1e+08): got 0.9308981  numpy 0.9316390  diff 0.000741
# sin(1e+09): got 1.0000000  numpy 0.5458434  diff 0.454
# sin(1e+10): got -1.0000000 numpy -0.4875060  diff 0.512
# sin(1e+20): got -1.0000000 numpy  0.6565767  diff 1.66
# sin(1e+30): got -1.0000000 numpy -0.7911634  diff 0.209
```

The cause is the range-reduction stage. The outputs still lie in [-1, 1], so nothing looks wrong locally. Any periodic computation on arguments of magnitude 1e9 or more is computing noise. At 1e8 the value is already off by 7e-4. The suite's tolerance there is 1e-6. Upstream Metal handles all these values.

### `log10` is about one ulp low on every value

`mx.log10(1000.0)` returns `2.999999761581421`, not `3.0`. The nearest float32 to `log10(1000)` is exactly `3.0`. `log10(100)` returns `1.9999999`, and `log10(1e6)` returns `5.9999995`. The cause is float32 scaling by `1/ln(10)`. `mx.log` and `mx.log2` are correctly rounded against numpy. So `mx.log(x) / mx.log(10.0)` is a correct substitute. Severity is low. It surfaces only where code pins exact equality.

## What to do

Treat every operation on this page as untrusted in v0.3.0-alpha.1. None of them raises, so error handling and exception tests pass anyway.

Some safe paths exist, and this page names them: attention with equal query and key head counts, flat negative `take`, same-dtype `full_like`, one- and two-axis integer sums, suffix-axis and whole-array NaN reductions, and `mx.log` or `mx.log2`.

Fixes are in progress on `feat/vulkan-primitives`. Prefer a later release, or a fresh wheel from that branch once the fixes land.

Named `[omarchy] ... is not implemented` errors remain the honest failure mode. The defects on this page are dangerous because they do not fail that way.
