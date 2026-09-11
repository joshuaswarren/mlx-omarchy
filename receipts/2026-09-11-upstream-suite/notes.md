# Upstream suite re-qualification — 2026-09-11

Dated full-suite snapshot at source commit `2a9add427cba2474e2c41b35105e5c7bf4402885`
(origin/main), pinned upstream MLX 0.32.2 `1f8e74e3` (archive sha256
`cb988a5b…2acc301`). Supersedes the 2026-09-06 Python snapshot
(`receipts/upstream-suite-2026-09-06-py4/`, source `ef188d58`) in README's
feature-parity table.

## What ran

- Protocol: `tools/run-upstream-suite.sh` from `2a9add42`, unedited, per-file
  watchdog 900 s (defaults), `DEVICE=gpu`, `MLX_ENABLE_TF32=0`,
  `MLX_OMARCHY_ALLOW_NON_APPLE=1`, `ulimit -c 0`. Identical invocation and
  parser to the 2026-09-06 Python receipt; the only runner change between the
  two commits is a `PYTEST_VENV` override knob (no semantic drift).
  The runner executed in two invocations at a phase boundary (C++ then Python)
  for GPU-serialization reasons; artifacts are merged here as the runner wrote
  them.
- Environment: Apple M1 (G13G B1), aarch64 omarchy Linux, kernel 7.1.6, Mesa
  Honeykrisp fork `26.3.0.devel.hk6f6afc8-1` (Vulkan 1.4.359, device
  "Apple M1"), Python 3.14.7 (cp314). Host referred to as `<m1-host>`.
- Wheel: `mlx_omarchy-0.32.2.dev202609112117+2a9add42-cp314-cp314-linux_aarch64.whl`,
  sha256 `9dc042d383b8453e055636bea9fc39b521da3a3f82438cfa2d1816a0471b8ef8`
  (see `wheel-sha256.txt`); provenance verified against the source commit
  (`python-provenance.json`, `SOURCE-VERIFIED`).

## Headline counts (comparable to the published table)

| Measure | 2026-09-06 (`ef188d58`) | 2026-09-11 (`2a9add42`) | delta |
|---|---|---|---|
| Upstream C++ cases passing on the GPU device | 251 / 251 | 251 / 251 | 0 |
| Upstream Python cases passing on the GPU device | 10,767 / 11,437 | 11,483 / 11,847 | +716 passed |
| Python failures: named refusals | 471 | 342 | −129 |
| Python failures: assertion failures (wrong value) | 149 | 13 | −136 |
| Python failures: other errors | 50 | 9 | −41 |
| Watchdog timeouts (Vulkan timeline counter) | 40 | 0 | −40 |

Every executed C++ case passes; all 20 upstream C++ TUs exit rc=0
(`cpp/summary.tsv`). Per-TU executed counts are identical to the deterministic
upstream registration (251 cases; e.g. `blas_tests` registers exactly one test
case, `ops_tests` 88 — same distribution as the 2026-09-02 x86_64 run), so the
251 denominator is directly comparable.

### Denominator honesty (the +410 executed cases)

Executed Python cases grew 11,437 → 11,847. The suite is pinned, but case
execution is not: upstream tests branch on `mx.default_device() == mx.gpu` and
sweep different dtype lists per branch, and one quantization sweep test is
guarded by `@unittest.skipIf("CI" in os.environ)` — the 2026-09-06
environment had `CI` set (its `test_quantized.py` shows exactly one skip), so
`test_qmm_non_transposed` never executed there. This run has no `CI` variable,
so the GPU-branch sweeps ran. Counts here are therefore compared per category
and per case (`delta-vs-2026-09-06.csv`), never as raw totals. The denominator
movement is a protocol artifact, not a parity gain.

## Ranked root-cause clusters (machine-readable: `root-cause-clusters.csv`)

### Named refusals — 342 cases, 3 clusters

| n | Refusal | Primitive and signature |
|---|---|---|
| 258 | `QuantizedMatmul weight layout` | `mx.quantized_matmul` / `mx.gather_qmm`, float16 and bfloat16 activations, 2/3/4/5/6/8-bit packed words, group sizes 32/64/128, transposed weight layouts beyond the supported packed-word form. Identical count at 2026-09-06. |
| 83 | `Quantize input dtype` | `mx.quantize` with bfloat16 input (packed-word output is uint32), group sizes 32/64/128, bits 2/3/4/5/6/8, weights `[K,N]` up to `[33000,128]`. Standing refusal — the guard is byte-identical at both snapshot commits; it only appears now because the sweep test was skipped in the prior environment (see denominator section). |
| 1 | `QuantizedMatmul rank` | `test_qvm_splitk`: rank-3 `mx.qvm` input not accepted. |

### Assertion failures (wrong value) — 13 cases, 7 clusters, ranked by coverage

| n | Root cause | Primitive and shape/dtype signature |
|---|---|---|
| 4 | f16 SDPA with causal mask on ragged lengths stores/uses narrow-type scores: max err 0.077-0.125 vs 3e-4 tol; one NaN at head_dim=128 with transposed V. | `mx.fast.scaled_dot_product_attention`, float16, B=1, qsl=127, ksl=65, n_q_heads=32, n_kv_heads=8 (GQA), head_dim=64 and 128, mask=causal, transpose=False and True. Same file held all 40 prior watchdog cases; it now completes in 23.7 s. |
| 2 | `mx.compile` composite diverges from eager: compiled `value_and_grad` + optimizer step vs eager step, max_abs_diff 1.1e-2 on `nn.Linear(10,10)` weights (reproduced standalone). | `mx.compile` + `nn.value_and_grad` + `optim.apply_gradients`, float32; and dynamic-dimension compile vs eager (`test_compile_dynamic_dims`). |
| 1 | Subnormal float32 → bool cast flushes to False; float16 path is correct (6e-08 → True). f32 1e-45 → False (must be True). Also fires under `mx.compile`. | `array.astype(bool)`, float32 subnormals (< 1.18e-38); bfloat16 subnormals in the compile variant. Regressed since 2026-09-06. |
| 1 | `mx.convolve` wrong values, state-dependent: fails inside the full suite, passes standalone with one subtest still flagged. | `mx.convolve` 1-D, float32, M=24, N=4, mode='same'. Regressed since 2026-09-06. |
| 1 | `mx.sort` wrong order on zero-strided broadcast input. | `mx.sort` on `broadcast_to([1,0,2,1,3,0,4,0], (16,8))`, axis 0/1, int64. Regressed since 2026-09-06. |
| 1 | `MaxPool1d` wrong values with padding. | `nn.MaxPool1d(kernel_size=2, stride=2, padding=1)`, float32 `[2,3,4]`-class inputs. Regressed since 2026-09-06. |
| 1 | `gather_qmm` sorted path wrong value (not a refusal): 0.29248 vs tol 1e-3. | `mx.gather_qmm`, float16, L=133, K=512, D=555, E=4, I=2, transpose=False, mode=affine. |

### Other errors — 9 cases, 3 clusters

| n | Root cause | Notes |
|---|---|---|
| 6 | `UnboundLocalError: custom_kernel` — upstream harness artifact in the pinned suite (the name is undefined in 0.32.2's test helper). Identical 6 cases at 2026-09-06. Not a backend defect. |
| 2 | Compiled-tape bfloat16 refusal (`bf16 fragments corrupt nondeterministically`): `test_inf_constant`, `test_compile_nonfinite_constants`. Explicit refusal, tracked in docs/known-defects.md. |
| 1 | `Sin` argument-magnitude refusal: `test_sin` extreme-magnitude subtest. Explicit range refusal. |

85 cases fixed (fail 2026-09-06 → pass now), 3 changed failure kind, 5 flagged
regression. The kind changes: `test_compile_dynamic_dims` moved from a named
refusal to a wrong-value assert and `test_compiled_subnormal_bool_cast` from an
error to a wrong-value assert — the compile path stopped refusing and now
produces wrong values in those two spots (clusters 2 and 3 above); `test_sdpa`
remains wrong-valued. Verdicts on the 5 flagged regressions, each verified:

1. `test_ops.py::test_subnormal_bool_cast` — REAL wrong-value regression (cluster 3 above).
2. `test_conv.py::test_numpy_conv` — REAL wrong-value regression, state-dependent (cluster 4).
3. `test_optimizers.py::test_compiled_optimizer` — REAL wrong-value regression (cluster 2).
4. `test_export_import.py::test_export_custom_metal_kernel_without_evaluation` — PROVEN TEST-SIDE ARTIFACT, not a backend defect. The test asserts that exporting a Metal custom kernel without evaluation raises `RuntimeError("No Metal back-end")` when `mx.metal.is_available()` is false. On this build `mx.metal.is_available()` is False, but the backend ships a Vulkan custom-kernel implementation (`patches/mlx-omarchy-metal-kernel.patch`), so the exporter legitimately accepts the graph and the expected refusal never comes. The upstream expectation ("no Metal back-end ⇒ custom kernels impossible") is stale for this backend. The pinned upstream suite is the oracle and is left untouched; the case is excluded from the defect count by this note.
5. `test_quantized.py::test_qmm_non_transposed` — PROTOCOL ARTIFACT, not a regression: the test was skipped by the `CI` environment flag at 2026-09-06 (see denominator section); today it ran and hit the standing `Quantize input dtype` refusal, which is byte-identical in both commits (`patches/mlx-omarchy-quantize-errors.patch`).

## Per-file wall times (`per-file-wall-times.csv`)

No file approaches the 900 s watchdog. Slowest: `test_blas.py` 34.3 s,
`test_array.py` 27.1 s, `test_quantized.py` 24.0 s, `test_fast_sdpa.py` 23.7 s
(junit per-suite seconds). The 2026-09-06 snapshot's 40 watchdog cases sat 39
in `test_fast_sdpa.py` and 1 in `test_ops.py`; neither file produces any this
run, and no phase aborted on timeout or signal.

## Raw evidence

`py/` and `cpp/`: per-file `.log` (pytest `-v` / doctest console) and `.xml`
(junit / doctest XML) for every upstream test file, plus `summary.tsv` per
phase. Phase artifacts: `source-commit.txt`, `mlx.lock`, `prepare.log`,
`cpp-binary.sha256`, `python-provenance.json`, `wheel-sha256.txt`.
Classification produced by `tools/analyze-py-suite.py` (canonical buckets;
run over `py/*.xml` excluding the all-skipped `test_conv_transpose.py.xml`,
whose empty report trips that tool's sanity guard — it contributes no failing
cases) and `cluster-root-causes.py` (clustering + delta, this directory).
