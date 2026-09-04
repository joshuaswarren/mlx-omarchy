# Retain KV state slices without copying at evaluation

The evaluation-wide slice densifier was removed at `db10f538dbc90217d9229dc3caad61c80f800465`. Dense-only primitive consumers now normalize their own operands; stride-aware matmul and attention retain views. Python packed-buffer exports retain an owned read-only copy when necessary. Stride-aware exports remain zero-copy, and exported function constants are serialized in logical C order.

## Native validation

Apple M1, Honeykrisp, 2026-09-04; binary source `db10f538dbc90217d9229dc3caad61c80f800465`.

- `strided_consumer_equivalence.py`: 27 numerical checks passed.
- `buffer_interop_check.py`: 49 checks passed.
- `check_strided_serialization.py` at `32c7f92a1d86212c74b6ae12282516ffcc5f5084`: actual MLX slices/transposes round-tripped through safetensors, NPY, and exported functions against independent NumPy expectations.
- `kv_state_views.py` at `ebd4523651d5a35a0e131cd5b4ec3dc96b928af6`, diagnostics build, `--expect-no-eval-copy`: all verdicts passed.

The bounded KV probe uses two-head float16 caches of shape `[1,2,256,64]`, with 41 retained positions. It recorded 13 dispatch events across the stages. Slice evaluation alone recorded no state-extent copy. Neither eager matmul nor fast SDPA added one. Write-only instrumentation recorded two fills and two paste copies, proving that empty slice-evaluation results were not an inactive profiler.

A lazy view first evaluated after later cache writes retained its expected pre-write values. An already evaluated state read also retained its entire expected array; fresh parent reads contained the later writes. Expectations come from known writes on the host, not an earlier backend snapshot.

Raw native output: `~/benchq/logs/performance-final-db10f53-20260904-batch3/kv_state_views.out`. These are copy-count and correctness results, not a standalone KV speedup claim.

## Broader check limitations

The broad llvmpipe C++ run was not green. It exposed tests that flattened retained views and tests that expected now-obsolete layout refusals. After correcting those contracts, the shape (24 cases/266 assertions), complex (16/519), indexing (43/1,786), and fast-op (21/39,914) suites passed.

A real empty-K matmul span-check regression was fixed in `3d978f9`: empty dimensions no longer enter subtraction-based span arithmetic. The selected matmul case passed all 667 assertions, including zero-valued empty-K matmul and scaled addmm results. This changes degenerate shapes only; the performance results above remain attributed to `db10f53`, not to an unmeasured later binary.

Separate broad-suite crash and timeout observations remain unqualified; no full-suite pass is claimed. The local large QMM case exceeded the watchdog, while the complete 2,928-assertion QMM sweep passed on M1. Local logs: `/tmp/strided-combined-76a06dc-172317/`.
