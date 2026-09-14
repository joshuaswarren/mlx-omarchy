# 2026-09-14: three numpy MIL evaluators collapsed into one tool

Three receipt directories each carried a private evaluator for the textual
MIL that `overlay/tools/coreml/mil_adapter.py` emits. All three shared the
`0xDEADBEEF` blob-v2 record reader, the MIL dtype table, and the fp16 contract
(accumulate in fp32, round once to the declared result dtype). They are now one
maintained module, `overlay/tools/coreml/mil_numpy.py`, with one test,
`tests/coreml/test_mil_numpy.py`.

## The forks

| Fork | Lines | Receipt | Golden check (max_abs / mean_abs / rel_l2, bound 0.3 / 0.02 / 0.1) |
|---|---|---|---|
| `receipts/2026-09-14-encoder-islands-exec-jw16mbp1-linux/mil_numpy.py` | 539 | `2026-09-14-encoder-islands-exec-jw16mbp1-linux.json` | 0.1509 / 0.00411 / 0.0247 (`interpreter-validation.json`) |
| `receipts/2026-09-14-encoder-islands-exec-jwm1-linux/derivation/milrun.py` | 532 | `2026-09-14-encoder-islands-exec-jwm1-linux.json` | 0.1389 / 0.00722 / 0.0436 (`validate-full.json`) |
| `receipts/2026-09-14-encoder-parity-ane/derivation/vulkan_encoder.py` | 874 | `2026-09-14-encoder-parity-ane.json`, `2026-09-14-parakeet-e2e.json` | 0.1349 / 0.00410 / 0.0246 (`compare-vulkan.json`, GPU control arm) |

What each fork had that the other two lacked:

- **jw16 `mil_numpy.py`** (pure numpy, pull-based `get`): rounds every result
  to the dtype the MIL declares for it rather than to a fixed fp16; parses the
  `func main` signature and shape-checks fed inputs against it; walks every
  blob record header and checks the storage dtype code (1 fp16, 2 fp32,
  3 uint8, 4 int8, 14 int32) before reading; accepts inline
  `tensor<...>(...)` literal operands and hex float literals; implements
  `identity`, `same` padding, `split_sizes`, and the `stride` key of
  `slice_by_index`; sign-split sigmoid that never overflows `exp`. It lacked
  `reduce_max`, the `x0..xN` concat form, an ops-executed count, and any
  memory release (it cached every value it produced).
- **jwm1 `milrun.py`** (pure numpy, front-to-back `run`): streaming
  `run(stop_after, wanted)` returning the executed-op count; memmap blob
  reader with a payload-length check; im2col conv through stride tricks
  (the jw16 fork looped over kernel taps); rank-1 conv lifted to a unit H
  axis; `reduce_max`; the `x0..xN` concat form; `strides` key of
  `slice_by_index`. It lacked the storage-code check, input shape checks,
  inline literals, `identity`, `same` padding, `split_sizes`, and released
  nothing during a run; its `main` hard-coded the jwm1 layer-0 staging paths.
- **EncoderParityAne `vulkan_encoder.py`** (mlx.core on `mx.gpu`): the same
  parser and op table as `milrun.py` ported to MLX, plus last-use memory
  release, on-demand `ensure` next to the streaming `run`, the ANE island A/C
  splice through the worker CLI, `mlx_omarchy_trace_snapshot` counters, an
  argparse CLI and a run-report JSON. It had no numpy path at all: it needs
  an installed `mlx-omarchy` and a device.

## The canonical tool

`overlay/tools/coreml/mil_numpy.py` (654 lines, numpy only) is the union:
declared-dtype rounding, `func main` input shape checks, memmap blob reader
with both the storage-code and the payload-length check, inline and hex
literals, the full op set of all three forks (`identity`, `same` padding,
`split_sizes`, `reduce_max`, both concat forms, both slice stride keys), im2col
conv with a single-pass depthwise path, on-demand `get` and streaming `run`
with last-use release, an executed-op count, a `validate()` function and a CLI
that scores a run against the reference lock's frozen encoder contract.

The ANE island splice, the MLX trace counters and the layer-0 staging scripts
were not folded in; they are receipt-specific drivers, not evaluator code.

## Verification on this host (omp-studio-local, Linux, numpy 1.26.4)

CLI, MIL `ac8e9526...` from `/tmp/coreml-text-adapter-pinned-source-v2`,
capture `~/.cache/mlx-omarchy/parakeet-reference/captures/b650695c-75aec2a/20260912T154759Z-librispeech/ane`
(golden `encoder_hidden.npy` sha256 `7e442034...`, matches the lock):

```
ops_executed 3351, encoder_hidden [1, 375, 640], encoder_mask_equal true
encoder_max_abs_err  0.14642333984375   (bound 0.3)
encoder_mean_abs_err 0.004131156485528  (bound 0.02)
encoder_rel_l2_err   0.024851588532329  (bound 0.1)
nan 0, inf 0, within_frozen_contract true, elapsed 14.3 s, exit 0
```

`python3 -m pytest tests/coreml/test_mil_numpy.py -q`: `1 passed in 19.93s`.
The test re-emits the MIL from the pinned `encoder.mlpackage` (same
`model.mil` sha256 `ac8e9526...`), checks the capture hashes against the
lock, and asserts 3351 ops, the shape, an exact mask and the frozen contract.
It skips when the pinned package or the capture is not installed.

## Evidence left untouched

The three fork files are byte-identical to what their receipts hashed:
`milrun.py` `dfd148da...` (pinned in the jwm1 receipt JSON) and
`vulkan_encoder.py` `0665dccd...` (pinned in the parity-ane and parakeet-e2e
receipt JSONs). Adding a header comment would have broken those pins, so the
pointer to the canonical tool is a one-line `CANONICAL-EVALUATOR.md` beside
each fork instead of a line inside it.
