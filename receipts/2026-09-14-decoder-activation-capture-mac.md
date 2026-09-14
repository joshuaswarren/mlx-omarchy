# 2026-09-14 Mac decoder activation capture (MLComputePlan)

Status: the Linux decoder contract is correctly-rounded fp16 `sigmoid`/`tanh`
(macOS CPU), not the H13 LUT. Named hole stays open until `vulkan_decoder`
uses that contract. No shipped code changed.

Quoted from `receipts/2026-09-14-decoder-activation-capture-mac/result.json`
(`schema` `mlx-omarchy.parakeet-decoder-activation-capture-mac/1`). Capture
`tdt_tensors.json` sha256 `6eaf2bfe801584af570cd66f42e8dc17efd67776fb181f691585d2bcab590e0a`.

- Still open: `parakeet.tdt.decoder-lstm.fp16-elementwise-activation-vs-native-coreml`.
- Not a hardware pass. This leaf hashed existing Mac artifacts and wrote this
  receipt. No GPU lock, no ANE, no `/tmp/m1-gpu.lock`.

## Host

`mac_host`: hostname `MacStudio.local`, `hw_model` `Mac13,2`, chip `Apple M1 Ultra`,
macos `26.6.2`, build `25G83`, Core ML `3520.5.1`, coremltools `9.0`, python
`3.11.16`, numpy `1.26.4`.

## Compute plan

Four Core ML compute-unit settings: `cpu_only`, `cpu_and_gpu`, `cpu_and_ne`, `all`.

Decoder LSTM preferred device is `MLCPUComputeDevice` under every setting.
LSTM does not list `MLNeuralEngineComputeDevice` as supported: under `all` and
`cpu_and_gpu` the supported set is CPU+GPU; under `cpu_and_ne` and `cpu_only`
it is CPU only. `ops_preferred_non_cpu` is empty on every unit.

One-op `ios18.sigmoid` and `ios18.tanh` also prefer `MLCPUComputeDevice` on
every unit, including `all` and `cpu_and_ne` where NE is in the supported set.

`units_identical_outputs`: `sigmoid` true, `tanh` true. All four units produce
identical one-op outputs.

## Decoder rerun versus original capture

`decoder_rerun_equal_to_capture` on every unit:

| tensor | equal / of |
| --- | --- |
| `decoder_hidden` | 640 / 640 |
| `next_cell` | 1280 / 1280 |
| `next_hidden` | 1280 / 1280 |

## CPU contract versus LUT

`mac_cpu_*_minus_correctly_rounded_ulps_1536`: 1536 / 1536 bit-exact, max_abs 0
ULP, empty exceptions, both `sigmoid` and `tanh`.

`mac_cpu_vs_jwm1_lut_equal_1536`: `sigmoid` 179 / 1536, `tanh` 152 / 1536.

## Native pins

| source | sigmoid | tanh |
| --- | --- | --- |
| mac CPU (`cpu_only` and the other three units) | 9 / 11 | 7 / 8 |
| jwm1 H13 LUT | 0 / 11 | 1 / 8 |

The two remaining native-pin mismatches versus mac CPU are original-capture-host
versus Mac Studio, not Linux versus macOS:

| op | argument | mac_cpu | native |
| --- | --- | --- | --- |
| sigmoid | 0.314453125 | 0.578125 | 0.57763671875 |
| sigmoid | 0.56640625 | 0.6376953125 | 0.63818359375 |
| tanh | 0.319091796875 | 0.30859375 | 0.308837890625 |

## Linux contract

Use correctly-rounded fp16 `sigmoid`/`tanh` (what this Mac CPU produced). Do
not use the H13 LUT. The named hole remains until `vulkan_decoder` uses that
contract.

## Sources

Everything under `receipts/2026-09-14-decoder-activation-capture-mac/` is hashed
in its `SHA256SUMS`. Mac run: `capture_mac.py` on MacStudio. Host score:
`analyze.py` wrote `result.json`.

## Not claimed

No GPU timing, no ANE execute, no driver reload, no code change, no merge.

- resolved_model: `xai-oauth/grok-4.6`
- fallback: false
