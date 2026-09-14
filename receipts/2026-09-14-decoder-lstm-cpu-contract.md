# 2026-09-14 Parakeet decoder LSTM CPU activation contract

Status: closed. Linux decoder LSTM activations are correctly-rounded fp16
`sigmoid`/`tanh`, matching macOS Core ML CPU.

- Closed: `parakeet.tdt.decoder-lstm.fp16-elementwise-activation-vs-native-coreml`
  on the unary contract: 1280/1280 versus the reduction-free gate arguments
  and Mac `y_cpu_only_*.bin`, and 1536/1536 on the full Mac fixture.
- Not claimed: token-exact end-to-end, decision 142, or bit-exact LSTM
  products. Independently rounded `sigmoid * tanh` still scores 413/640
  `next_cell` against the original capture; Mac CPU decoder *rerun* matches
  that capture 1280/1280, so the remaining LSTM hole is state algebra /
  product rounding, not the unaries.

No GPU lock. No ANE. No `/tmp/m1-gpu.lock`. No jwm1. No formatter.

## Execution

Host-only numeric proof of the unaries. `_lstm` now calls `fp16_sigmoid` /
`fp16_tanh` (host round, then `mx.array(..., dtype=mx.float16)`). Gate
matmuls stay on `mx.gpu`.

- host: `omp-studio-local`, kernel `6.17.2-1-pve`
- python `3.11.2`, numpy `1.26.4`
- worktree HEAD `7f8786b02f88d56cb98f5779aea7b9ab4a715972`

## Proof

```text
python3 -m pytest -q tests/coreml/test_fp16_activations.py
....                                                                     [100%]
4 passed in 0.17s
```

Landed `fp16_sigmoid` / `fp16_tanh` versus Mac CPU bins and the 1280
transition-0 arguments (`layout.json` `args` `[0, 1280]`, concatenated
`x_{op}_{0,1,2}.bin`):

| op | 1280 args | 1536 fixture | max abs | fp16 chain 1280 | H13 LUT 1280 |
| --- | --- | --- | --- | --- | --- |
| sigmoid | 1280/1280 | 1536/1536 | 0 | 899/1280 | 116/1280 |
| tanh | 1280/1280 | 1536/1536 | 0 | n/a | 99/1280 |

`y_cpu_only_sigmoid.bin` sha256
`f9c34dafaaa89890fe7c0e9fc1e6e58f95ba05746794d6ef98364e3d8ff9e1a0`
`y_cpu_only_tanh.bin` sha256
`aa62e7af7d2411f40a2b266c5cb9a30b590c1d378697e3ba004315a6227f4115`

Capture `tdt_tensors.json` sha256
`6eaf2bfe801584af570cd66f42e8dc17efd67776fb181f691585d2bcab590e0a`.

Mac CPU versus correctly-rounded fp16 was already 1536/1536, 0 ULP, in
`receipts/2026-09-14-decoder-activation-capture-mac/result.json`. H13 LUT
versus that Mac CPU is 179/1536 sigmoid and 152/1536 tanh. The Linux
functions are the Mac CPU contract, not the LUT and not the fp16 chain.

## Landed diff

| file | sha256 |
| --- | --- |
| `overlay/tools/coreml/vulkan_decoder.py` | `e50bc54b3b890420a0fe81584facc50b4d6ae6b8579e434f23d50dfbb4c30c1a` |
| `tests/coreml/test_fp16_activations.py` | `188c101e10c5a79396631f8c54a1b7cf0d42e3a9d4852ac5736af5a51a65f9d2` |
| `tests/coreml/test_vulkan_decoder.py` | `526c162eb205ac29497ee8ec2e8c0ee8d4345c786da3dd3891632541dc2c8a92` |

`git diff --stat` against HEAD `7f8786b0`:

```text
 overlay/tools/coreml/vulkan_decoder.py | 36 +++++++++++++++++++++++++++-------
 tests/coreml/test_vulkan_decoder.py    | 19 +++++++++---------
 2 files changed, 38 insertions(+), 17 deletions(-)
```

`tests/coreml/test_fp16_activations.py` is new. Numpy LSTM in
`test_vulkan_decoder._mil_lstm` uses the same functions. Named decoder
input errors (`TypeError` / `ValueError` on dtype and shape) are
unchanged.

## Not claimed

No GPU timing, no ANE execute, no driver reload, no merge, no token-exact
E2E. GPU `_lstm` activations host-round; that is the Mac CPU contract, not
a Honeykrisp unary kernel.

- resolved_model: `xai-oauth/grok-4.6`
- fallback: true
