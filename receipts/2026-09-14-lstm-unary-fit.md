# 2026-09-14 LSTM unary fit: the searched fp16 contract space cannot flip e101

Status: named, not closed. The host search over fp16 unary contracts —
correctly rounded, staged fp16 chains, exp2 forms, magnitude-scaled, sigmoid-
derived, capture pins, and piecewise combinations, 4760 pairs — finds no pair
bit-exact on the reduction-free lanes (best 517 of 640 `next_cell`, 434 of 640
`next_hidden` under the full contract), and the live native-encoder decode
shows the fit frontier moves the emission-101 gap the wrong way. No unary in
the space flips e101. Nothing lands.

- Hole: `parakeet.tdt.decoder-lstm.fp16-recurrent-vs-native-coreml`, open.
- The shipped correctly-rounded pair (macOS CPU contract, `c9f78a9c`) is
  unchanged; `overlay/tools/coreml/vulkan_decoder.py` still hashes
  `bea0e2e6f503cb635c40928b1f52bbd0e63d09f94c67ed16bcd85ed0e241fd47`.
- The H13 LUT stays refuted (0/11 and 1/8 pins; 116 and 99 of 1280 in the
  earlier fixture) and was never a candidate here.

No GPU lock stolen: two jwm1 runs, each one `flock -w 900 /tmp/m1-gpu.lock`
acquisition, ~19 s wall each. jw16 untouched. `63c1d3cf` not merged.

## Execution

Two legs, mirroring the e101 receipt.

Host leg — `probe_unary_fit.py`, host-only NumPy at fp64 with explicit fp16
rounding stages; no MLX, no GPU, no ANE, no lock, jwm1's gate spy reused from
the authenticated e101 artifacts (`shipped_gates.npz`, `c17bbf3b…`).

- host: `omp-studio-local`, kernel `6.17.2-1-pve`
- python `3.11.2`, numpy `1.26.4`
- worktree `LstmUnaryFit` at `agent/lstm-e101` `79aaeea9`

Device leg — `probe_unary_live.py` on jwm1, five greedy-decode arms of the
native-encoder path in one process per run, E2E overlay bytes (decoder
`bea0e2e6…`, joint `bf31537d…`), native encoder
`/var/tmp/EncoderParityAne/capture/encoder_hidden.npy` (`7e442034…`), native
tokens `token_ids.json` (`a175a5f9…`).

## The searched space

Sigmoids (28): correctly rounded; staged fp16 chains `1/(1+exp(-x))` over all
8 round/neg × round/exp × round/add subsets; `exp2` with log2(e) at fp16/32/64;
`0.5*tanh(x/2)+0.5`; each also with the 11 capture pins overriding. Tanhs
(170): correctly rounded; `(e-1)/(e+1)` chains over 8 staging subsets;
`1-2/(e+1)` at 12 stagings; multiply-by-reciprocal magnitude/sign at fp16 and
fp32 intermediates; `cr*(1+k)` magnitude scale, k in {2.5e-5 … 2.5e-4};
`2*sigmoid(2x)-1` and `sigmoid(2x)-sigmoid(-2x)` built from every sigmoid;
each also with the 8 capture pins overriding. 28 × 170 = 4760 pairs.

Scoring is the reduction-free step (transition 0, layer 0: zero embedding row
and zero entry state make the gates equal the fp16 bias bits), with the exact
product and one fp16 state rounding — the shape the e101 receipt settled:

- `next_cell` = `f16(S(b_i) · T(b_g))` (the forget term vanishes exactly)
- `next_hidden` = `f16(S(b_o) · T(cell))`, tanh at the candidate cell (full
  contract) and at the native cell (identity view) both recorded

## Host results

| pair | cell | hidden (full) | hidden (identity) | joint |
| --- | --- | --- | --- | --- |
| fp16 chain sigmoid, `cr*(1+1e-4)` tanh | **517** | **434** | 501 | **951** |
| fp16 chain sigmoid, correctly rounded tanh | 500 | 409 | 486 | 909 |
| correctly rounded, `cr*(1+1e-4)` tanh | 426 | 311 | — | 737 |
| correctly rounded both, the shipped pair | 413 | 291 | 403 | 704 |

No pair of 4760 reaches 640/640 on either tensor. The baselines reproduce the
prior receipts exactly (413 and 291; chain-sigmoid cell 500, hidden 486 under
the identity view). Pin reproduction: the chain family with rounding after
exp, add and divide reproduces all 11 sigmoid pins; `cr*(1+1e-4)` and
`cr*(1+1.5e-4)` reproduce all 8 tanh pins — the fitted magnitude bias sits
inside the +0.124 to +0.267 ulp window the stages receipt derived, now
confirmed as a pin-complete materialization (it remains a fit, not an
identification).

Generalization disagrees with the reduction-free fit. Over all 15 comparable
transitions, native-injected state, shipped gate reduction:

| pair | `next_cell` of 19200 | `next_hidden` of 19200 |
| --- | --- | --- |
| correctly rounded both (shipped) | 3369 | 3185 |
| fp16 chain sigmoid, CR tanh | 3821 | 2890 |
| fp16 chain sigmoid, `cr*(1+1e-4)` tanh | 3735 | 2786 |

The tanh bias that wins the reduction-free lanes loses the recurrence. On the
145 × 2 recorded GPU gate rows of the e101 shipped arm, the frontier pair
changes a mean 238 cell and 286 hidden lanes of 640 per call-layer (chain
sigmoid alone: 202/247); the pins override touches 125 of 290 rows, 1–2 lanes
each.

## Live: no arm flips emission 101

Five arms, native emissions `…8135, 7892, 7863, 8135` at 99–102. Every arm
matches native through emission 100 and emits **8029** at 101. The shipped arm
reproduces the e101 receipt bit for bit: `shipped_e101_token_logits.npy`
hashes `62532f16…` and `shipped_e101_decoder_state.npy` hashes `9d028f19…`,
identical to the hashes that receipt records, on a fresh process and a fresh
lock acquisition.

| arm | argmax logit (8029) | logit 7892 | gap to native | flip |
| --- | --- | --- | --- | --- |
| shipped (CR both) | −12.6875 | −12.921875 | −0.234375 | no |
| `cr*(1+1e-4)` tanh only | −12.625 | −12.8828125 | −0.2578125 | no |
| chain sigmoid + `cr*(1+1e-4)` tanh | −12.6015625 | −12.8671875 | −0.265625 | no |
| chain sigmoid + `cr*(1+1.5e-4)` tanh | −12.6484375 | −12.90625 | −0.2578125 | no |
| CR + capture pins, piecewise | −12.6484375 | −12.859375 | −0.2109375 | no |

Every functional-form candidate moves the gap away from the flip (to
−0.258…−0.266); only the piecewise pins arm moves toward it, to −0.2109375,
an improvement of 0.023 against the 0.235 still needed. The measured spread of
the whole searched space at e101 is about ±0.03 of gap, an order of magnitude
short of the margin, and the direction of the functional forms is wrong.

## Remaining hole

`parakeet.tdt.decoder-lstm.fp16-recurrent-vs-native-coreml` stays open, now
with the catalogue space exhausted under the reduction-free metric, the pin
machinery, and the live flip metric. What is missing is unchanged from the
stages receipt: the identity of the native `tanh` (and residual per-argument
`sigmoid`) evaluation inside the capture backend's lstm op. Closing it needs
either a single-operator authenticated capture of fp16 `tanh`/`sigmoid` over
an argument sweep, or the compiled compute plan that names the evaluation. A
form that reproduces the 19 pins *and* wins the recurrence *and* moves the
live gap +0.235 would be an identification; nothing catalogued comes close.

## Not claimed

No token-exact or transcript-exact E2E (all arms still miss at 101 with 104
emissions). No closed LSTM operator. No code change, no test change: the
contract tests pass untouched — `tests/coreml/test_fp16_activations.py` and
`test_vulkan_decoder.py`, 4 passed, 1 skipped. No ANE execute, no module load,
no jw16 contact, no merge.

## Artifacts

`derivation/`: `probe_unary_fit.py` `8b360869…`, `probe_unary_live.py`
`97f1df4f…`, `result.json` `aa69d816…`, `result_live.json` `fd83506e…`,
`shipped_gates.npz` `c17bbf3b…` (copied from the e101 artifacts),
`SHA256SUMS`, and per-arm `*_e99/*_e101_*` decoder state and token logits.
On jwm1: `/var/tmp/LstmUnaryFit_probe.py`, `/var/tmp/LstmUnaryFit_out/`.

```text
# host leg
python3 receipts/2026-09-14-lstm-unary-fit/derivation/probe_unary_fit.py \
  --capture-dir ~/.cache/mlx-omarchy/parakeet-reference/captures/\
b650695c-75aec2a/20260913T105550Z-librispeech-tdt-tensors/ane \
  --package ~/.cache/mlx-omarchy/parakeet-reference/mweinbach1/\
parakeet-tdt-0.6b-v3-coreml/b650695c2322ee5281dff48d7345b2f3a58ff018/decoder.mlpackage \
  --overlay overlay \
  --live-gates receipts/2026-09-14-lstm-unary-fit/derivation/shipped_gates.npz \
  --output receipts/2026-09-14-lstm-unary-fit/derivation/result.json

# device leg (jwm1)
ssh jwm1 'flock -w 900 /tmp/m1-gpu.lock ~/venv-agxgen/bin/python \
  /var/tmp/LstmUnaryFit_probe.py \
  --overlay-tools /var/tmp/TdtEmission99/overlay/tools \
  --model ~/.cache/mlx-omarchy/parakeet-reference/mweinbach1/\
parakeet-tdt-0.6b-v3-coreml/b650695c2322ee5281dff48d7345b2f3a58ff018 \
  --native-encoder /var/tmp/EncoderParityAne/capture/encoder_hidden.npy \
  --native-tokens /var/tmp/EncoderParityAne/capture/token_ids.json \
  --out /var/tmp/LstmUnaryFit_out'
```

- resolved_model: `zai/glm-5.3-flash`
- fallback: false
