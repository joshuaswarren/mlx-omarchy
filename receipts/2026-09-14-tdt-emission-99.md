# 2026-09-14 TDT emission 99

Status: named cause, not closed. Token-exact E2E is not claimed.

Cause **(a)**: encoder residual accumulated into TDT.

Named tensor: Linux `decoder_state` at emission 99 (projected LSTM
hidden; `next_hidden` / `next_cell` with it).

Rule: the emission-99 argmax is that recurrent state, not encoder frame
373 and not joint/projector. Native `encoder_hidden` through the same
decoder emits native token 7863 (`▁`) at 99. ANE `encoder_hidden`
emits 7883 (`.`) at 99. Swapping only frame 373 does not flip either
state. ANE `decoder_state` at 99 matches native-encoder `decoder_state`
at 98 (hidden rel_l2 0.00364): the ANE residual is already inside LSTM
recurrence, one emission behind.

Native capture has no hidden, cell, or joint tensors at emission 98 or
99. Traces stop at decision 15.

## Hardware

- host: `jwm1-linux`, kernel `7.1.6-1-1-ARCH`, aarch64
- device: `Apple M1 (G13G B1)`, `Device(gpu, 0)`
- mlx `0.32.2.dev202609121216+8f6de34c`
- lock: `/tmp/m1-gpu.lock` inode 29; one outer `flock -w 900`; never
  unlinked; free after the run
- worktree: `TdtEmission99` at `8efa3ad3`

Decoder/joint bytes match the E2E run:

| file | sha256 |
| --- | --- |
| `overlay/tools/coreml/vulkan_decoder.py` | `bea0e2e6f503cb635c40928b1f52bbd0e63d09f94c67ed16bcd85ed0e241fd47` |
| `overlay/tools/coreml/vulkan_joint.py` | `bf31537d504612c6d9873051c9ad5f7b84a978268e6d7ed4eaec173a18626115` |
| `probe.py` | `7f31764037ba2e4a1e2ea919855fc60746ad00d16d030ed1c71be3ffe9e64e91` |

ANE encoder `b3d60c81` (same as
`receipts/2026-09-14-parakeet-e2e-ane-encoder.json`). Native encoder
`7e442034`.

## Command

```text
flock -w 900 /tmp/m1-gpu.lock \
  ~/venv-agxgen/bin/python /var/tmp/TdtEmission99/probe.py \
    --overlay-tools /var/tmp/TdtEmission99/overlay/tools \
    --model ~/.cache/mlx-omarchy/parakeet-reference/mweinbach1/parakeet-tdt-0.6b-v3-coreml/b650695c2322ee5281dff48d7345b2f3a58ff018 \
    --ane-encoder /var/tmp/ParakeetE2EAne/out/encoder_hidden.npy \
    --native-encoder /var/tmp/EncoderParityAne/capture/encoder_hidden.npy \
    --native-tokens /var/tmp/EncoderParityAne/capture/token_ids.json \
    --out /var/tmp/TdtEmission99/out
```

Wall 7.81 s. Two TDT arms, 145 decoder / 146 joint calls each.

## Emissions 98 and 99

Native tokens 96..103 at frame 373, duration 0:
`7883,7883,7883,7863,8135,7892,7863,8135`.

| arm | token 98 | token 99 | first token miss |
| --- | --- | --- | --- |
| ANE encoder | 7883 | **7883** | 99 (7883 vs 7863) |
| native encoder | 7883 | **7863** | 101 (8029 vs 7892) |

ANE-arm duration already differs at emission 1 (1 vs native 2). Token
prefix still matches through 98 and both arms sit on frame 373 at 99.

Joint logits at the emission-99 call, duration index 0 on every row:

| state × frame | argmax | logit 7883 | logit 7863 | gap |
| --- | --- | --- | --- | --- |
| ANE state 99 × ANE frame 373 | 7883 | -8.9296875 | -15.546875 | +6.6171875 |
| ANE state 99 × native frame 373 | 7883 | -8.96875 | -15.5625 | +6.59375 |
| native-enc state 99 × ANE frame 373 | 7863 | -14.5390625 | -14.125 | -0.4140625 |
| native-enc state 99 × native frame 373 | 7863 | -14.5703125 | -14.1484375 | -0.421875 |

Frame 373 itself is close: max_abs 0.009765625, rel_l2 0.01385. Full
encoder residual is larger (max_abs 0.14716, rel_l2 0.02492) and has
already entered the LSTM.

## Hidden / cell

ANE emission 99 vs native-encoder emission 98:

| tensor | max_abs | rel_l2 |
| --- | --- | --- |
| `next_hidden` | 0.0090332 | 0.003638 |
| `next_cell` | 0.742188 | 0.004676 |
| `decoder_state` | 0.0136719 | 0.005798 |

Same-index ANE vs native-encoder at 99 is a full step apart
(`decoder_state` rel_l2 0.384). Snapshots:
`receipts/2026-09-14-tdt-emission-99/`.

## Why not (b) or (c)

(b) LSTM recurrence beyond the activation is still open, but not this
break. Native encoder plus this decoder (correctly-rounded fp16
activations, serial fp16 projector) emits 7863 at 99 and first misses
at 101 (`8029` vs native `7892`). That remaining hole stays
`parakeet.tdt.decoder-lstm.fp16-recurrent-vs-native-coreml`.

(c) Joint/projector is not the argmax path. Projector is bit-exact on
captured transitions. Joint follows `decoder_state`: swapping only
frame 373 leaves both argmaxes unchanged, and the ANE-path gap at 99
is 6.62, not a near-tie on the current encoder frame.

## Not claimed

Token-exact E2E, transcript-exact E2E, bit-exact encoder, or a closed
LSTM operator. Native tensors at emission 99 do not exist in the
capture.
