# 2026-09-14 Vulkan TDT decision 142

Status: diverged. Named remaining operator:
`parakeet.tdt.decoder-lstm.fp16-recurrent-vs-native-coreml`.

Not Phase 7. No ANE encoder execute. No 375-wide ANE select. No CPU
tensor fallback. No 8289b1bc promotion.

## Hardware

- host: `jwm1-linux`, kernel `7.1.6-1-1-ARCH`
- device: `Apple M1 (G13G B1)`, architecture `honeykrisp`
- driver: Honeykrisp, `Mesa 26.3.0-devel (git-6f6afc8968)`
- lock: `/tmp/m1-gpu.lock` was free; one outer `flock -x -w 120`; never unlinked
- runtime: `mlx 0.32.2.dev202609131130+e4d079c`
  - `libmlx.so` sha256 `f604640d8c125eb2fb9082381691cd94b4f3aa42b169499561a6fd0d5eda542f`
  - core sha256 `4ad850da16c300ea9f60aa7247f034f358aa16d7ba2fd146979a01c867e3cfa5`
  - source commit `e4d079c883b8606dddef7720bd302c5756be0692`

## Sources

Worktree `CoremlTdtStream` at `1f8804cd087e60b7ab34a2a730510977f5aca0c3`.

| file | sha256 |
| --- | --- |
| `overlay/tools/coreml/vulkan_decoder.py` | `5a163be6f78b1e1314040175ac768a4dbf3c270a0a0fa5b2fe5c88595fd46fb4` |
| `overlay/tools/coreml/vulkan_joint.py` | `bf31537d504612c6d9873051c9ad5f7b84a978268e6d7ed4eaec173a18626115` |

Capture: authenticated `20260913T105550Z` encoder hidden
`7e442034cb42521f7b76ec9645e8d0082899a5f551ea8d046c62a9283cc6e4e5`.
Model revision `b650695c2322ee5281dff48d7345b2f3a58ff018`.

## Command

```text
hostname
flock -x -w 120 /tmp/m1-gpu.lock \
  env PYTHONPATH=/tmp/vulkan-tdt-142 \
  control-venv/bin/python /tmp/vulkan-tdt-142/diagnose_decision_142.py
```

Wall 3.067 s. GPU trace delta: 17302 `vk_compute_dispatches`, 31003
`gpu_primitive_dispatches`, 630 submissions. Device after the run still
enumerates `Apple M1 (G13G B1)` / Honeykrisp.

## Free decode

145 decoder calls, 146 joint calls. First divergence is emission 101 /
decision 142:

- actual token 8029, frame 373, duration 0
- native token 7892, frame 373, duration 0
- decoder input token 8135
- actual logit -12.6171875, native-token logit -12.890625, gap 0.2734375
- duration index 0 on both (logits `[13.21875, 8.7265625, -10.0625, -45.96875, -8.3984375]`)
- encoder frame sha256 `421387a213f3d218dfd55d08a0a22446b7252810711aceb31307b0e598c2d56d`
- vulkan decoder_state sha256 `135f9f69b51d5e42d00d5cfbc1dc76bd3fb3a37ba49bde1d70bb1e55ba77daeb`

## Localization

Joint matmul is not the miss. Feeding captured native
`joint_decoder_state` plus native encoder frames through
`run_joint` matches all 16 captured token decisions and all 16 duration
decisions.

Decoder LSTM state is the miss. On the 15 captured decoder transitions,
native-injected inputs still disagree with native Core ML:

| output | trace 0 max abs | worst max abs |
| --- | --- | --- |
| `next_cell` | 0.0028076171875 | 0.03125 |
| `next_hidden` | 0.000732421875 | 0.019287109375 |
| `decoder_hidden` | 0.01953125 | 0.01953125 |

`next_cell` is already non-bit-exact at the first transition, so the
recurrent LSTM operator itself does not match native Core ML.

Related, not sufficient to close 142: projector on native LSTM
`next_hidden` still disagrees with native `decoder_output_hidden`
(trace 0 max abs 0.01953125). That is
`parakeet.tdt.decoder-projector.fp16-linear-vs-native-coreml`. Joint
decisions remain exact when the native projector output is supplied, so
this is not the decision-142 argmax path.

## One-line operator

None. `run_joint` on the vulkan decoder_state at decision 142 already
selects 8029. Changing LSTM or projector to `addmm` / fp32 does not
establish native Core ML accumulator semantics; 8289b1bc stays excluded.

## Proof limit

The native capture stops tensor traces at decision 15. There is no
native decoder state or token logit tensor for decision 142. Closing the
hole needs those authenticated tensors, not a joint-arithmetic tweak.

## Not claimed

Phase 7 E2E, encoder ANEC, token-exact TDT, or Honeykrisp-specific
cause. Software-Vulkan previously reproduced the same 8029/7892
emission; this receipt only adds Apple GPU execution and LSTM-vs-joint
localization.
