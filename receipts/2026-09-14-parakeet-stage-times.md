# Parakeet stage times on jwm1 with cached SPIR-V mel

Date: 2026-09-14

One end-to-end process on `jwm1-linux`: vulkan mel (SPIR-V disk cache hit) + ANE encoder islands A+B+C + vulkan decoder + TDT. Token-exact E2E is not claimed.

## Verdict

Cached mel in this pipeline is 224.445 ms. The prior ANE-encoder E2E (`receipts/2026-09-14-parakeet-e2e-ane-encoder.json`) paid a cold `glslc` and recorded 9790.079 ms for the same stage. Isolated cache-hit mel on this host is 188.1 ms (`origin/main` `receipts/2026-09-14-mel-frontend-perf.md`). The 36 ms gap is wrapper cost around the same 8 compute dispatches / 1 submission.

Whole-pipeline wall is 22268.333 ms versus native P50 ~450 ms for the private TalkTastic dictation model on an M1 Mac Mini (plan §1 quote). This public fixture is 10.435 s of LibriSpeech audio. Encoder wall is 18070.683 ms (81% of the pipeline); ANE worker exec inside that is 5056.347 ms across 72 one-shot submits. This is not a hardware pass of the 450 ms target.

## Identity

- `origin/main`: `b1570fdfde11adcb1db17b56cf8da4f5d426d549` (`receipts: TDT emission 99 break is encoder residual in decoder_state, not frame-373`). Includes `deb9c72bc12156df88a738afc781b3644a998d7b` and `18989098ef083b8ae7b5ffca640a0f156a4d5d00`.
- Decoder: `/var/tmp/ParakeetE2E/pkg/coreml/vulkan_decoder.py` SHA-256 `bea0e2e6f503cb635c40928b1f52bbd0e63d09f94c67ed16bcd85ed0e241fd47` (byte-identical to `origin/main` `overlay/tools/coreml/vulkan_decoder.py`).
- Mel: `vulkan_mel.py` SHA-256 `4f61cd5cd1ebabb2a96d3270e347d3a6607be9b9c186f56663c5f9d14ba095c0` (pkg, MelFrontendPerf, and local overlay).
- Encoder runner: `/var/tmp/ParakeetE2EAne-stage/vulkan_encoder.py` SHA-256 `cd858ed352f6b0d25e035643cef3e5bd43ad689614fe7e396a3f7af1d4edec6d` (islands ABC).
- libmlx: `0.32.2.dev202609141626+05015a76` (SPIR-V disk cache). Overlay of `05015a76` is the implementation that landed as `18989098`. libmlx SHA-256 `65a641e4a9e6973734a6f25f810e62c7d9e60c4a4a0e44e4490fd9518a25f2e4`. Device `Apple M1 (G13G B1)`, Honeykrisp, `Device(gpu, 0)`.
- ANE module `6fa243a`, `/dev/accel/accel0`.
- Host: `jwm1-linux`, kernel `7.1.6-1-1-ARCH`, aarch64, Python 3.14.7.
- Lock: `/tmp/m1-gpu.lock` inode 29, `flock -w 900`, not stolen, not unlinked. Held 2026-09-14T18:48:41Z–18:49:03Z UTC. Post-run `flock -n` free.
- Agent model: `xai-oauth/grok-4.6`; fallback: true (not OpenAI).

## Stage wall

| Stage | this run ms | prior E2E cold-glslc ms | notes |
| --- | ---: | ---: | --- |
| audio_load | 170.465 | 338.954 | pinned FLAC, 166960 samples @ 16 kHz |
| mel_frontend | 224.445 | 9790.079 | SPIR-V cache hit; 8 vk compute, 1 submit; mel bit-exact |
| encoder_ane | 18070.683 | 18082.765 | islands ABC, 72 submits, 0 timeouts, 96 ANE ops / 1278 GPU ops |
| decoder_load | 106.739 | 141.754 | |
| tdt_decode | 3659.408 | 3585.693 | vulkan decoder + joint + greedy TDT; 145 decoder / 146 joint calls |
| detokenize | 36.591 | 42.220 | |
| **total_pipeline** | **22268.333** | **31981.465** | |

ANE worker exec 5056.347 ms (prior 5086.665). Decoder mean 20.487 ms, joint mean 4.116 ms.

Cache directory `MLX_OMARCHY_SPIRV_CACHE=/var/tmp/MelFrontendPerf/spirv-ab.3KDGHZ`: 9 pre-existing `.spv` files kept the same `mtime_ns` (mel hits). One new file `cbdf649251c42df007eaf40d902c06e77190b4955931b1dbb0dc7d130bfc4c58.spv` (9044 bytes) was written during this process (decoder custom kernel, not the mel set).

## Comparison to native P50 ~450 ms

The 450 ms figure is the owner's private post-trained Parakeet on an M1 Mac Mini with encoder on ANE, not a measurement of this public `mweinbach1/parakeet-tdt-0.6b-v3-coreml` clip. Against that note, this Linux pipeline is ~49× the 450 ms P50 (22268 / 450). Removing the old cold-glslc tax does not move the encoder: 18.07 s of 22.27 s remains ANE-island submit plus GPU encoder ops in one-shot worker processes.

## Correctness (not token-exact)

- Mel and encoder input features/mask: bit-exact vs the macOS capture.
- Encoder hidden: in frozen bounds (max 0.14716, mean 0.004182, rel_L2 0.024917); byte-identical to `/var/tmp/EncoderParityAne/out-ane/encoder_hidden.npy` (`b3d60c81bcd9c63fcdcb5c5578d765cc221b0126af3482fb165d769a65c23e6e`).
- Tokens: diverged. Matching prefix 99 / 105 actual emissions (native 104). Same transcript SHA-256 `7de78db9a83f6efc496e8c155b10dbd3d0c7ee1343bb3b4c6949c7480d76dd96` as the prior ANE-encoder E2E. `cpu_tensor_events` 0.

## Artifacts on host

Root `/var/tmp/ParakeetStageTimes/out`

| file | SHA-256 |
| --- | --- |
| e2e-report.json | `ba670630e4b8b9357664fc54cd895c2cced926424a9f2ce29d4caaac073e231e` |
| transcript.txt | `7de78db9a83f6efc496e8c155b10dbd3d0c7ee1343bb3b4c6949c7480d76dd96` |
| token_ids.json | `d9184e2f6c71e9262019154e7cb76d69a3aece11470f2f7097fa811098fc9d24` |
| encoder_hidden.npy | `b3d60c81bcd9c63fcdcb5c5578d765cc221b0126af3482fb165d769a65c23e6e` |
| mel.npy | `5b54f4a9a2ba3434cd69b6e48e6780d3bcb6c635d9ce85cda3d85c60f2455bde` |

## Reproduce

On `jwm1-linux`, lock inode 29, never steal, never unlink:

```text
export MLX_OMARCHY_SPIRV_CACHE=/var/tmp/MelFrontendPerf/spirv-ab.3KDGHZ
export PYTHONPATH=/var/tmp/MelFrontendPerf/venv-cache/lib/python3.14/site-packages
flock -w 900 /tmp/m1-gpu.lock \
  /home/joshuawarren/venv-agxgen/bin/python /var/tmp/ParakeetE2E/parakeet_e2e.py \
  --audio /var/tmp/ParakeetE2E/audio/fixture.flac \
  --golden /var/tmp/EncoderParityAne/capture \
  --model ~/.cache/mlx-omarchy/parakeet-reference/mweinbach1/parakeet-tdt-0.6b-v3-coreml/b650695c2322ee5281dff48d7345b2f3a58ff018 \
  --pkg /var/tmp/ParakeetE2E/pkg \
  --encoder-runner /var/tmp/ParakeetE2EAne-stage/vulkan_encoder.py \
  --source /var/tmp/EncoderParityAne/encoder-source \
  --bundles /var/tmp/jwm1-encoder-islands/bundles \
  --worker /var/tmp/jwm1-select-island-exec2/mlx-omarchy-ane-worker \
  --libane /var/tmp/AneWorkerValidation-c05ba1df/receipts/2026-09-13-ane-worker-validation/libane.so \
  --scratch /var/tmp/ParakeetStageTimes/scratch \
  --ane-reference /var/tmp/EncoderParityAne/out-ane/encoder_hidden.npy \
  --out /var/tmp/ParakeetStageTimes/out \
  --deadline-ms 20000
```

`PYTHONPATH` selects the `05015a76` cache wheel; agxgen supplies protobuf. `venv-cache/bin/python` alone lacks `google.protobuf`.

## Not claimed

- token-exact E2E
- transcript-exact E2E
- bit-exact encoder vs native capture
- a 450 ms hardware pass
- Phase 9 latency
