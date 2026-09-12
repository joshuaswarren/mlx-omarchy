# Parakeet reference freeze (phase 1)

Status: **frozen** — golden tensors, token IDs and transcript captured on the
reference Mac and pinned by hash in
[`overlay/tools/coreml/parakeet-reference.lock`](../overlay/tools/coreml/parakeet-reference.lock).

## What is pinned

| Pin | Value |
|---|---|
| Reference implementation | [`mweinbach/parakeet-coreml-swift`](https://github.com/mweinbach/parakeet-coreml-swift) @ `75aec2a1c991319657ff4dec5f602c12da6c5012` (Apache-2.0) |
| Public model | [`mweinbach1/parakeet-tdt-0.6b-v3-coreml`](https://huggingface.co/mweinbach1/parakeet-tdt-0.6b-v3-coreml) @ `b650695c2322ee5281dff48d7345b2f3a58ff018` (CC-BY-4.0, inherits `nvidia/parakeet-tdt-0.6b-v3`) |
| Package files | 12 files pinned by size + SHA-256 (HF LFS oids verified against the live tree at download time) |
| Quantization | fp16 compute; encoder 4-bit palettized (kmeans, per-grouped-channel, conv excluded); decoder/joint fp16 |
| Audio fixture | LibriSpeech test-clean, 1089-134686-0000, CC-BY-4.0; Narsil/asr_dummy@8d141c84e3f84c54cd7bbaa851d24edd0f559734:1.flac; SHA-256 30885601…ed94c2 |
| Golden outputs | `waveform`, `mel`, `mel_mask`, padded `encoder_input_*`, `encoder_hidden`, `encoder_mask`, `token_ids`, `transcript`, `environment`, `crosscheck` — SHA-256 in the lock |

Model spec facts (official `coremltools` 9.0 schema inspection, ML Program,
specification version 9, minimum deployment target macOS 15):

```text
encoder:  input_features f32 [1,3000,128], attention_mask i32 [1,3000]
          -> encoder_hidden f32 [1,375,640], encoder_mask i32 [1,375]
decoder:  input_ids i32 [1,1], hidden f16 [2,1,640], cell f16 [2,1,640]
          -> decoder_hidden f16 [1,1,640], next_hidden/cell f16 [2,1,640]
joint:    encoder_frame f32 [1,640], decoder_state f32 [1,640]
          -> token_logits f32 [1,8193], duration_logits f32 [1,5]
```

## Reference conventions (derived from the pinned Swift source)

* Audio: `AVAudioConverter` (quality `high`) → 16 kHz mono f32 in [-1, 1];
  fixed 30 s chunks (3000 frames × hop 160), last chunk zero-padded.
* Mel (matches HF `ParakeetFeatureExtractor`): preemphasis 0.97 (`y[0]=x[0]`);
  centred STFT, `n_fft=512`, `win_length=400` (symmetric Hann), hop 160,
  zero pad-mode; `|STFT|²` via sqrt-then-square; Slaney mel, 128 bins,
  0–8000 Hz; `log(mel + 2^-24)`; per-bin mean/std over frames with Bessel
  correction and `eps=1e-5`: `(x - mean) / (std + eps)`.
* Decode: greedy TDT, blank id 8192, durations `[0,1,2,3,4]`, vocab 8193,
  max 10 symbols/step, zero-init LSTM state, mask-truncated frame loop.

Every numerics-bearing step in the golden capture ran inside the pinned
reference library (SwiftPM `exact revision` dependency); the capture harness
(`overlay/tools/coreml/capture/`) only orchestrates and dumps `.npy`/JSON.

## Numerical comparison contract (frozen before Linux execution work)

Golden reference host: **Apple M1 Ultra** (Mac13,2), macOS 26.6.2 (25G83),
CoreML framework 3520.5.1, compute units `cpuAndNeuralEngine`. It is never
labelled as any other SoC.

The numerical limits remain those frozen from the original reference. The
licensed replacement clip passes them without adjustment:

```text
encoder_hidden (vs golden ANE capture):
  max |Δ|            ≤ 0.30      (licensed clip: CPU 0.145203, GPU 0.139700)
  mean |Δ|           ≤ 0.02      (licensed clip worst: 0.004393)
  relative L2        ≤ 0.10      (licensed clip worst: 0.025172)
  NaN / Inf          = 0
decoder/joint: emitted token IDs must match exactly
  (plan section 40, layer 6; duration/frame metadata remains diagnostic)
transcript: must match exactly
```

Host-side preprocessing (waveform, mel, padded inputs, mask) was
**bit-identical** across all three compute plans, so it is compared exactly.

The licensed clip produces identical 104-token outputs on ANE and GPU.
The pinned reference emits repeated punctuation and a Cyrillic suffix; these
remain in the golden transcript. CPU emits 100 tokens and does not match that
transcript. Each compute plan matches its own end-to-end reference transcriber.
This freeze records native behavior, not clean transcription quality. See the
[licensed reference receipt](../receipts/2026-09-12-licensed-parakeet-reference.json)
for exact text, token counts, and duration/frame differences.

## Downloader

```bash
python3 overlay/tools/coreml/fetch_parakeet_reference.py download  # fetch + verify
python3 overlay/tools/coreml/fetch_parakeet_reference.py verify    # re-hash cache
python3 overlay/tools/coreml/fetch_parakeet_reference.py path      # print cache dir
python3 overlay/tools/coreml/fetch_parakeet_reference.py info      # lock summary
```

Integrity rules: every cache file is fully re-hashed on every verify (nothing
is trusted on first use, on size, or on prior stamps); content that does not
match the pin is re-fetched; freshly fetched content that still mismatches is
a hard error; the live HF tree is cross-checked against the pin before any
download (LFS oids = SHA-256, non-LFS compared by git blob id); a verifying
cache needs no network at all. `MLX_OMARCHY_HF_ENDPOINT` overrides the HF
base (tests/mirrors); `MLX_OMARCHY_CACHE_DIR` overrides the cache root.
Tests: `python3 -m pytest tests/coreml/ -q`.

Cache layout: `$MLX_OMARCHY_CACHE_DIR|~/.cache/mlx-omarchy/parakeet-reference/<model-repo>/<revision>/`.

## Golden capture

Bulk capture data lives **outside git** at
`~/.cache/mlx-omarchy/parakeet-reference/captures/b650695c-75aec2a/` with
`manifest.sha256` per capture; the lock pins each artifact's SHA-256.
Primary golden: `20260912T154759Z-librispeech/ane`. Matched GPU and CPU
captures are sibling directories in that run. Earlier JFK captures are historical
evidence, not the current golden. Reproduce on an Apple Silicon Mac with:

```bash
cd overlay/tools/coreml/capture
swift build -c release
.build/release/parakeet-reference-capture \
  --audio 1089-134686-0000.flac --models <model-dir> --out <capture-dir> \
  --compute-units ane --expect <path>=<sha256> ...
```

The harness verifies all `--expect` hashes before any inference and cross-
checks its composed run against the reference end-to-end `ParakeetTranscriber`
(tokens and transcript must match; they do).

## Licensing record

* `parakeet-coreml-swift` source: Apache-2.0 (repo `LICENSE`).
* Model packages + tokenizer: CC-BY-4.0, inherited from
  `nvidia/parakeet-tdt-0.6b-v3` (recorded in package spec metadata and HF card).
* Audio fixture: [LibriSpeech ASR corpus, SLR12](https://www.openslr.org/12/),
  CC-BY-4.0. Attribution: Vassil Panayotov, Guoguo Chen, Daniel Povey, Sanjeev
  Khudanpur. The pinned mirror is byte-identical to
  `LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac` in the official
  test-clean archive. Mirror bytes are unmodified; reference preprocessing
  converts them to float32 and zero-pads the chunk as specified above.
* `coremltools` (schema inspection + vendored proto schema under
  `overlay/tools/coreml/schema/`): BSD-3-Clause (`LICENSES/coremltools.txt`).
* No third-party weights are committed to `mlx-omarchy`; the downloader
  fetches the pinned revision instead.
