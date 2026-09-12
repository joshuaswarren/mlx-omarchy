# Receipt: Parakeet reference freeze (phase 1), 2026-09-12

Branch `wave/coreml-mac-reference` (isolated worktree from
`origin/wave/parakeet-reference` @ `be7c649a`). Phase 1 of
`docs/plans/2026-09-12-coreml-parakeet-ane-plan.md`: complete numerical
reference freeze + reproducible downloader, independent of the Linux parser.

## Pinned inputs (all verified by executed commands)

* Model: `mweinbach1/parakeet-tdt-0.6b-v3-coreml` @ `b650695c2322ee5281dff48d7345b2f3a58ff018`.
  Cache `~/.cache/mlx-omarchy/parakeet-reference/mweinbach1/<rev>/` hashed locally
  and diffed against the live HF API tree: every LFS oid matched the local
  SHA-256 (command output in session log; `fetch_parakeet_reference.py download
  --force` re-runs the drift check, exit 0).
* Reference implementation: `mweinbach/parakeet-coreml-swift` @
  `75aec2a1c991319657ff4dec5f602c12da6c5012` (cloned, `git rev-parse HEAD`
  confirmed; Apache-2.0 LICENSE read).
* Audio fixture: `openai/whisper@86098128c0b4f24f0e2aa2994de830614b474227:tests/jfk.flac`,
  SHA-256 `63a4b1e4c1dc655ac70961ffbf518acd249df237e5a0152faae9a4a836949715`,
  1152693 bytes, 44.1 kHz stereo 24-bit FLAC, 11.0 s (`soundfile.info`).
  Real speech, not silence: transcript below.

## Official schema inspection (no handmade parser oracle)

`coremltools` 9.0 on Linux (`pip install --user coremltools`), spec dump of all
three packages (session log has full output):

* encoder: ML Program, spec v9; `input_features` f32 [1,3000,128],
  `attention_mask` i32 [1,3000] → `encoder_hidden` f32 [1,375,640],
  `encoder_mask` i32 [1,375]; fp16 precision, 4-bit palettization
  (kmeans/pgc, conv excluded), min target macOS 15, `cpu_and_ne`.
* decoder: `input_ids` i32 [1,1], `hidden`/`cell` f16 [2,1,640] →
  `decoder_hidden` f16 [1,1,640], `next_hidden`/`next_cell` f16 [2,1,640].
* joint: `encoder_frame` f32 [1,640], `decoder_state` f32 [1,640] →
  `token_logits` f32 [1,8193], `duration_logits` f32 [1,5].

## Golden capture on macstudio (Apple M1 Ultra — never labelled T8103)

Host: `ssh macstudio` → `MacStudio.local`, macOS 26.6.2 (25G83), arm64,
Swift 6.3.3, CoreML framework 3520.5.1, 128 GB. Fleet-SSH skill followed
(alias from `~/.ssh/config`, BatchMode probe, no sudo, no firmware/driver
writes, no reboots; user-space SwiftPM build + model inference only).

Staging: pinned revision downloaded on the Mac from HF resolve URLs and
verified with `shasum -a 256 -c` — all 12 files OK plus the audio fixture
(output in session log). Capture harness (`overlay/tools/coreml/capture/`,
SwiftPM dependency pinned by `revision:` to the reference commit) verified
all 7 `--expect` hashes before any inference.

Command: `capture/.build/release/parakeet-reference-capture --audio audio/jfk.flac
--models models/mweinbach1 --out captures/<stamp>-<plan> --compute-units ane|gpu|cpu --expect ...`

Results (golden = `20260912T135844Z-ane`):

```text
waveform: 176000 samples @16kHz (11.00s); chunks: 1; mel: 3001 frames x 128 bins
encoder_hidden: [1, 375, 640] dtype=f32
tokens: 52
transcript: And so, my fellow Americans, ask not what your country can do for
            you ask what you can do for your country................
crosscheck: tokens_match=true transcript_match=true   (composed run vs
            pinned end-to-end ParakeetTranscriber)
```

GPU and CPU captures produce the **identical** token/duration/frame sequences
and transcript. Cross-plan comparison (numpy, in session log):

```text
waveform/mel/mel_mask/encoder_input_features/encoder_input_mask: bit-identical
encoder_mask: identical
ANE vs GPU: max_abs=0.11483 mean_abs=0.007074 rel_l2=0.038581
ANE vs CPU: max_abs=0.13129 mean_abs=0.008553 rel_l2=0.047111
NaN/Inf: 0
```

Frozen contract (lock `numerical_contract`): encoder max_abs ≤ 0.30,
mean_abs ≤ 0.02, rel_l2 ≤ 0.10 (= 2× measured worst, rounded up), NaN/Inf = 0,
token/duration/frame sequences + transcript exact. Host preprocessing
compared exactly (bit-identical observed).

Note: the encoder input is 3000 frames (spec maxTime); the reference mel of a
padded 30 s chunk yields 3001 frames and the reference input construction
copies `min(frames, 3000)` — captured inputs are the actual tensors fed
(`encoder_input_features.npy` etc.), so the golden is self-consistent.

## Bulk data location (outside git)

`~/.cache/mlx-omarchy/parakeet-reference/captures/b650695c-75aec2a/`
(primary ANE capture + GPU/CPU cross-plan captures, each with
`manifest.sha256`; artifact hashes pinned in the lock's
`macos_reference_paths`). A partial capture from a crashed first attempt was
kept only long enough to record preprocessing determinism (mel/waveform/mask
SHA-256 identical to the golden across independent process runs) and then
deleted. No bulk tensors are committed.

## Downloader + tests

* `overlay/tools/coreml/reference.py` — lock schema (v1), validation, full-hash
  cache verification, stamps as receipts only (never trust shortcuts),
  `git_blob_sha1` for non-LFS drift checks, `MLX_OMARCHY_HF_ENDPOINT` override.
* `overlay/tools/coreml/fetch_parakeet_reference.py` — `download|verify|path|info`
  CLI; refuses moved upstream, refuses mismatched content, offline when cache verifies.
* `parakeet-reference.lock` — real pins incl. frozen contract + golden hashes.
* `tests/coreml/` — 28 tests, all pass: hermetic fake-HF server exercises
  fresh download, cached reuse without refetch (request-count assertion,
  server dead on second run), missing file refetch, tampered local repair,
  upstream drift refusal, corrupt-transfer refusal (size-only integrity
  rejected), predictable path, info state; lock validation rejects; real-cache
  integration test (`verify` on the real 458 MB cache, `OK: 12 files verified`).

Real-command receipts (session log): `download` short-circuits on verifying
cache; `download --force` runs the drift check (exit 0 after non-LFS blob-id
path was fixed); `verify` → `OK: 12 files verified`.

## Interface agreement

With `CoreMLPackageReader` (hub): `reference.py` public surface as listed in
their `__init__.py` docstring; CLI advertised as plain script
`python3 overlay/tools/coreml/fetch_parakeet_reference.py download|verify|path`;
no downloader→inspector dependency; their `proto.py`/`mlpackage.py`/
`__init__.py`/inspector files untouched by this branch.

## Deviations / notes

* Audio provenance corrected during capture: the fixture is the JFK inaugural
  address excerpt ("ask not…"), not the "man in the arena" passage first
  assumed; lock note fixed before commit. Model transcript (with the model's
  emitted trailing periods) is the pinned expectation.
* Joint-logit tensor bounds are `None` in the contract by design: logits are
  internal to the reference decode loop; the enforced observable is exact
  token/duration/frame sequence equality.
