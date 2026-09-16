# Parakeet E2E on a current non-diag wheel: new stage-time baseline (2026-09-15)

Verdict: the full Parakeet E2E re-run on jwm1 with a **release (non-diag)
wheel built from origin/main `b5bf90e`** — which contains the
`ConvF32 → MatmulF32` encoder conv land (`e55c1fae`) that every earlier
stage-time receipt predated — holds the full correctness contract:
**104/104 emissions, transcript sha `db501a8c…`, `encoder_hidden` = pin
`38c73261…`, mel bit-exact**, and the encoder stage drops **19386.0 →
12638.7 ms (−6747 ms, −34.8%)**, whole pipeline **22268.3 → 16068.0 ms**.
This receipt supersedes `receipts/2026-09-14-parakeet-stage-times.md` as the
Parakeet baseline.

## Wheel and provenance

- Built with `scripts/build-wheel.sh` (no `--diagnostics`) in worktree
  `~/src/mlx-e2e-curb5bf90ee` at `b5bf90ee` (origin/main tip; `108fd4b5` is an
  ancestor), upstream mlx pin `1f8e74e3` per `mlx.lock`.
- Wheel `mlx_omarchy-0.32.2.dev202609152338+b5bf90e-cp314-cp314-linux_aarch64.whl`,
  sha256 `b62dd34b5b65d3397d74e2122e7084d579dee3279cf4d9ca766060c4e574f397`.
- Installed to `/var/tmp/ParakeetE2ECurrentWheel/site`;
  `scripts/mlx_provenance.py --expect-wheel <whl>` → **`verified: "match"`**,
  `dist_version == mx_version == 0.32.2.dev202609152338+b5bf90e`.
- Profiling gate literals (`MLX_OMARCHY_GPU_PROFILE`) in `libmlx.so`: **0**.

## E2E result (jwm1-linux, real pipeline)

Lock `/tmp/m1-gpu.lock` (inode 29), `flock -w 900`, never stolen, never
unlinked; lock free after the run. Same fixture, golden capture, islands,
worker and libane as the prior receipts.

| quantity | value |
| --- | --- |
| status | **match** |
| emissions actual / native | **104 / 104**, matching_prefix_length 104 |
| transcript | match, sha256 `db501a8c080380ea027ffa50a4b4956c39df77cb692c4fb78e556311a11a0790` |
| encoder_hidden sha256 | **`38c73261f29230276ed76f1fc017b76b024156d79218bd5f1347fdc7e7d43ec7`** (pin) |
| mel sha256 | `5b54f4a9a2ba3434cd69b6e48e6780d3bcb6c635d9ce85cda3d85c60f2455bde` (unchanged, bit-exact) |
| encoder bounds | max/mean/rel-L2 PASS (identical bytes ⇒ identical stats to the pin run) |
| ANE | islands ABC, 72 submits, 0 timeouts, exec 5041.5 ms |
| decoder / joint calls | 145 / 146 |

## Stage wall

| Stage | this run ms | 2026-09-14 wheel `05015a76` ms | notes |
| --- | ---: | ---: | --- |
| audio_load | 187.398 | 170.465 | pinned FLAC, 166960 samples @ 16 kHz |
| mel_frontend | 250.107 | 224.445 | SPIR-V cache warm (`spirv-ab.3KDGHZ`); mel bit-exact |
| encoder_ane | **12638.745** | 18070.683 | islands ABC, 5202 vk compute dispatches — the `e55c1fae` conv win, now in a shippable wheel |
| decoder_load | 99.113 | 106.739 | |
| tdt_decode | 2880.339 | 3659.408 | warm compiled-cache; 145 decoder / 146 joint calls |
| detokenize | 12.338 | 36.591 | |
| **total_pipeline** | **16068.041** | **22268.333** | |

## Identity

- Backend: wheel `b5bf90e` (see provenance above), loaded via
  `PYTHONPATH=/var/tmp/ParakeetE2ECurrentWheel/site:/var/tmp/MelFrontendPerf/venv-cache/lib/python3.14/site-packages`.
- Encoder runner: origin/main fused runner, `overlay/tools/coreml/vulkan_encoder.py`
  sha256 `6c175adf1bf5a39717496924b1ef4579bb1f8eb5ff42064b081b0caf037a6f0a`,
  staged copy in this receipt dir.
- Decoder: `vulkan_decoder.py` sha256
  `637f077ece00025e5be3c341db256c8933ae19655804d3439490709d9ebc60ef`, from
  `/var/tmp/ParakeetE2EAneBnns/pkg`. **Trap recorded:** the canonical
  `/var/tmp/ParakeetE2E/pkg` still carries the stale pre-token-exact decoder
  (`bea0e2e6…`); pointing the E2E at it diverges 99/105 even with an exact
  encoder. The token-exact decoder lives only in the `AneBnns` staged pkg.
- e2e runner `/var/tmp/ParakeetE2E/parakeet_e2e.py` sha256
  `1623707880831c963514eb07b533a3e707bfda6ce05208932a0a4522b10f12a7`; run
  command in `run-e2e.sh` in this receipt dir.
- Worker `762dd1de…` (`jwm1-select-island-exec2`), libane `1ab9d95d…`,
  bundles `island-attn-a-kt` / `island-select-8head-scratch417` / `island-pv`.
- Host `jwm1-linux`, aarch64, kernel `7.1.6-1-1-ARCH`, Python 3.14.7,
  `/dev/accel/accel0`, device Apple M1 (G13G B1), Vulkan Mesa Honeykrisp.

## Artifacts

`receipts/2026-09-15-parakeet-e2e-current-wheel/`:
`e2e-report.json` (same content as this dir's sibling
`2026-09-15-parakeet-e2e-current-wheel.json`, sha256
`c4cade4f81c2053a752ac76681c48b08d510169817d5ed0915f26deed9674c9e`),
`transcript.txt` (`db501a8c…`), `token_ids.json`, `encoder_hidden.npy`
(`38c73261…`), `mel.npy`, `vulkan_encoder.py`, `run-e2e.sh`.

## Not claimed

- No cold-SPIR-V mel run in this receipt; mel quoted warm. The cold-mel
  penalty is unchanged from prior receipts (~9.7 s first compile).
- No claim on Phase 9 latency, macOS, other fixtures, or batching/resident-
  worker improvements (owned elsewhere); ANE exec 5041.5 ms matches the
  long-standing 5.0 s attribution.
- tdt_decode warmed the compiled cache from prior same-day runs; the
  3659.4 → 2880.3 ms delta mixes wheel and cache-warmth effects.

## Superseded (2026-09-16)

This baseline was measured on wheel `b5bf90e` with launch-mode islands
(72 submits) and the host greedy TDT control loop. The full E2E has been
re-run on a non-diag release wheel built from origin/main `7d82ec94` —
island batch (`d8c9afce`), GPU-resident TDT loop default (`992feea9`),
and the strict libane pin (`f20c634d`) all in the product wheel — with
the correctness contract holding on six runs (104/104, transcript
`db501a8c…`, `encoder_hidden` pin `38c73261…`): encoder_ane 9375.2 ms,
tdt_decode 832.0 ms, whole pipeline 10730.3 ms (5-run medians). The new
baseline is `receipts/2026-09-16-parakeet-e2e-baseline.md` / `.json`.
