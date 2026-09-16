# Parakeet E2E baseline on 7d82ec94: GPU TDT loop default, island batch, strict libane (2026-09-16)

Verdict: **LAND.** The full Parakeet E2E re-run on jwm1 with a release
(non-diag) wheel built from origin/main `7d82ec94` holds the full correctness
contract on all six runs: **104/104 emissions, transcript sha `db501a8c…`,
`encoder_hidden` = pin `38c73261…`, mel bit-exact, `control: gpu-loop` with
`tdt_fallback_reason = null`**, island batch engaged (1 batch submit, 1 worker
start, 0 timeouts per pass), and the pipeline median drops
**16068.0 → 10730.3 ms (−5337.7 ms, −33.2%)** — encoder_ane
12638.7 → **9375.2 ms** (island batch −2.4 s ANE exec), tdt_decode
2880.3 → **832.0 ms** (GPU-resident loop default). This receipt supersedes
`receipts/2026-09-15-parakeet-e2e-current-wheel.md` as the Parakeet baseline.

## Wheel and provenance

- Built with `scripts/build-wheel.sh` (**non-diag**, no `--diagnostics`) in
  worktree `/var/tmp/BaselineE2E-7d82ec94` at `7d82ec94` (origin/main tip;
  island batch `d8c9afce`/`bff4fc34`, fused leftover `108fd4b5`, GPU TDT loop
  default `992feea9`, strict libane `f20c634d`, coopmat revert `b9b5bf69` all
  ancestors), `MLX_OMARCHY_ANE_SOURCE_DIR` → `~/src/omarchy-ane-lifecycle-rebase`
  (omarchy-ane `6fa243a`, clean).
- Wheel `mlx_omarchy-0.32.2.dev202609160525+7d82ec94-cp314-cp314-linux_aarch64.whl`,
  7 856 601 bytes, sha256
  `e54cb680db437377f65ad6bc65b051e1c5664fd30d482caf382d59283f4ab832`.
- Installed to `/var/tmp/ParakeetE2EBaseline7d82/site`;
  `scripts/mlx_provenance.py --expect-wheel <whl>` → **`verified: "match"`**,
  `dist_version == mx_version == 0.32.2.dev202609160525+7d82ec94`, both
  `mlx.core` and `libmlx.so` match the wheel RECORD member-for-member.
- Profiling gate literals (`MLX_OMARCHY_GPU_PROFILE`) in `libmlx.so`: **0**
  (profiling harness compiled out).
- Wheel-shipped runtime worker `mlx/bin/mlx-omarchy-ane-worker` (fd-protocol,
  statically carries the strict `mlx_omarchy_libane` pinned at omarchy-ane
  `6fa243ac…` with `LIBANE_CONFIG_STRICT_BIND`) sha256
  `5a3b28c136ef7da7b9284a191e2e7bce378ac4dcac9fcf5f4dd5154f2fe94fb5`.

## E2E result (jwm1-linux, real pipeline, 6 runs: r1 + 5 repeats)

Lock `/tmp/m1-gpu.lock` (inode unchanged), `flock -w 900`, never stolen, never
unlinked; `/run/lock/mlx-omarchy-ane/quarantine` 0 before and after every run;
lock free after the battery. No decode flags on any run: the GPU-resident
greedy TDT loop is the default path (`control: gpu-loop`,
`tdt_fallback_reason: null` on all six). Island batch selected with
`ANE_ISLAND_MODE=resident-batch` (the runner's env knob; launch stays the
runner default). Strict libane worker per run:

- worker `mlx-omarchy-ane-worker` (serve CLI, built from the same tree lineage
  as the wheel pin; ane/ sources byte-identical 46bb0700 → 7d82ec94) sha256
  `f171a61ecfc89942b009dc379d6636b17b0dda6f583841470a1d789853f5952e`,
- libane `libane-strict.so` (omarchy-ane `6fa243a` + `LIBANE_CONFIG_STRICT_BIND`)
  sha256 `56b462346128b04139cb7c9397b06ca8d513c680c972d8914395ddd3aa978ba7`,
- bundles `island-attn-a-kt` / `island-pv` / `island-select-8head-scratch417`
  from `/var/tmp/island-reexport/bundles` (mil-hwxc `b61de468`; byte-identical
  to `receipts/2026-09-14-encoder-split-plan/bundles` at `7d82ec94`).

| quantity | value (every run r1–r6) |
| --- | --- |
| status | **match** |
| emissions actual / native | **104 / 104**, matching_prefix_length 104 |
| tokens / durations / frame indices | all match |
| transcript | match, sha256 `db501a8c080380ea027ffa50a4b4956c39df77cb692c4fb78e556311a11a0790` |
| encoder_hidden sha256 | **`38c73261f29230276ed76f1fc017b76b024156d79218bd5f1347fdc7e7d43ec7`** (pin) |
| mel sha256 | `5b54f4a9a2ba3434cd69b6e48e6780d3bcb6c635d9ce85cda3d85c60f2455bde` (bit-exact) |
| encoder bounds | max/mean/rel-L2 PASS |
| ANE | 1 batch submit, 1 worker start, 0 timeouts |
| decode control | `gpu-loop`, fallback `null`, 0 decoder/joint host calls |
| cpu_tensor_events | 0 |

## Stage wall

| Stage | r1 ms | **median r2–r6 ms** | median r1–r6 ms | prior `b5bf90e` ms |
| --- | ---: | ---: | ---: | ---: |
| audio_load | 172.952 | 177.050 | 175.001 | 187.398 |
| mel_frontend | 238.871 | 232.082 | 235.476 | 250.107 |
| encoder_ane | 9398.912 | **9375.209** | 9382.201 | 12638.745 |
| decoder_load | 98.685 | 97.273 | 97.742 | 99.113 |
| tdt_decode | 949.095 | **831.978** | 832.327 | 2880.339 |
| detokenize | 58.602 | 45.475 | 45.556 | 12.338 |
| ANE exec (worker-side) | 2624.432 | 2623.262 | 2623.847 | 5041.534 |
| **total_pipeline** | **10917.116** | **10730.336** | **10751.385** | **16068.041** |

Per-run totals: r1 10917.1, r2 10699.1, r3 10642.5, r4 10730.3, r5 10785.4,
r6 10772.4 ms. r1's tdt_decode (949.1 ms) is the first gpu-loop pass on this
boot; every later run sits at 831–834 ms, matching the `992feea9` gate
(831.3 ms).

## Attribution

- encoder_ane −3263.5 ms (median vs prior run) with ANE exec 5041.5 → 2623.3 ms
  (−2418 ms): the island batch (`d8c9afce`) — 72 per-layer submits became 1
  bounded batch submit on one resident worker, plus row-wise tile pack.
- tdt_decode −2048.4 ms: the GPU-resident greedy TDT loop (`992feea9`) as the
  default decode path, no flags.
- detokenize +33.1 ms vs the prior single run: small-numbers noise on a
  12→45 ms stage across two days; no claim beyond the medians above.

## Identity

- Host `jwm1-linux`, aarch64, kernel `7.1.6-1-1-ARCH`, Python 3.14.7,
  `/dev/accel/accel0`, device Apple M1 (G13G B1), Vulkan Mesa Honeykrisp.
- Harness `fused_e2e.py` sha256
  `0e38e7b15a36d9a8244bd523ff6845f6b44cb7ad4a7444dc30e8177e97cabb97`
  (byte-identical to `receipts/2026-09-15-tdt-gpu-loop/derivation/fused_e2e.py`
  at `7d82ec94`); pkg `/var/tmp/TdtLoopDefault/pkg` whose `coreml/` is
  byte-identical to `overlay/tools/coreml/` at `7d82ec94` (decoder
  `637f077e…` token-exact, `parakeet_tdt.py` `c842e011…` gpu-loop default,
  runner `240e3b63…`). Fixture FLAC sha `30885601…`, 166960 samples @ 16 kHz;
  model pin `b650695c…`; golden capture `/var/tmp/EncoderParityAne/capture`.
- Run command in `run-e2e.sh` in this receipt dir (r1 landed in `out-rr1`
  because the script appends its index after a literal `r`; runs r2–r6 in
  `out-r2`…`out-r6` — naming only, no behavioral difference).

## Artifacts

`receipts/2026-09-16-parakeet-e2e-baseline/`: `e2e-report-r1.json` (full
schema-1 report, run r1), `transcript.txt` (`db501a8c…`), `token_ids.json`,
`encoder_hidden.npy` (`38c73261…` pin), `mel.npy`, `run-e2e.sh`,
`verify-all-runs.py`, `collect-medians.py`. Stage walls for all six runs and
both median sets live in `receipts/2026-09-16-parakeet-e2e-baseline.json`.
jwm1 stage: `/var/tmp/ParakeetE2EBaseline7d82/` (site, out-rr1/out-r2..r6,
collect/verify scripts), wheel worktree `/var/tmp/BaselineE2E-7d82ec94`.

## Not claimed

- No cold-SPIR-V mel run; mel quoted warm (`spirv-ab.3KDGHZ`). The cold-mel
  penalty is unchanged from prior receipts.
- tdt_decode r1 vs r2–r6 warmth note above; the median table is the claim.
- No claim on Phase 9 latency, macOS, other fixtures, batching beyond the
  resident-batch pass, or the reverted coopmat path (measured slower,
  reverted at `7d82ec94`).
- `63c1d3cf` not merged, not touched.

## Superseded (2026-09-16, evening)

This baseline was measured at `7d82ec94` (10730.3 ms total median on jwm1).
Later same-day work landed `83de2255`, `ecd3e81f`, `69fd5397` (total
8460.9 ms) and then the `296c4352` fold — which, once measured end-to-end
on a clean provenance-verified environment, turned out to corrupt the
encoder silently (deterministic wrong `encoder_hidden`, 0/104 emissions)
and was reverted on main (`616b5b89`, tip `f43ab71c`). The current Parakeet
baseline is `receipts/2026-09-16-parakeet-e2e-both-hosts.md` / `.json`:
f43ab71c, no flags, 12/12 runs green on jwm1 (8541.8 ms median r2-r6) AND
jw16 (6418.5 ms, first E2E on the M1 Max), transcript `db501a8c…`,
`encoder_hidden` pin `38c73261…` on both SoCs.
