# Parakeet E2E on both laptops at f43ab71c: jwm1 8541.8 ms, jw16 6418.5 ms — first E2E on the M1 Max (2026-09-16)

Verdict: **LAND.** The full Parakeet E2E now runs green on both laptops with
the non-diag release wheel from origin/main `f43ab71c` — the tip after
`616b5b89` reverted the `296c4352` fold rebase — with **no flags**: 12/12
runs hold **104/104 emissions, transcript sha `db501a8c…`, `encoder_hidden`
= pin `38c73261…` (identical bytes on T8103 and T6001), mel bit-exact,
encoder bounds PASS**, island batch engaged (1 batch submit, 1 worker start,
0 timeouts per pass), `control: gpu-loop`, `tdt_fallback_reason: null`.
Medians (r2–r6): **jwm1 8541.8 ms, jw16 6418.5 ms total pipeline** — the
first Parakeet E2E ever on the M1 Max. This receipt supersedes
`receipt/2026-09-16-parakeet-e2e-baseline` as the Parakeet baseline.

## What happened first: the fold on main was silently corrupting

The assignment targeted `296c4352` (fold default-on) with an expectation of
< 8000 ms total. Measuring it found the opposite: on jwm1, with a
provenance-verified wheel and the exact worker/libane/bundles of every prior
green battery, all six runs **diverged deterministically** — 0/104
emissions, empty transcript, bounds fail, `encoder_hidden` = `dadd090d…` on
every run — while the islands reported a healthy pass (72 rounds, 0
timeouts). No tm -5, no wedge: silent corruption.

A three-way A/B isolates it:

| arm | runner | fold | result |
| --- | --- | --- | --- |
| 1 | `296c4352` (`d7dadeb1`) | on (default) | **DIVERGE** (`dadd090d…`) |
| 2 | `296c4352`, `MLX_OMARCHY_CHAIN_FUSION=0` | off | **PASS** (pin `38c73261`, `db501a8c`, 8512.2 ms) |
| 3 | local `7da42928` (`c42ad3c2`) | on | **PASS** (pin `38c73261`, `db501a8c`, 10566.0 ms) |

`7da42928` is the same-named fold commit on the local
`/var/tmp/mlx-main-strict` tree — but on the **pre-coopmat base**: the fold
that was actually validated never met the coopmat-leftover-linear + fused
pointwise lineage. The rebase onto it (`296c4352`) broke the fold while
keeping the old validation claim in the message. Diff of the two runner
blobs is archived as `fold-runner-diff.txt` in this receipt dir.

Main's call: revert on main, rebuild, run both hosts with no flags, and do
**not** baseline a flagged-off configuration. `616b5b89` reverts the fold;
`f43ab71c` is the tip measured here. Its runner blob `ccdbf12a` is
byte-identical to the `69fd5397` runner — the state that went 6/6 green in
`receipts/2026-09-16-encoder-gpu-busy`.

## Wheel and provenance

- Built with `scripts/build-wheel.sh` (**non-diag**) in `/var/tmp/E2EREV`
  worktrees at `f43ab71c`, `MLX_OMARCHY_ANE_SOURCE_DIR` pinned to a clean
  omarchy-ane `6fa243ac7241119a9eb229abbf8cb4dd8949f915` checkout (the
  canonical CMake check) on each host.
- jwm1 wheel `mlx_omarchy-0.32.2.dev202609161313+f43ab71-cp314-cp314-linux_aarch64.whl`,
  7 856 626 bytes, sha256 `bfb6dc93dcc5a3de13af9a0a35a52ba8a03947d2686b4edd4469d08ed3eb4215`.
- jw16 wheel same filename, 7 856 610 bytes, sha256
  `54d520c738f6a36d3de54713cba9e3a0cc520d78f9a9857af47c779d6f4f044c`
  (minute-level timestamps make builds non-byte-reproducible; sources are
  the same commit).
- `scripts/mlx_provenance.py --expect-wheel <whl>` → **`verified: "match"`**
  on both hosts, `dist_version == mx_version == 0.32.2.dev202609161313+f43ab71`.
- Profiling gate literals (`MLX_OMARCHY_GPU_PROFILE`) in `libmlx.so`: **0**
  on both wheels.

## Runtime identity (identical on both hosts)

- Worker `mlx-omarchy-ane-worker` (standalone fd-protocol serve CLI)
  sha256 `f171a61e…` — the same binary as the `69fd5397` battery and the
  baseline receipt; worker sources are unchanged across
  `69fd5397` → `296c4352` → `f43ab71c` (span diff = runner + receipts). The
  wheel-shipped workers (jwm1 `85c24c6f`, jw16 `5e896572`) require an
  inherited ANE hardware lock and answer `expected STAGING_BYTES` on this
  fd protocol — a different spawn contract, not used.
- libane `libane-strict.so` sha256 `56b46234…` (omarchy-ane `6fa243a` +
  `LIBANE_CONFIG_STRICT_BIND`), same bytes on both hosts.
- Bundles: the `island-reexport` set (mil-hwxc `b61de468`, byte-identical to
  the `7d82ec94` split-plan bundles), the **same tree deployed on both
  hosts** (tree digest `18e3b7ea…`). jw16's old T6001 island set
  (`/var/tmp/jw16-encoder-islands/bundles`) predates the strict worker's
  surface-naming check and is rejected at load ("program 0 task stream does
  not name every surface; channel map is positional"); the unit-op family
  already proved ANE programs byte-identical across T8103/T6001
  (`receipts/2026-09-14-t6001-export-family.json`), and the identical
  `encoder_hidden` pin below proves it for the islands.
- Harness `fused_e2e.py` sha `0e38e7b1…` on both hosts (same bytes as the
  baseline receipt); model pin `b650695c…` with decoder+joint trees
  hash-identical across hosts; `ANE_ISLAND_MODE=resident-batch`; lock
  `/tmp/m1-gpu.lock` (`flock -w 900`, never stolen, never unlinked) free
  after each battery.

## E2E result (6 runs per host: r1 + 5 warm repeats)

Gates, every run on both hosts: status **match**, emissions **104/104**
(matching prefix 104), tokens/durations/frame indices match, transcript
`db501a8c…`, `encoder_hidden` **`38c73261…`**, mel bit-exact, encoder
bounds PASS, decode `gpu-loop` with fallback `null`, 1 batch submit /
1 worker start / 0 timeouts, `cpu_tensor_events` 0.

| Stage | jwm1 median r2–r6 | jwm1 median r1–r6 | jw16 median r2–r6 | jw16 median r1–r6 |
| --- | ---: | ---: | ---: | ---: |
| audio_load | 177.8 | 177.6 | 74.7 | 76.1 |
| mel_frontend | 247.4 | 248.4 | 164.7 | 164.0 |
| encoder_ane | 7215.3 | 7219.1 | 5083.0 | 5110.5 |
| decoder_load | 84.5 | 84.8 | 82.4 | 82.7 |
| tdt_decode | 830.8 | 831.3 | 961.2 | 961.6 |
| detokenize | 40.1 | 40.1 | 56.2 | 56.0 |
| ANE exec (worker-side) | 2650.9 | 2644.7 | 2579.9 | 2601.0 |
| **total_pipeline** | **8541.8** | **8565.0** | **6418.5** | **6444.8** |

Per-run totals — jwm1: r1 20716.1 (cold SPIR-V + first gpu-loop pass this
boot), r2 8476.0, r3 8588.2, r4 8532.4, r5 8541.8, r6 8642.1. jw16: r1
9553.8 (cold SPIR-V), r2 6471.0, r3 6418.5, r4 6503.9, r5 6075.9, r6
6310.0. Only the warm r2–r6 medians are the claim.

The Max runs the same island bytes with the same worker: encoder_ane
−2132 ms and mel −83 ms vs jwm1 (10 cores, more GPU), tdt_decode +130 ms
slower — its first-ever decode pass on this stack. Against the prior
measured state (69fd5397 cut arm, 8460.9 ms total / 7135.6 ms encoder on
jwm1), today's jwm1 numbers sit +81 ms / +83 ms — day-to-day variance on
the same runner blob, no claim either way.

## jw16 first-E2E logistics

llama-server (`llm-inference.service`, Main-confirmed not a router leg) was
stopped for each attempt and restarted after; the service is `active` now.
Three windows: 08:16:38–08:16:50 (stale bundles rejected), 08:18:59–08:19:00
(deployed script missing `--libane`, aborted at arg validation), and the
recorded 08:19:47–08:20:31 — **44 s stopped**, zero contention on any
recorded run, well inside the 30-min cap. Staged fresh on jw16 for this
first E2E: venv, harness, pkg, golden capture, encoder-source (1.46 GB
relayed), mel venv-cache, libane, worker, bundles.

## Wedge surveillance

No `tm completion failed: -110` and no `preserving resources until reboot`
during either battery. jwm1's only -110 lines this boot are at uptime
4419–4519 s — AneTmRecoveryBisect's recovery testing ~50 min earlier, each
ending `tm recovered: idle, accepting work again`; the battery ran at
uptime ~7540–7645 s. jw16 logged no ANE wedge lines at all.

## Artifacts

`receipts/2026-09-16-parakeet-e2e-both-hosts/`: `jwm1/` and `jw16/` each
hold `e2e-report-r1.json` (schema-1 report), `transcript.txt`
(`db501a8c…`), and the exact `run-battery.sh`; plus `collect-medians.py`
and `fold-runner-diff.txt`. Host-side: jwm1 `/var/tmp/E2EREV-battery/`
(out-r1..6, scratch), jw16 `/var/tmp/E2EREV-battery/`; worktrees
`/var/tmp/E2EREV`, sites `/var/tmp/E2EREV/site`.

## Not claimed

- No cold-SPIR-V mel numbers; r1 quoted separately, penalty unchanged from
  prior receipts.
- The < 8000 ms expectation existed only for the fold; the fold as pushed
  never produced correct output and is reverted — today's baseline is
  8541.8 ms (jwm1) / 6418.5 ms (jw16).
- No claim on macOS, other fixtures, other SoCs, or fold performance.
- `63c1d3cf` not merged, not touched.
