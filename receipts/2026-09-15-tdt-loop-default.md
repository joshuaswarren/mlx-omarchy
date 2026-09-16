# 2026-09-15 TDT loop as the default decode path (agent/tdt-loop-default)

Status: **default greedy TDT decode is the GPU-resident loop — token-exact on
both arms, 831 ms on the default run, host control kept behind `--tdt-host` /
`MLX_OMARCHY_TDT_HOST`. LAND.**

Base: origin/main `9b02870c` (the loop landed there as opt-in via harness
copies). Commit `63c1d3cf` untouched, not merged.

## Gates (jwm1-linux, Apple M1 G13G B1 / honeykrisp, flock -x /tmp/m1-gpu.lock)

| gate | result |
| --- | --- |
| e2e default run, **no flags** | 104/104 emissions, transcript sha `db501a8c080380ea…`, durations+frames match — **PASS** |
| e2e default timing | `tdt_decode` stage **831.331 ms** (< 1000 ms target; host leg 2956.209 ms) — **PASS** |
| e2e default routing | report `execution.control = "gpu-loop"`, `tdt_fallback_reason = null` — **PASS** |
| native arm default (fused_diagnose, no flags) | `first_divergence = null`, `decode_path = "gpu-loop"` — **PASS** |
| A/B `--tdt-host` | `control = "host"`, reason `--tdt-host`, still 104/104 match — **PASS** |
| unit (test_tdt_decode_path.py, 8 tests + test_tdt_control.py 3 tests) | **11/11 — PASS** (also green locally on x86) |

## What changed

`overlay/tools/coreml/parakeet_tdt.py` — `tdt_decode()` is the pipeline decode
entry. Default: `vulkan_tdt_loop.run_tdt_loop` (one dispatch, no per-emission
host sync). The host control loop (`greedy_tdt_decode`, unchanged) runs when:

* `--tdt-host` is passed (`force_host`) — the A/B and llvmpipe escape hatch;
* `MLX_OMARCHY_TDT_HOST` is set (repo `MLX_OMARCHY_*` env convention);
* the device lacks the loop's capabilities — `vulkan_tdt_loop.device_blockers()`
  names them from `mx.device_info()`: workgroup invocations/size < 1024,
  shared memory < 28768 B (the loop kernel's workgroup arrays), missing
  `shader_float16` / `storage_buffer_16bit_access`, missing limit axes
  (non-omarchy backend), llvmpipe/lavapipe, capability-simulation profiles;
* the loop kernel fails to launch — the exception is reported as the reason.

The chosen path and fallback reason ride on `TdtOutput.decode_path` /
`TdtOutput.fallback_reason` (defaulted, so existing constructors are
unchanged) and land in both harness reports (`execution.control`,
`free_decode.decode_path`) — a fallback can never pass silently: the report
names it, and the < 1000 ms stage gate exposes host-speed fallbacks.

`receipts/2026-09-15-tdt-gpu-loop/derivation/fused_e2e.py` and
`fused_diagnose.py` (the harness copies main carries) — `--tdt-loop` removed,
`--tdt-host` added, decode routed through `tdt_decode`.

`overlay/tests/omarchy/coreml/test_tdt_decode_path.py` — routing contract:
gpu-loop default, blocker/env/flag/host-launch-failure fallbacks each name
their reason, config validation precedes any dispatch.

## Notes

* `compare_vulkan_capture` (pinned capture comparator) stays on
  `greedy_tdt_decode` on purpose: it is the reference instrument whose schema
  pins `control: greedy_tdt_decode` and source hashes — that is exactly the
  A/B surface the host path is kept for.
* `device_blockers()` refuses llvmpipe/lavapipe by name even when the software
  rasterizer advertises sufficient limits; simulation profiles are refused
  like every evidence-producing surface.
* On jwm1 the real probe returns no blockers (`Apple M1 (G13G B1)`,
  honeykrisp) — proven by the default run actually taking the gpu-loop path.

## Artefacts

- `overlay/tools/coreml/parakeet_tdt.py`, `overlay/tools/coreml/vulkan_tdt_loop.py`
- `overlay/tests/omarchy/coreml/test_tdt_decode_path.py`
- `receipts/2026-09-15-tdt-gpu-loop/derivation/{fused_e2e,fused_diagnose}.py` (flipped)
- `receipts/2026-09-15-tdt-loop-default/derivation/` — gate wrappers
  (`run_e2e_default.sh`, `run_e2e_host_ab.sh`, `run_diag_default.sh`,
  `run_unit.sh`), gate evidence (`e2e-default-report.json`,
  `e2e-host-ab-report.json`, `token_ids.json`, `transcript.txt`,
  `fused_diagnose_out.json`, `unit-tests.log`)
- jwm1 stage: `/var/tmp/TdtLoopDefault/{pkg,derivation,e2e-after,e2e-host-ab,e2e-scratch,e2e-host-scratch}`

Resolved model: zai/glm-5.3-flash
