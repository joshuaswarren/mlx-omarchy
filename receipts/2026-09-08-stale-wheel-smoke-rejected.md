# Stale-wheel GPU microbench smoke on jwm1-linux -- REJECTED (2026-09-08)

## Status: REJECTED. Numbers in this receipt are invalidated.

The GPU smoke on jwm1-linux that produced /tmp/gpu_bench_smoke.json used
mlx-omarchy 0.32.2.dev202609032000+18f59e3 (the Sept3 wheel, installed
in venv-bqm1-ship). That wheel pre-dates c2548675 and predates the
addition of qmm_vec and qmm_coopmat GPU kernels to the Omarchy Vulkan
backend. The "4-bit quantized_matmul falls back to CPU stream, ~104 ms
per projection" observation that came out of the smoke is an artifact of
that stale wheel, not a current fact about the integrated baseline. On
c2548675 the GPU kernels exist and the 4-bit quantized_matmul path runs
on the GPU device.

Likewise the eager-interpreter fp16/bf16 GEMV medians in the smoke are
not valid current numbers -- the smoke ran with the eager interpreter
because the Honeykrisp tape-compile defect is still unpinned in the
Sept3 wheel; the integrated baseline owner measures the verified
tape-compile path. This receipt does not present those numbers because
publishing them as current would be wrong.

## Exact UTC timestamps of the rejected smoke

- Script copied to M1: /tmp/gpu_bench.py mtime 2026-09-08 11:33:02.295
  local (-0500) = 2026-09-08 16:33:02.295Z UTC.
- Smoke output written: /tmp/gpu_bench_smoke.json mtime 2026-09-08
  11:33:10.476 local (-0500) = 2026-09-08 16:33:10.476Z UTC.
- Smoke wall time: 8.34 sec. Window: 2026-09-08T16:33:02Z to
  2026-09-08T16:33:11Z UTC.

## Lock state during the smoke

- /tmp/m1-gpu.lock existed as an empty file (created 2026-09-08 09:57
  local = 2026-09-08 14:57:00Z UTC). At 2026-09-08 16:36:02Z UTC
  (after the smoke completed), `lsof /tmp/m1-gpu.lock` and `fuser
  /tmp/m1-gpu.lock` returned empty, confirming no flock holder.
- ParityBaseline's earlier message "machine is not yet mine; I'll
  announce claim and release on this channel" preceded the smoke. No
  claim announcement arrived before the smoke started. From this side,
  no overlap with baseline activity is visible in the timestamp/lock
  evidence above.
- The smoke was outside the shared lock. ParityBaseline was DM'd
  directly to confirm whether any baseline activity ran during
  16:33:02Z-16:33:11Z UTC so that any baseline numbers from that
  window can be invalidated on their side as well.

## What remains valid from this session

- Compilation provenance for the ANE candidates (omarchy-ane 96f5ea3,
  mil-hwx-compiler ane-parity @ 42fd0bb). No measurements in those
  receipts; only program counts, BLOBFILE format, and refusal modes.

## Plan

- Re-run the GPU microbench on the c2548675-built wheel after
  ParityBaseline's release. Same case list, 3 warmups + 30 samples,
  host->host transfer, no forced CPU stream.
- Re-run the ANE b1/b2 timed measurements via the lifecycle6
  benchmark-packages.py harness in the same shared window.
- Until paired-hardware numbers land, no published ANE-vs-GPU comparison.
