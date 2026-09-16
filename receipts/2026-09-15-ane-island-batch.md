# ANE island batch: one deadline-bounded submit per encoder pass (2026-09-15)

Verdict: **LAND.** The 72 per-layer island submits of one encoder pass became
**1 batch submit** (72 lockstep rounds) on one resident worker process, with a
row-wise tile pack/unpack replacing the per-element scatter/gather. On jwm1,
same lock window, same boot, all arms byte-identical to the frozen pin
`38c73261…`, and the full E2E is **104/104 tokens_match, transcript sha
`db501a8c…`** with the batched path end to end.

## Before / after (jwm1-linux, one `flock /tmp/m1-gpu.lock` window, never steal)

| arm | submits | worker starts | ane exec ms | encoder wall ms | encoder_hidden |
| --- | ---: | ---: | ---: | ---: | --- |
| before: launch per submit (pinned stage runner `de633dc4`, worker `762dd1de`) | 72 | 72 | 4943 | 19593 | `38c73261` |
| + row pack only (my runner, launch mode, my worker) | 72 | 72 | 2997 | 17553 | `38c73261` |
| + resident batch (my runner `ANE_ISLAND_MODE=resident-batch`, my worker) | **1** | **1** | **2653** | **16394** | `38c73261` |

ane_worker_exec −46% (4943 → 2653 ms), encoder wall −3.2 s (19593 → 16394 ms),
submits 72 → 1. Per-round means inside the batch: island-attn-a-kt 53.6 ms,
island-select 40.5 ms, island-pv 16.5 ms (launch before: 90.0 / 78.4 / 41.3).

Batch open cost 24 ms once (bundle parse + program load); rounds are the only
recurring cost, which is exactly what the batching was for.

## What changed (branch `agent/ane-island-batch`, commit `957cd21a`)

- `tile_layout.h`: `ane_pack_rows` / `ane_unpack_rows` — per-row memcpy
  (one memcpy when the tile is fully dense), same byte placement as the
  per-element reference; partial tails fall back per element.
  Host-proven: the per-element pack measured ~50-80 ms per island submit.
- `worker.{h,cpp}`: batch scope. `open_batch(deadline)` fixes one absolute
  deadline; `submit()` inside the scope is bounded by it and counts rounds;
  `close_batch()` closes the scope. Deadline miss or failed round still
  quarantines and ends the session — the bounded unit is the batch, the
  safety model is otherwise unchanged (one private serve child, no retry).
- `mlx-omarchy-ane-worker/main.cpp`: `batch DEADLINE_MS` / `batch-end`
  serve lines, named refusals (exit 64) for misuse.
- `ane_resident.py`: `begin_batch` / `end_batch`, client guard extended to
  the batch deadline while a scope is open, counters `batch_opens`/`batch_rounds`.
- `vulkan_encoder.py`: `ANE_ISLAND_MODE=resident-batch` opts AneIsland into
  the batched resident path (launch stays default and byte-compatible);
  report gains `mode`/`rounds`/`batch_open_ns`; `close()` releases the session.

## Tests

- jwm1 `omarchy_ane_worker_tests`: **16 cases / 199 assertions, SUCCESS**
  (new: row pack vs per-element reference on contiguous / row-padded /
  plane-padded / bool / truncated bindings; batch scope happy path, deadline
  miss quarantine, misuse refusals).
- Python client suite `test_ane_resident`: **10/10 OK**
  (new: batch scope happy path + refusal naming against the fake worker).

## E2E gate (jwm1, real pipeline, batched runner + batch worker)

| quantity | value |
| --- | --- |
| status | **match** |
| emissions | **104/104**, first_divergence null |
| transcript_match | true, sha256 `db501a8c080380ea027ffa50a4b4956c39df77cb692c4fb78e556311a11a0790` |
| encoder_hidden sha256 | `38c73261f29230276ed76f1fc017b76b024156d79218bd5f1347fdc7e7d43ec7` (pin, unchanged) |
| ane | mode resident-batch, **1 submit, 1 worker start, 0 timeouts**, exec 2671 ms |
| pipeline | 29453 ms (launch arm same boot: ~30470 ms with old worker) |

Worker binary `mlx-omarchy-ane-worker` sha256 `c64a5e75c959a08d0df41f874a870af37a300a9f871290fcc5f9da1ad5cedbdd`
(see "Bundle parser conflict" for what it contains). Runner staged at
`/var/tmp/AneIslandBatch/stage-runner/vulkan_encoder.py`
(`240e3b63e1a3d190713e1fa5c05e092cfca2b4951004462ec43fdb06e17260ea`).
Artifacts: `receipts/2026-09-15-ane-island-batch/artifacts/` (three run
reports + e2e-report); build tree `/var/tmp/AneIslandBatch/` on jwm1.

## Bundle parser conflict (finding for Main — blocks origin/main builds of this path)

Origin/main's bundle contract (`fb4dfa86` "refuse positional channel maps")
**refuses the pinned island bundles**: loading `island-pv` from a strict
origin/main build fails with "task stream does not name every surface". The
bundles predate that gate and their channel maps are hardware-proven by the
pins, but no origin/main-built worker can run the pinned encoder E2E until
either the bundles are re-exported to the derived-channel contract or the
gate gains a pinned-bundle allowance. The measurement binary above therefore
carries the pre-`fb4dfa86` `bundle.cpp` (identity untouched — pins reproduce);
**my branch does not touch `bundle.cpp`**, so landing it changes nothing about
that decision. Ticket `EncoderWallAttribution` and any origin/main wheel
rebuild will hit the same refusal.

## Not claimed

- No perf claim beyond the same-window arms above; no thermal soak.
- The batch deadline is 120 s by default (`ANE_ISLAND_BATCH_DEADLINE_MS`): a
  wedged round burns up to that long before quarantine, by design, since the
  bounded unit is the pass.
- E2E was run twice in launch mode (same pins) and once batched; the batched
  run is the recorded gate.
- `63c1d3cf` not merged.

resolved_model: this session (`zai/glm-5.3-flash`).
