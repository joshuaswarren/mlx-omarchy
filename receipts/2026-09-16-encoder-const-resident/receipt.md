# Encoder const-resident cache (opt-in): warm same-process passes drop
# ~1.05–1.13 s; default single-pass E2E unchanged (2026-09-16)

Attack on the remaining ~1.04 s warm-run host residual priced by `receipts/
2026-09-16-encoder-host-residual.md`, of which ~0.7–0.84 s was `eval_const`
materializing the 1977 const statements (the 1.2 GB of fp16 weights) on
every encoder pass. Verdict: **LAND (opt-in, default off)** — identity and
correctness gates all green (pin exact on every leg, E2E 6/6 = 104/104,
transcript exact, no leak). An in-process cache cannot cut the E2E: the
pipeline calls run() once per process, and the cache-on default cost
+235.9 ms median r2–r6 there, so Main dispositioned the cache opt-in
(default off): single-pass E2E default unchanged, multi-pass consumers
set the knob and save −1050..−1130 ms per warm pass.

## What landed

`overlay/tools/coreml/vulkan_encoder.py`, on top of `0101d673` (origin/main;
the base runner file is byte-identical to the fold-reland staged runner
`57fb19c5…`):

1. `EncoderRunner.const_values` / `const_meta` — after `_eval_const`
   materializes a const (blob read + `mx.array` upload), the runner keeps
   the device-resident array referenced. The cache is instance-local:
   entries are produced only by THIS runner's parse of THIS runner's blobs,
   so a different model constructs a different runner with its own cache and
   no stale-byte path exists. Blob identity rides the statement's own
   attrs (path, offset, byte count, dtype, shape) already parsed into
   `Statement.const_kwargs` by the host-residual receipt's precompute —
   attrs are never re-parsed.
2. `run()` warm-pass path — second and later passes restore every const
   with a dict insert (buffer re-reference, zero upload, zero blob read),
   keep const statements `done`, and reset `done` on every non-const
   statement so the graph re-executes identically. Before this change a
   second `run()` on the same runner was impossible: `Statement.done` was
   never reset, so the loop re-executed nothing and raised `never produced
   [...]` (observed on the first repeat probe).
3. `MLX_OMARCHY_ENCODER_CONST_CACHE` (opt-in, **default off** per Main's
   disposition below) — when on, later passes re-reference the cache;
   when off (default) every pass re-materializes at the pre-cache cost,
   and single-pass pipelines see no change from this edit.
4. `--repeat N` on the standalone leg: N passes in one process, per-pass
   wall, `encoder_hidden` sha256, Vulkan dispatch deltas, `mx` active/peak
   memory and RSS. Default 1 keeps the single-pass report shape.

Release semantics are unchanged: `run()` still deletes values at last use,
including consts; the cache is what keeps the device buffers alive through
those deletions and serves the next pass.

## Mechanism fact: where the win can and cannot appear

`fused_e2e.py` constructs the runner and calls `run()` exactly once per
process; the E2E battery is six separate processes. A fresh process must
upload the 1.2 GB once regardless of any in-process cache — there is no
cross-process device cache on this stack. The const cache therefore pays on
same-process warm passes only. Measured, it is a large win there
(−1050..−1130 ms) and a small regression on fresh-process passes (below).

## Why fresh-process passes regress slightly

With the cache resident, the MLX allocator cannot recycle the
progressively-freed const buffers for intermediates. Knob-off passes free
~1.2 GB of const buffers into the pool during the pass and reuse them
(peak live device memory 250.5 MB); cache-on passes hold 1157.7 MB plus
allocate intermediates on top (peak 1374.8 MB, fresh allocations early in
the pass). On a multi-pass process the pool warms and warm passes run at
~6.05 s; a fresh process pays the allocation cost once, ~+100..+300 ms.
This is the price of residency, not a leak (RSS is flat across passes).

## Standalone evidence (jwm1 T8103, ANE islands ABC, resident-batch)

After runner `e7ecb83c…`, before runner `57fb19c5…` (= `0101d673`). Pin
`38c73261f29230276ed76f1fc017b76b024156d79218bd5f1347fdc7e7d43ec7` on
every leg below, cold and warm, both arms, all passes.

| leg | passes | wall ms per pass | active MB | peak MB | RSS MB |
| --- | --- | ---: | ---: | ---: | ---: |
| after, cold process ×2 | 1/1 | 7221.1, 7176.8 | 1157.7 | 1347.9 | 1277.8, 1274.6 |
| after, `--repeat 4` | 1 | 7167.6 | 1157.7 | 1347.9 | 1276.8 |
| | 2 | **6042.5** | 1157.7 | 1374.8 | 1268.1 |
| | 3 | **6110.2** | 1157.7 | 1374.8 | 1268.1 |
| | 4 | **6046.0** | 1157.7 | 1374.8 | 1268.1 |
| before, separate process ×3 | 1/1 | 6986.0, 7118.2, 6821.7 | — | — | — |
| after, knob off, `--repeat 2` | 1 | 7057.6 | 2.4 | 250.5 | 1262.1 |
| | 2 | 6734.4 | 2.4 | 251.4 | 1279.8 |

- Warm same-process pass vs its own cold pass-1: **−1121.5 ms** (7167.6 →
  6046.0, median warm 6046.0 vs before-arm single-pass median 6986.0 →
  **−940 ms cross-arm**).
- `vk_compute_dispatches` per pass: 2885, 2885, 2885, 2885 (identical —
  the compute program stream does not change); `gpu_primitive_dispatches`
  5696 pass-1 vs 5328 warm passes (the const upload/copy primitives
  disappear).
- Knob off: both passes pay full materialization, 2.4 MB resident, and
  dispatch counts match the before arm (2885/2887 compute) — the switch
  reproduces pre-cache behavior per pass.
- `--no-ane --repeat 3` control (GPU-only hash `e832110d…`, islands off):
  2733.7 / 2436.9 / 2449.4 ms, hidden identical all passes.
- Leak watch (gate 4): `--repeat 4` RSS 1276.8 → 1268.1 → 1268.1 →
  1268.1 MB, active flat 1157.7 MB. Literal 6-pass leg: see repeat-6
  table below.

cProfile on a before-arm warm leg (7.284 s under profiler): `eval_const`
0.841 s cumulative over 1977 calls (0.465 s own), `Blobs.read_bytes` 0.334 s
over 740 blob consts, `apply` 0.106 s, `_parse` 0.065 s — matches the
unprofiled −1121 ms warm-pass delta (profiler distorts; the wall is the
claim).

## E2E battery (6 runs, no flags, after runner)

Harness `fused_e2e.py` `0e38e7b1…`-era environment identical to the
fold-reland battery: non-diag wheel site `/var/tmp/E2EREV/site`
(`mlx_omarchy-0.32.2.dev202609161313+f43ab71`), worker
`f171a61e…`-era path, libane-strict `56b46234…`, bundles
`/var/tmp/island-reexport/bundles`, model `b650695c…`,
`ANE_ISLAND_MODE=resident-batch`, fresh SPIR-V cache dir.

All 6 runs: `status=match`, emissions 104/104 with matching_prefix 104,
`mel_bit_exact=True`, encoder `all_bounds_pass=True`, `control=gpu-loop`,
`tdt_fallback_reason=null`, `cpu_tensor_events=0`, island 1 submission /
1 worker start / 0 timeouts. Transcript `db501a8c080380ea…0790`.
`encoder_hidden` pin-exact in all 6.

Same-boot-era A/B, after battery at 09:45, before battery at 09:50:

| arm | encoder_ane r1..r6 (ms) | median r2–r6 | median r1–r6 |
| --- | --- | ---: | ---: |
| before (`57fb19c5…`) | 6953.9, 6730.3, 7001.1, 6945.8, 7034.8, 6980.6 | **6980.6** | 6967.2 |
| after (`e7ecb83c…`) | 7182.7, 7216.5, 7259.4, 7052.3, 7339.6, 5961.2 | **7216.5** | 7199.6 |

Delta: **+235.9 ms median r2–r6** (fresh-process residency cost, mechanism
above). r6 is an outlier in the fast direction (ANE exec 1997 ms vs
2475–2626 on the other runs — machine variance on the ANE leg, seen in
both directions today). Knob-off E2E legs with the same after runner
(below) land on the before-arm numbers, pinning the delta on the cache
residency rather than the runner edit.

## Kill-switch and leak legs

Knob-off E2E legs, after runner `e7ecb83c…` with
`MLX_OMARCHY_ENCODER_CONST_CACHE=0`, run at 09:54 between the after and
summary batteries: `7076.0 / 6987.6 / 6995.2` ms encoder_ane, median
**6995.2** — on the before-arm numbers (6980.6), not the cache-on numbers
(7216.5). All three `match`, prefix 104, bounds pass, cte 0, pin exact.
The E2E delta is attributable to the cache residency, not to the rest of
the runner edit.

Leak leg, `--repeat 6` one process (gate 4):

| pass | wall ms | active MB | peak MB | RSS MB | hidden |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 7198.6 | 1157.7 | 1347.9 | 1275.8 | pin |
| 2 | 6194.3 | 1157.7 | 1374.8 | 1265.8 | pin |
| 3 | 6236.3 | 1157.7 | 1374.8 | 1268.0 | pin |
| 4 | 6199.2 | 1157.7 | 1374.8 | 1265.8 | pin |
| 5 | 6104.4 | 1157.7 | 1374.8 | 1268.0 | pin |
| 6 | 6067.4 | 1157.7 | 1374.8 | 1282.7 | pin |

RSS flat (±14 MB wiggle, no trend), active/peak flat from pass 2 — no leak
growth across 6 runs. Warm passes 2–6: median 6199.2 ms, best 6067.4.

## Disposition (Main, 2026-09-16): opt-in, default off

Option (b): the cache is opt-in for multi-pass-per-process consumers —
warm pass −1050..−1130 ms, break-even at 2 passes per process — and the
single-pass E2E default is unchanged (the +235.9 ms residency cost is
avoided; the knob-off E2E median 6995.2 ms pins the no-delta claim against
the before arm's 6980.6 ms). `CONST_CACHE_ENABLED` default flipped to off;
the cache legs in this receipt were run with the knob explicitly on and
remain the evidence for opt-in consumers. Default-path re-verify below.

## Default-path E2E re-verify (after the default flip)

3 runs, no flags, default runner `df7e4ba9…` (cache off by default, no env
vars), 10:01–10:02, lock inode 35 held/freed, `LOCK-FREE-OK`:

| run | status | emissions | prefix | mel | bounds | cte | fallback | ane timeouts | encoder_ane ms | hidden | transcript |
| --- | --- | ---: | ---: | --- | --- | ---: | --- | ---: | ---: | --- | --- |
| d1 | match | 104/104 | 104 | exact | pass | 0 | null | 0 | 7025.5 | pin | exact |
| d2 | match | 104/104 | 104 | exact | pass | 0 | null | 0 | 7015.0 | pin | exact |
| d3 | match | 104/104 | 104 | exact | pass | 0 | null | 0 | 6987.3 | pin | exact |

Default-path encoder_ane median **7015.0 ms** — on the before-arm band
(6980.6 median era), confirming the default carries no regression.

## Identity

- Host jwm1-linux, aarch64, kernel `7.1.6-1-1-ARCH`, `/dev/accel/accel0`,
  Mesa Honeykrisp Vulkan, Apple M1 (T8103).
- Lock `/tmp/m1-gpu.lock` inode 35 throughout, `flock -w 900`, never
  stolen, never unlinked; `flock -n` free after every battery.
- Runner arms staged on jwm1: before `57fb19c5…` (= `0101d673` content,
  fold-reland staged bytes), cache-on `e7ecb83c…` (first commit, ran every
  cache leg), default `df7e4ba9…` (opt-in flip, ran the re-verify);
  `ane_resident.py` `1972bc80…` (same bytes the hostres/fold batteries
  used).
- Artifacts on jwm1: `/var/tmp/enc-const-resident/` (all runner arms,
  `gate-jwm1.sh`, `gate-jwm1-e2e.sh`, `gate-jwm1-before-e2e.sh`,
  `gate-jwm1-knob.sh`, `gate-jwm1-default.sh`, `verify*.log`, per-leg
  out/scratch dirs).
- Artifacts in this receipt dir: the five gate scripts.
- Worktree `~/src/mlx-omarchy-enc-const`, branch `agent/encoder-const-resident`.

## Not claimed

- No E2E encoder_ane improvement; measured +235.9 ms median r2–r6 from
  residency (see mechanism section). The ticket's −400..−700 ms E2E
  expectation cannot be produced by an in-process cache against a
  one-pass-per-process pipeline; the realized win is −1050..−1130 ms on
  same-process warm passes (standalone `--repeat`, or any future
  multi-pass pipeline).
- No claim on the ANE submit path internals, the coopmat/fold GPU buckets,
  cold-SPIR-V walls, macOS, jw16, other fixtures, or the TDT leg.
- `63c1d3cf` not merged, not touched.
- Landing: handed to LandHostLeaves after the default-path re-verify;
  the cache is opt-in (default off) per Main's disposition.
