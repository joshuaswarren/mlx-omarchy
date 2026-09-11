# Decode attribution: hook root-caused, ablation table delivered

- schema: mlx-omarchy/decode-attribution-finish/1
- date_utc: 2026-09-11
- host: jwm1-linux (Apple M1, Honeykrisp, linux-asahi 7.1.6, glslc), coordinated GPU lock `/tmp/m1-gpu.lock`
- receipt-only: nothing merged into main. This branch (`receipts/decode-attribution-finish`, off main 1242cb32) carries the receipt; the code fixes live on jwm1 only (patch attached as `hook-fix.patch`).

## What this receipt completes

The PARTIAL receipt already on main (`receipts/2026-09-10-decode-attribution`, commit d1dfaf2b) documented a hook non-engagement defect and prescribed the root-cause procedure. Both prescribed steps are done:

1. **Root cause: suspect 1 (environment hand-off), exactly.**
   `run_arms.py` built the child env as `{k: v for k, v in os.environ.items() if not k.startswith("MLX_")}` and **never set `MLX_OMARCHY_ABLATE`**. The engine subprocess never saw the spec, so the ablate map was empty in every arm run. The manual smoke worked because the shell exported the variable.
   Fix: three lines in `run_arms.py` (`env["MLX_OMARCHY_ABLATE"] = arm` + debug opt-in for non-baseline arms). See `hook-fix.patch`.

2. **Suspects 2 and 3 refuted with one-line evidence** (getenv debug print at the `shader_bytes` entry point, wheel rebuilt `0f79507`+debug, single-class smoke leg through the scripted `run_arms` -> `bench_matrix` -> engine path):
   - `[ablate-debug] MLX_OMARCHY_ABLATE spec seen at first shader_bytes call, map_size=9` — getenv works at static-init (suspect 3 dead; the map is a function-local static built on first call, after the process env exists).
   - With `MLX_OMARCHY_ABLATE=all`: 9 distinct `[ablate-debug] kernel=<id> swapped` lines covering every ablated class, qmm and non-qmm — every kernel routes through `create_pipeline(kernel)` -> `shader_bytes`; the only bypass is the `cache_key`+spirv `Custom` path, which none of the ablated classes use (suspect 2 dead).
   - Engagement in the scripted path, same as manual: `smoke-fix/gemv-scripted.json` — short-decode-32 digest flips 7fd25a869ff21678 -> **8a86dd30276a7f38**, 112.5 -> **313.6 tok/s**. Full smoke in `smoke-fix/`.

## The attribution table (complete, 220/220 quiet legs)

Baseline and table describe **mlx-omarchy commit `b6d662a`** (release wheel `mlx_omarchy-0.32.2.dev202609101916+b6d662a`, ablation wheel `0f79507`+debug `dev202609110036`, libmlx.so sha256 68e477ed...). 12 interleaved reps per arm-workload (8 for non-baseline long legs), canonical digests asserted on every baseline leg, ablate digests flip per class. Per-leg records: `arms/legs.ndjson` (fields include arm, rep, workload, decode_tok_s, digest, peak_loadavg_1m).

```## short-decode-32
| arm | n | tok/s median | token ms |
|---|---|---|---|
| baseline | 12 | 112.705 | 8.8727 |
| gemv | 12 | 313.37 | 3.1911 |
| rope | 12 | 114.325 | 8.747 |
| rms | 12 | 116.495 | 8.5841 |
| kvwrite | 12 | 112.855 | 8.861 |
| attn | 12 | 114.02 | 8.7704 |
| swiglu | 12 | 113.155 | 8.8374 |
| sampler | 12 | 116.48 | 8.5852 |
| all | 12 | 395.895 | 2.5264 |

marginals (baseline - ablated), ms/token:
- gemv: 5.6816
- rope: 0.1257
- rms: 0.2886
- kvwrite: 0.0117
- attn: 0.1023
- swiglu: 0.0353
- sampler: 0.2875
- skeleton (all ablated): 2.5264
- sum(marginals)+skeleton: 9.0591 vs token 8.8727 -> residual -0.1864 ms
- skeleton minus 273x4.5us cadence: 1.2979 ms (host pacing + tail kernels + bodies)

## long-decode-128
| arm | n | tok/s median | token ms |
|---|---|---|---|
| baseline | 12 | 109.2 | 9.1575 |
| kvwrite | 8 | 109.53 | 9.1299 |
| attn | 8 | 111.195 | 8.9932 |
| all | 8 | 178.79 | 5.5933 |

marginals (baseline - ablated), ms/token:
- kvwrite: 0.0276
- attn: 0.1643
- skeleton (all ablated): 5.5933
- sum(marginals)+skeleton: 5.7852 vs token 9.1575 -> residual 3.3723 ms
- skeleton minus 273x4.5us cadence: 4.3648 ms (host pacing + tail kernels + bodies)

## longctx-1024-decode-32
| arm | n | tok/s median | token ms |
|---|---|---|---|
| baseline | 12 | 97.945 | 10.2098 |
| gemv | 8 | 151.425 | 6.6053 |
| rope | 8 | 98.86 | 10.1153 |
| rms | 8 | 99.33 | 10.0675 |
| kvwrite | 8 | 97.21 | 10.287 |
| attn | 8 | 101.77 | 9.8261 |
| swiglu | 8 | 97.18 | 10.2903 |
| sampler | 8 | 99.495 | 10.0514 |
| all | 8 | 155.655 | 6.4247 |

marginals (baseline - ablated), ms/token:
- gemv: 3.6045
- rope: 0.0945
- rms: 0.1423
- kvwrite: -0.0772
- attn: 0.3837
- swiglu: -0.0805
- sampler: 0.1584
- skeleton (all ablated): 6.4247
- sum(marginals)+skeleton: 10.6504 vs token 10.2098 -> residual -0.4406 ms
- skeleton minus 273x4.5us cadence: 5.1962 ms (host pacing + tail kernels + bodies)

## microbench cross-check (per token, ms)
| chain | dispatches | per-dispatch us (median) | minus floor | gpu chain ms est |
|---|---|---|---|---|
| gemv_layer | 48 | 72.658 | 68.158 | 3.2716 |
| harness | 24 | 40.221 | 35.721 | 0.8573 |
| kvwrite | 48 | 322.479 | 317.979 | 15.263 |
| lm_head | 24 | 1642.645 | 1638.145 | 39.3155 |
| rms | 49 | 23.796 | 19.296 | 0.9455 |
| rope | 48 | 36.272 | 31.772 | 1.5251 |
| sampler | 48 | 181.424 | 176.924 | 8.4923 |
| sdpa@seq1024 | 24 | 88.162 | 83.662 | 2.0079 |
| sdpa@seq16 | 24 | 30.838 | 26.338 | 0.6321 |
| sdpa@seq48 | 24 | 190.289 | 185.789 | 4.4589 |
| swiglu | 24 | 66.882 | 62.382 | 1.4972 |

```

## Reading of the table

- **GPU execution vs host pacing (short-decode-32):** baseline 8.873 ms/token. Sum of GPU-side class marginals (gemv 5.68 + rms 0.29 + sampler 0.29 + rope 0.13 + attn 0.10 + swiglu 0.04 + kvwrite 0.01) = **6.55 ms (74%)**. The all-ablated skeleton is 2.526 ms; minus the 273 x 4.5 us pre-recorded dispatch cadence (1.229 ms) leaves **1.30 ms of host pacing + tail kernels + no-work shader bodies**.
- **Unattributed residual, explicit:** sum(marginals)+skeleton vs measured token time = 9.059 vs 8.873 (**-0.186 ms, -2.1%** at short; -0.44 ms at longctx-1024). Negative residual means small class interactions/noise; per-class marginals are upper-bound-fair. At long-decode-128 only kvwrite/attn/all arms exist by design, so its 3.37 ms residual is **unattributed by design** (gemv/rope/rms/sampler/swiglu were never ablated at that workload), not a measurement gap.
- gemv dominates everywhere: 5.68 ms/token short (64%), 3.60 ms longctx.

## Per-row microbench cross-check (agreement/disagreement stated)

Cross-check instrument: serial dependent chains, release wheel, 30 reps (`microbench.json`). Its known harness floor is ~40-58 us/op eager wall cost vs the 4.5 us pre-recorded dispatch floor, so it can only resolve classes costing more than ~40 us/dispatch.

| class | ablation marginal | microbench | verdict |
|---|---|---|---|
| gemv | 5.68 ms/token short = 58.6 us/disp (97 disp) | 72.7 us/disp (48-disp layer chain, 68.2 minus floor) | **AGREE** within ~15-20%; ablation slightly lower, consistent with the microbench double-counting part of the eager host floor |
| attn | 0.10 ms short (24 disp) growing to 0.38 ms at ctx1024 (16 -> 38 us/disp) | sdpa@seq16 26 us/disp -> @seq1024 84 us/disp | **AGREE on direction and context scaling**; ablation values are lower bounds, microbench is floor-inflated |
| rms | 0.29 ms = 5.9 us/disp | 19.3 us/disp | **DISAGREE in magnitude, explained**: microbench is harness-floor dominated; ablation removes only the payload, so rms real cost is <= 0.29 ms/token |
| rope | 0.13 ms = 2.6 us/disp | 31.8 us/disp | **DISAGREE in magnitude, same cause** as rms |
| swiglu | 0.04 ms = 1.5 us/disp | 62.4 us/disp upper bound (receipt already flagged unstable) | **CONSISTENT** — microbench was declared an upper bound; ablation is far below it |
| sampler | 0.29 ms/token (2 disp) | 176.9 us/disp, floor-dominated, "not resolvable" | **RESOLVED by ablation where microbench could not**: ~144 us/token attributable to logsumexp+argreduce |
| kvwrite | 0.012 ms ~ zero | 15.3 ms — INVALID (eager slice_update materializes a cache copy) | **AGREE by elimination**: the only valid statement about kvwrite remains "not cross-checkable eagerly"; the ablation now shows its marginal at these contexts is negligible |

## Incident: concurrent build polluted 134 legs (quarantined)

During window2, a sibling agent's `build-wheel.sh` (cc1plus storm, 1-min loadavg 3.3-6.7) ran on jwm1 without the GPU lock. Per directive, every leg whose window overlapped a loadavg sample >= 3.0 was quarantined (`*.polluted` / `*.invalid-hook-dead` on jwm1; the pre-fix invalid run's 48 legs also in `arms-invalid-run1/` here). Window3/4/5 re-ran the quarantined legs; final dataset is 220/220 legs with peak in-window loadavg < 3.0 recorded per leg. In-window samples of 1.5-1.9 during sequential-bench decay are self-paced and kept; the quiet gate (1-min loadavg < 1.0, 3 checks) preceded every window.

## Main-based rerun (partial, reported honestly)

Main moved twice mid-work (9737c36e -> d1dfaf2b -> 1242cb32). A main-based pair was built (`d1dfaf2b` release wheel `dev202609110220+d1dfaf2`; hook cherry-picked onto `d1dfaf2b` as `74f170e4`, ablate wheel `dev202609110217+74f170e4`) and its arms run was launched (window6d, ~37 legs measured at write time). Partial but decisive fact: **the d1dfaf2b baseline reproduces the canonical b6d662a digests exactly** (7fd25a869ff21678 / 4cc08910089477fd / 7da83f06ec9f001d) at ~111.5/108.0/97.3 tok/s — decode outputs are unchanged on main, so the b6d662a table above carries over to main modulo the known dispatch-count change (273 -> 249, kvwrite row shrinks, skeleton cadence 1.10 ms). The complete main-based table is NOT delivered here; window6d was stopped per wrap-up directive. Runner for it: `run_arms_main.py` (self-calibrating baseline digest pin) + `analyze_main.py` (249-dispatch cadence) on jwm1.

## Policy compliance

- never merged: fixes and instrument live on jwm1 (`wave/DecodeAttribution`, `wave/DecodeAttributionMain` at `74f170e4`); this branch is receipt-only.
- digest gates untouched: canonical digests asserted only on unmodified release wheels; ablate wheels used for measurement only.
- receipts: every claim above maps to a file here (`table.md/json`, `arms/legs.ndjson`, `smoke-fix/`, window logs, `hook-fix.patch`) or on jwm1 under `~/src/mlx-DecodeAttribution/receipts-workdata-20260910/` (quarantined raw legs, window6 logs).
