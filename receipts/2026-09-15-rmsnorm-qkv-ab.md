# 2026-09-15 Attention-scoped RMSNorm GEMV fold (qkv knob) — 12-round interleaved A/B on jw16 and jwm1

## Verdict

**NO-LAND.** The land rule required the ctx1053 decode median to rise on
both hosts AND the short median to fall no more than 1% on either host.
The attention-scoped arm passes the ctx gate only on jw16 and breaches
the short floor on both hosts; on jwm1 it loses both legs. The branch
default stays the full fold (154 dispatches); the "≤154 relaxed to 178"
question is moot — no evidence supports defaulting the knob.

| host | leg | base median tok/s | qkv median tok/s | delta | gate |
|---|---|---|---|---|---|
| jw16 (jw16mbp1-linux) | short 30/32 | 190.7190 | 185.6778 | **−2.64%** | FAIL (floor −1%) |
| jw16 (jw16mbp1-linux) | ctx1024 1053/32 | 135.5372 | 145.6476 | **+7.46%** | pass |
| jwm1 (jwm1-linux) | short 30/32 | 116.3445 | 108.5062 | **−6.74%** | FAIL (floor −1%) |
| jwm1 (jwm1-linux) | ctx1024 1053/32 | 102.3498 | 93.7904 | **−8.36%** | FAIL (needs rise) |

12-round interleaved batteries, arm order alternated per round, one
`bench_decode.py` process per leg, every leg's generated-ID digest
asserted against the pinned pair (short `7fd25a869ff21678`, ctx1024
`7da83f06ec9f001d`). Pins held on all 48 legs on each host.

## Arms

- **base** = a21b3c81-equivalent, fold off, 201 dispatches.
  - jw16: wheel `+a21b3c81` (dist-cand).
  - jwm1: wheel `+7e64e41c` — code-identical to a21b3c81: the only files
    differing between the two commits are under `receipts/`
    (`git diff --name-only 7e64e41c a21b3c81 | grep -v '^receipts'` is
    empty; the `mlx/backend/omarchy` diff is empty). Dispatches
    re-measured at 201 on both hosts.
- **qkv** = branch tip c3553eb2 with `MLX_OMARCHY_FUSED_GEMV_RMSNORM_MLP=0`
  (attention-only scope), 178 dispatches on both hosts.
  - jw16: wheel `+c2fc9246` — code-identical to c3553eb2 (the only diff
    is `receipts/2026-09-15-rmsnorm-gemv-fix.md`).
  - jwm1: wheel built fresh at `+c3553eb2`
    (`mlx_omarchy-0.32.2.dev202609160123+c3553eb2`, sha256 `02cc2909…`).
- Dispatch counts measured per host before each battery
  (`MLX_DISABLE_COMPILE=1`, 8 tokens): base 201, qkv knob=0 178, qkv
  default (full fold) 154 on both hosts — the arm definition in the
  ticket holds exactly. Wheel provenance verified per leg by
  `bench_decode.py`'s refusal protocol (`verified=match`).

## Protocol

- Harness: `ab_decode.py` (same script family as the 2026-09-15
  rmsnorm-gemv-fix lane), `--rounds 12`,
  `--env MLX_OMARCHY_FUSED_GEMV_RMSNORM_MLP=0` applied to every arm (the
  variable does not exist in base code and is ignored there).
- `MLX_DISABLE_COMPILE=1 HF_HUB_OFFLINE=1`, `--tokens 32 --temp 0.0
  --seed 0 --warmup-tokens 4`; prompt-token counts asserted 30 (short) /
  1053 (ctx1024) per leg.
- One `flock /tmp/m1-gpu.lock` hold per host for the whole battery (the
  script's nested `flock -n` check confirms the hold); lock inode
  unchanged at exit; never unlinked. jwm1 waited out the shared-host
  lock via `flock` — no stealing.
- Model: `mlx-community/Qwen2.5-0.5B-Instruct-4bit` snapshot
  `a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3` (same snapshot on both
  hosts — this is what makes the pins portable).
- Windows: jw16 01:20:22Z→01:22:06Z, jwm1 01:29:11Z→01:31:35Z
  (2026-09-16 UTC timestamps in started/finished).

## Per-round values

jw16 (`jw16/ab.json`):

- short base: 190.22 190.91 189.59 190.85 191.06 190.32 190.51 190.74 190.70 189.84 190.95 190.85 → median 190.7190
- short qkv: 186.05 185.60 186.01 185.60 186.23 182.50 184.00 185.72 186.01 185.50 185.64 185.95 → median 185.6778
- ctx base: 134.34 137.60 132.98 134.40 142.26 146.19 131.33 129.75 124.54 140.78 149.03 136.68 → median 135.5372
- ctx qkv: 137.33 148.13 131.04 142.63 148.78 145.83 149.25 145.46 130.03 156.25 157.39 143.42 → median 145.6476

jwm1 (`jwm1/ab.json`):

- short base: 114.36 119.85 114.37 113.87 113.41 119.72 115.68 119.98 119.56 114.34 117.01 119.84 → median 116.3445
- short qkv: 108.54 108.54 108.61 108.57 105.19 108.66 108.13 108.47 108.37 108.32 108.25 108.66 → median 108.5062
- ctx base: 96.96 102.79 103.62 99.32 98.25 102.39 102.59 102.31 102.79 99.70 103.01 102.18 → median 102.3498
- ctx qkv: 93.40 93.89 93.73 94.11 91.74 89.51 91.17 93.92 91.75 94.21 94.31 93.85 → median 93.7904

## Reading

1. **jw16 short loss is real and reproduces.** −2.64% today vs −2.5% in
   the gemv-fix receipt's 5-round battery: the attention-scoped trade
   costs ~5 tok/s at short on jw16, consistently outside the 1% floor.
2. **jwm1 rejects the trade outright.** Both legs lose, with disjoint
   clusters: short qkv sits in a tight 105.2–108.7 band against base's
   113.4–120.0 (median −6.74%); ctx qkv 89.5–94.3 against base's
   97.0–103.6 (median −8.36%). The interleaved design cancels
   window-level drift, so this is arm effect, not contention.
3. **The jw16 ctx win does not travel.** The gemv-fix receipt's "+10%
   ctx median" (5-round, jw16-only, wide spread) repeated directionally
   on jw16 (+7.46%) but inverts to −8.36% on the base M1. Scoping the
   prologue fold to the attention GEMV groups is host-specific: the
   112/144-workgroup attention GEMVs absorb the per-workgroup reduction
   on jw16mbp1, not on jwm1's smaller GPU.
4. **Decision:** NO-LAND under the written rule. No knob-default change;
   the ≤154→178 relaxation Main floated is not supported by evidence.

## Artifacts

- jw16: `receipts/2026-09-15-rmsnorm-qkv-ab/jw16/` — `ab.json` (48 legs,
  per-round values + medians; sha256 prefix `1ee480f9c4d3ed75`),
  `run-qkv-ab.sh`, `dispatch-{base,qkv-off,qkv-default}.json`,
  host/lock/started/finished. Host copies:
  `/var/tmp/SwigluRmsJw16/jw16-out-qkv/`.
- jwm1: `receipts/2026-09-15-rmsnorm-qkv-ab/jwm1/` — `ab.json` (sha256
  prefix `da09dfeb34a40f83`), `run-qkv-ab.sh`, `build-qkv.sh`,
  `dispatch-{base,qkv-off,qkv-default}.json`,
  host/lock/started/finished. Host copies:
  `/var/tmp/SwigluEpilogue/out-qkv/`; qkv wheel
  `/var/tmp/SwigluEpilogue/dist-qkv/mlx_omarchy-0.32.2.dev202609160123+c3553eb2-cp314-cp314-linux_aarch64.whl`.
