# 2026-09-03 — 8-core re-measurement of the README host-bound rows on jwm1

Follow-up to the one-core annotation added today to `README.md` and the
affected receipts. Between 2026-08-25 and 2026-09-03 every jwm1-linux
host-bound timing was taken with the GRUB default on `Omarchy ANE test`
(static `/boot/ane.dtb`, CPU1-7 offline). The machine now boots all 8
cores (`nproc` = 8, online = 0-7, up 19 min, load < 1 during all runs).
This receipt re-measures the README tok/s rows under the same protocol
and records the result, which did **not** improve.

## Protocol (identical to the original receipts)

Model: `mlx-community/Qwen2.5-0.5B-Instruct` at the same pinned
revisions (`a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3` 4-bit,
`56d07e766edd7159fbe12ed12d9cf114bf38bf1e` bf16), same prompts,
`--max-tokens 32 --temp 0 --seed 0`, chat template default,
`MLX_DISABLE_COMPILE=1` (same gate the Vulkan legs always ran), warm run
reported, cold run shown. 5 runs per leg; medians below.

Wheel caveat, stated up front: the recorded README numbers were measured
on commit `ceae628`. The wheel available today is the released
`mlx_omarchy-0.32.2.dev20260903` (the one `~/venv-community` ships). So
this re-measure holds model/prompt/seed/token-count fixed and uses the
current released wheel; the core count and the wheel are both different
from the recorded run. See "Reference-build attempt" for why the
ceae628 wheel could not be used to separate the two.

Exact commands (on jwm1, fresh venv `~/venv-coreaudit` with the released
wheel + `mlx-lm==0.31.3 --no-deps`):

```
MLX_DISABLE_COMPILE=1 python -m mlx_lm generate \
  --model <snapshot-dir> --prompt "Hi" \
  --max-tokens 32 --temp 0 --seed 0          # bf16 leg
MLX_DISABLE_COMPILE=1 python -m mlx_lm generate \
  --model <snapshot-dir> \
  --prompt "What is the capital of France? Answer in one word." \
  --max-tokens 32 --temp 0 --seed 0          # 4-bit leg
```

## Results

| Metric | recorded (1 of 8 cores, ceae628) | re-measured 8-core median (dev20260903) | change |
|---|---|---|---|
| bf16 prefill tok/s | 23.9 | 17.095 (5 runs: 17.306, 17.095, 16.462, 17.148, 17.112) | **-28.5%** |
| bf16 decode tok/s | 3.56 | 2.040 (runs: 1.966, 2.040, 2.054, 2.023, 2.057) | **-42.7%** |
| 4-bit prefill tok/s | 25.3 | 18.389 (runs: 18.385, 18.389, 18.228, 19.700, 18.576) | **-27.3%** |
| 4-bit decode tok/s | 6.46 | 3.882 (runs: 3.892, 3.882, 3.751, 4.111, 3.794) | **-39.8%** |
| peak memory | 0.993 / 0.292 GB | 0.993 / 0.292 GB | unchanged |

Every row moved by far more than a few percent, and all of them moved
**down**, not up. Generated text was identical in every run (`Hello! How
can I assist you today?` bf16, `Paris` 4-bit).

## A/B diagnostics

- `taskset -c 0` (1 core, same released wheel, bf16): prefill
  **22.162** tok/s, decode 1.987. The prefill drop therefore tracks the
  core count: 1 core on this wheel lands within ~7% of the recorded
  23.9, while 8 cores is 28% below it. Host-side work on this stack
  appears to run **slower** with all 8 cores online than with one.
- The decode gap does not track the core count (1.987 on 1 core is as
  far from the recorded 3.56 as 2.040 on 8 cores is). Decode moved with
  the wheel, not the core count.

## Reference-build attempt (excluded as invalid)

To separate wheel from core count, `ceae628` was prepared
(`scripts/prepare-mlx.sh`) and built into a wheel
(`mlx_omarchy-0.32.2.dev202609031328+ceae628`) on the now-8-core
machine. Results were pathological and unreproducible against either
recorded or current-wheel numbers: bf16 prefill 13.0 tok/s, bf16 decode
6.4, 4-bit prefill **0.111** tok/s (~90 s/token), and 3.3 tok/s under
`taskset -c 0`. The machine was idle (load < 1) throughout. This build
is not equivalent to whatever produced the recorded ceae628 numbers and
is excluded from every table above. The honest conclusion: with the
artifacts available today, the wheel and the core count cannot be fully
separated for the prefill rows beyond the `taskset` A/B above.

## Not re-measured here

The README's "where the remaining time goes" wall-share percentages come
from the instrumented dispatch profiler
(`receipts/2026-09-02-gpu-profile-decode.md`) and need that profiling
build, not the released wheel; they are not re-run in this receipt.

## Condition of the numbers

Recorded 1-core figures stay in place with their annotation. The README
performance table now carries the 8-core column from this receipt.
