# Window3 verification results — fix wheel 69a01db3

Wheel: `mlx_omarchy-0.32.2.dev202609120301+69a01db3`, sha16
`f29b8dcb6bc0536e`, built on jwm1 from the pushed branch commit (farm4;
all mlx/ wheel members hash-verified against the installed venv).

## Restoration (Q4 short, fork, corrected instrument)

| wheel | prefill_s | tok/s | ratio vs native 294.1 |
|---|---|---|---|
| 63c9a8d8 (published baseline) | 0.089306 | 335.9 | 1.142 |
| ab08be8b (regressed tip) | 0.140323 | 213.8 | 0.727 |
| **69a01db3 (fix)** r1/r2/r3 | 0.089505 / 0.089884 / 0.090755 | 335.2 / 333.8 / 330.6 | **1.135-1.140** |

The fix wheel is back at the published baseline within session noise.
Stock driver recovers the same way: 135.4 regressed, 167.4-193.7 fixed
(published stock baseline 155-177 ms spans).

## Digest gates (fix wheel, 3 reps x both drivers, canonical manifest)

**36/36 HELD.** Q4 pins `7fd25a869ff21678` / `4cc08910089477fd` /
`7da83f06ec9f001d` and bf16 pins (fork `f26175202f3dabe9` /
`8690dc83246b39f8` / `ff502900d2a179a5`; stock `7fc0f968789b1882` /
`46108ad71157cb4d` / `ff502900d2a179a5`) reproduced exactly on every
measured leg of every rep. No digest moved.

## Battery

- `omarchy_copy_offset_tests`: **26/26 passed, exit=0** — the
  "back-to-back scalar fills stay ordered in one submission" case passes
  as written: ordering intact, single submission restored.
- `omarchy_primitive_tests` affine case: exit=1 — UNCHANGED, as expected:
  pre-existing known defect (see README item 2), not touched by this fix.

## Corrected cells context

The sibling's corrected 12-cell table (composed-main-qual
corrections/README.md) measured the regressed tree honestly: Q4 short
224.0 tok/s = 0.762 of native. With this fix the leg returns to
330-335 tok/s = 1.13-1.14 of native, so the README's published 1.13
fraction is reproducible on a fixed tree and the regression is the tree,
not the table.
