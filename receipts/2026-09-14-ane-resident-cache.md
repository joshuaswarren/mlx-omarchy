# Ane resident worker, inline payloads

Date: 2026-09-14
Host: jwm1-linux (aarch64), boot `95872130-fd28-4d67-8247-03a0e8b61202` since 2026-09-13 09:43:25
Worktree: `~/.config/superpowers/worktrees/mlx-omarchy/AneResidentCache` at `a8ce9dd0` plus uncommitted overlay
Worker: `/var/tmp/AneResidentCache/mlx-omarchy-ane-worker-inline` sha256 `6dcf9a451e103039cc4dfbd5871cf322360b1e3fdbf458b2f17dd7260fc77cf6`
Model: xai-oauth/grok-4.6 (parent: OpenAI-named routes currently resolve to grok; do not abort)

Resume of AneResidentCache. File staging on a resident session cost 14-19 ms per submit on jwm1 and made residency slower than launch-per-submit. Payloads now travel inline on the worker stdin/stdout. Bounded-submit safety is unchanged: one private `--serve` child, per-submit deadline, failure ends the session, no retry.

This is 24 layers × 2 island submits = 48 worker jobs carrying 72 ANE programs (`ane_ops=72`).

## Verdict

Inline dropped the worker's own host staging. It did not drop the ~13 s encoder remainder.

That remainder is GPU work between ANE islands, not file I/O. Launch-per-submit wall minus summed ANE submit time is 12.6 s here (1302 `gpu_ops`, 6010 `vk_compute_dispatches`). File reads were 14-19 ms × 48 ≈ 0.7-0.9 s.

Resident + inline is faster than resident + files on the same boot, and still slower than launch-per-submit.

## Same-window measurement

One `flock -w 1800 /tmp/m1-gpu.lock` window, never stolen. Same inline-capable binary for both arms. Command: `/var/tmp/AneResidentCache/run_inline_encoder.sh` (14:07:03–14:07:36 -05:00). Workers before and after: 0. Timeouts: 0. CPU tensor events: 0.

| arm | wall_ms | ANE exec_ms | starts | submits | stage_ms sum | notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| launch (before) | 15739.6 | 3180.4 | 48 | 48 | n/a (files) | current ANE encoder path |
| resident + inline (after) | 16182.3 | 4447.0 | 1 | 48 | 220 (mean 4.6) | `--inline` / `--emit` |
| EncoderParityAne `out-ane` | 15961.5 | 3179.5 | 48 | 48 | files | earlier same-boot current encoder |

Same-boot file-staging resident, not re-run in this window (`out/resident-1`, 11:28): wall 20258.3 ms, exec 6987.9 ms, 1 start, 48 submits. Inline recovered 4.1 s of that vs files and still lost 0.44 s vs launch.

Resident open 13.2 ms, close 11.0 ms. Bundle loads 2, device program loads 3. Scratch after the resident arm: only `resident-worker.stderr`.

First jobs on the resident arm:

```
job status=0 bundle=island-attn-a-kt elapsed_ms=124 ... stage_ms=4 save_ms=10
job status=0 bundle=island-pv       elapsed_ms=45  ... stage_ms=4 save_ms=0
```

## Identity

Byte-identical to the current ANE encoder output `b3d60c81bcd9c63fcdcb5c5578d765cc221b0126af3482fb165d769a65c23e6e` (`/var/tmp/EncoderParityAne/out-ane/encoder_hidden.npy`).

| tensor | sha256 |
| --- | --- |
| launch `encoder_hidden.npy` | `b3d60c81bcd9c63fcdcb5c5578d765cc221b0126af3482fb165d769a65c23e6e` |
| resident `encoder_hidden.npy` | `b3d60c81bcd9c63fcdcb5c5578d765cc221b0126af3482fb165d769a65c23e6e` |
| encoder_mask (both; sum 375, exact vs golden) | `d8bf6ec9a4065ce6dff4d9252edc07e2910eeeac87f8fc3c05bd949e1e7861a7` |

Vs macOS golden (bounds still pass; Linux ANE is not bit-exact to Apple): max abs 0.147156, mean abs 0.004182, rel L2 0.024917, NaN 0, Inf 0.

## Host tests

No ANE, no GPU lock.

- `omarchy_ane_worker_tests` on jwm1: 12 cases, 125 assertions, SUCCESS
- `python3 -m unittest overlay.tests.omarchy.coreml.test_ane_resident -v` on omp-studio-local: 8 tests, OK (5.357 s)

## Source

Uncommitted on the worktree:

- `overlay/mlx/backend/omarchy/ane/worker.{h,cpp}` — resident session, inline frames over a socketpair, poll not sleep
- `overlay/tools/mlx-omarchy-ane-worker/main.cpp` — `--serve`, `--inline NAME=BYTES`, `--emit`
- `overlay/tools/coreml/ane_resident.py` — client sends payloads on stdin, not files
- `overlay/tests/omarchy/ane/test_worker.cpp` — one load, deadline, death, named failure
- `overlay/tests/omarchy/coreml/test_ane_resident.py` — payloads never touch the filesystem

Rebuild on jwm1: `ninja -C /var/tmp/AneWorkerValidation-c05ba1df/.work/mlx/build-ane-device mlx-omarchy-ane-worker omarchy_ane_worker_tests`

Raw reports: `receipts/2026-09-14-ane-resident-cache/out-inline/`
