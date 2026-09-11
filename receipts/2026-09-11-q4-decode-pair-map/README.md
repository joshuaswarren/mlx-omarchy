# Q4 decode pair map at model level: faster at the pattern level, slower at the model level, and no closer to native — rejected for the pin and for landing

- schema: mlx-omarchy/q4-decode-pair-map/1 (2026-09-11)
- agent: Q4PairMapMeasure; branch wave/Q4DecodePairMap (off origin/main bb534004), receipt + env-gated measurement variant only. Never merged; never proposed for merge.
- host: jwm1-linux (<m1-host>), Apple M1 (G13G B1), linux-asahi 7.1.6.asahi1-1, installed fork driver mesa-honeykrisp-omarchy 26.3.0.devel.hk6f6afc8-1 verified in-window (`pacman -Q` in m1-logs/); stock arm = same wheel under VK_DRIVER_FILES=/home/joshuawarren/stock-mesa/stock-icd.json
- wheel: mlx_omarchy-0.32.2.dev202609111121+ef49fbc5-cp314-cp314-linux_aarch64.whl, sha256 9a0e0b15c58276dc4af080597e4385ff6cd25d54cf32167f53e388a9ebea75da
- windows: two ONE-flock windows on /tmp/m1-gpu.lock — accuracy + smoke 11:37:54–11:40:51Z, legs 11:40:51–11:47:04Z. Quiet gate loadavg 0.02–0.25 before windows; per-run loadavg_1m recorded on all 36 legs, all < 1.0.
- timing: wall-clock only (bench_decode); no device-timestamp number is used (known ~2.07x undercount on this driver).

## What was measured and why

receipts/2026-09-11-q4-memory-roof left exactly one winning access pattern
untaken: two consecutive packed words per lane per step (the native qmv_fast
map), +13% at the memory-pattern level (67–68 GB/s vs the production map's
60). It was not taken because it changes per-lane word order and therefore the
pinned accumulation order. This receipt measures the full model-level cost and
benefit so the pin decision rests on evidence instead of inference.

The variant ships as `-DQMM_VEC_Q4_PAIR=1` shader builds (single-weight,
QMM_VEC_MULTI fused, subgroup and tree reductions, f32/f16/bf16) selected only
by `MLX_OMARCHY_QMM_VEC_Q4_PAIR=1` at dispatch. Default binaries, dispatch,
and digests are untouched: every base arm below reproduces the six canonical
digests, and `bench_matrix` binary provenance verified the wheel on all 12
invocations. Engagement is proven by output change, not by assertion: the pair
map changes 2–4 of up to 4864 f16 outputs on the large projections (a
non-engaged variant would change none).

## 1. Decode performance (3 reps per cell, median tok/s)

| driver | leg (prompt/decode) | base | pair | pair/base | base ×native | pair ×native |
|---|---|---|---|---|---|---|
| fork | short 30/32 | 111.27 | 109.84 | **0.987** | 0.739 | 0.729 |
| fork | long 262/128 | 108.21 | 105.91 | **0.979** | 0.737 | 0.722 |
| fork | longctx 1053/32 | 96.52 | 95.14 | **0.986** | 0.688 | 0.678 |
| stock | short 30/32 | 102.97 | 100.25 | **0.974** | 0.684 | 0.666 |
| stock | long 262/128 | 81.30 | 80.38 | **0.989** | 0.554 | 0.548 |
| stock | longctx 1053/32 | 52.55 | 52.40 | **0.997** | 0.374 | 0.373 |

Native baseline = committed base-M1 macOS MLX 0.32.2 receipt
(native-baseline-2026-09-06): 150.57 / 146.77 / 140.38 tok/s. Per-rep values in
verdict.json `performance[].*_all_tok_s`. The +13% pattern-level win inverts
into a consistent 1.0–2.6% model-level loss on every driver × leg cell: in the
real model the q4 GEMV chain is latency/occupancy-bound (the roof receipt's own
conclusion), so trading single-word streams for wider per-lane streams buys
nothing and the extra register pressure taxes the rest.

## 2. Generated-id digests — the decisive datum

| driver | leg | base (canonical) | pair | pair equals |
|---|---|---|---|---|
| fork | short | 7fd25a869ff21678 | 7fd25a869ff21678 | **native and linux-base** (unchanged) |
| fork | long | 4cc08910089477fd | 55215e22d7f1b864 | **third value** |
| fork | longctx | 7da83f06ec9f001d | 31267e7ed4c6d0dc | **third value** |
| stock | short | 7fd25a869ff21678 | 7fd25a869ff21678 | **native and linux-base** (unchanged) |
| stock | long | 4cc08910089477fd | f873dc2bdd34c0fe | **third value** |
| stock | longctx | 7da83f06ec9f001d | 7da83f06ec9f001d | **native and linux-base** (unchanged) |

Native macOS values: short 7fd25a869ff21678, long 254d73fd93164b98, longctx
7da83f06ec9f001d. All 36 legs digest-stable across reps.

**The pin question does not invert.** The current kernel matches native on two
legs (short, longctx). The pair map matches native on at most the same two
(stock) and on fewer (fork: longctx moves away to 31267e7ed4c6d0dc), and it
moves the long leg — the one leg where Linux already diverges from native — to
a third value on both drivers, never to native's 254d73fd93164b98. The change
is slower AND not closer to native.

Interpretive note [INFERENCE, flagged as such]: fork longctx pair digest
31267e7ed4c6d0dc is byte-identical to the digest recorded in
docs/known-defects.md for the mesa-coopmat build run with AGX_SIMDMAT=1. A
64-bit digest collision across two setups is implausible; the natural
explanation is that both setups run the same qmv_fast-order arithmetic for
decode — and since that is not native macOS's digest either, it corroborates
that native runs plain qmv order at K=896 (K % 512 != 0), exactly what the
kernel comment says. Also notable: fork and stock disagree with each other
under the pair map on the long leg (55215e… vs f873dc…), exposing a
driver-level arithmetic difference between honeykrisp and stock Mesa that the
pinned order currently masks.

## 3. Accuracy against a float64 RNE reference (seven decode projections, seeded inputs, both maps)

Per shape: elements differing from round_f16(f64 reference), max abs error;
references are refA = x_f64 @ dequant(w)_f64 (model level) and refB = the
kernel's own group algebra in f64. Full table in verdict.json `accuracy`.

| shape (K×N) | base max abs err (refA/refB) | pair max abs err (refA/refB) | pair≠base outputs |
|---|---|---|---|
| q 896×896 | 7.32e-4 / 7.32e-4 | 7.32e-4 / 7.32e-4 | 0 / 896 |
| k 896×128 | 1.22e-3 / 1.22e-3 | 1.22e-3 / 1.22e-3 | 0 / 128 |
| v 896×128 | 7.32e-4 / 5.49e-4 | 7.32e-4 / 5.49e-4 | 0 / 128 |
| o 896×896 | 9.77e-4 / 9.77e-4 | 9.77e-4 / 9.77e-4 | 1 / 896 |
| gate 896×4864 | 1.08e-3 / 1.01e-3 | 1.08e-3 / 1.01e-3 | 4 / 4864 |
| up 896×4864 | 9.77e-4 / 9.77e-4 | 9.77e-4 / 9.77e-4 | 3 / 4864 |
| down 4864×896 | 1.95e-3 / 1.95e-3 | 1.95e-3 / 1.95e-3 | 2 / 896 |

Both maps sit at identical distance from the f64 reference on every shape
(same max abs error, same differing counts against refA); the reorder only
flips a handful of outputs at the last f16 rounding. No precision regression,
and the residual distance from pure f64 is the native-inherent half-precision
quad chain, not the map.

## 4. Prefill

Prefill seconds per leg (bench_decode times the first generated token with
prefill; that step is a decode dispatch, so a real pair-map effect would show
here): fork 1.000 / 1.026 / 1.000 and stock 0.996 / 1.000 pair-over-base
ratios on the other cells — unchanged within noise. The one outlier, stock
short at 1.127 (0.158 → 0.178 s median), is warmup-ambiguous: base rep 1
measured 0.177 s (pair level) before settling to 0.158, and the same-cell fork
ratio is 1.000. Recorded as measured; not evidence of a prefill effect.

## Decision

Rejected on both axes the assignment set: no performance gain at model level
(consistently 1–2.6% slower), and no digest improvement over the current
kernel (native matches: 2/3 stock, 1/3 fork, versus 2/3 today) — so the pin
stays with the current one-word-per-lane map and the change is not landed.
The env-gated variant and the six pair shaders stay on this branch only, as
measurement instrumentation; main and the integration worktree are untouched.
Nothing merged; no test added (nothing landed).

## Provenance

- branch wave/Q4DecodePairMap: c6a28841 (pair-map variant), 71ee79d4 (shader
  macro-body fix caught by the glslc gate), ef49fbc5 (shader header includes)
- legs/: manifest-q4.json, 12 run JSONs + logs (r1–r3 × fork/stock ×
  base/pair), summary.json (36 leg records with per-run loadavg)
- accuracy/: report.json + base/pair index.json; capture npz files stayed on
  jwm1 (~/src/mlx-pairmap-m1/receipts-workdata/accuracy/, ephemeral)
- m1-logs/: window3.log (accuracy + smoke window), window-legs2.log (legs
  window)
- harness: run_pair_legs.py, run_accuracy_arm.py, compare_accuracy.py,
  analyze_pair.py, window.sh, window_legs.sh (this directory)
