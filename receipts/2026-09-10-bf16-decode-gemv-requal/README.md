# Requalification: native-order dense BF16 decode GEMV against the parity contract

Date: 2026-09-10. Agent: Bf16NativeOrderRequalify.
Branch under test: `wave/Bf16DecodeGemvRebased` — 6f1612c2 ("perf: match native
BF16 decode GEMV order") cherry-picked onto origin/main 7aa8cf0f; harness
commits 4a856151..ed4d378c on the same branch. NOT landed; this receipt is the
decision evidence.

## Verdict

**NOT LANDABLE under `docs/parity-id-policy.md` — and the premise is measured
false: the kernel does not reproduce macOS Metal bits on real data.** It is
bit-identical to the current sequential kernel on every captured decode
projection, including all bias-cancellation elements, and both are
RNE(f64)-exact where Metal deviates. Its end-to-end digest changes on the
262-token leg are incidental f32-order tie flips in gate/down/lm_head that
land on forbidden third values (neither the Linux value nor native's), and it
splits the drivers on the short leg. The 2.3–2.7x decode speedup is real and
carries no measured f64-accuracy cost — but it buys zero parity gain, so the
digest churn is exactly what the contract forbids.

## 1. BF16 generated-id digests vs native macOS (3 reps per cell, identical)

Native targets: short `7fc0f968789b1882`, long `407b7624ed1b3b29`, 1K ctx
`ff502900d2a179a5`.

| leg | driver | base (b6d662a8 wheel) | candidate | verdict |
|---|---|---|---|---|
| BF16 short (30/32) | fork | `f26175202f3dabe9` | `f26175202f3dabe9` | unchanged, still divergent from native |
| BF16 short (30/32) | stock | `f26175202f3dabe9` | `7fc0f968789b1882` | moved to native — **fork/stock now disagree** |
| BF16 long (262/128) | fork | `ad964232ee67fecd` | `8690dc83246b39f8` | **FORBIDDEN** third value (same as the pre-rebase candidate) |
| BF16 long (262/128) | stock | `c5be9207833d2a26` | `46108ad71157cb4d` | **FORBIDDEN** fourth value, new |
| BF16 1K ctx (1053/32) | fork+stock | `ff502900d2a179a5` | `ff502900d2a179a5` | native held (policy rule 2) |

The 262-token leg fails policy rule 1 ("may move to native, not anything
else") on both drivers. The short leg creates a new cross-driver stream split
(fork-only arithmetic is a defect per the v0.4.1 rule; here the kernel is
selected on both, but the streams diverge anyway).

## 2. Q4 canonical digests untouched

All six unchanged at 3/3 reps on both drivers: short `7fd25a869ff21678`,
long `4cc08910089477fd`, 1K ctx `7da83f06ec9f001d` (fork and stock each).
Policy rule 2 holds.

## 3. Paired medians (3 reps/cell, quiet gated M1) vs committed base-M1 native (ea09eefa)

Decode tok/s (median, base→cand, speedup, candidate fraction of native
56.43/55.72/54.55):

| leg | fork | stock |
|---|---|---|
| BF16 short | 11.74→31.73 (2.70x) [0.566 nat] | 11.66→29.61 (2.54x) [0.525 nat] |
| BF16 long | 11.26→28.18 (2.50x) [0.513 nat] | 11.23→26.77 (2.38x) [0.473 nat] |
| BF16 1K ctx | 10.02→22.93 (2.29x) [0.420 nat] | 9.16→20.77 (2.27x) [0.380 nat] |

BF16 decode goes from ~0.18–0.21 of native to ~0.38–0.57 of native. The
original candidate's fork numbers reproduce (32.04/28.37/22.64 in its single
session vs 31.73/28.18/22.93 here).

Prefill tok/s (candidate leaves M>1 kernels untouched; prefill digests are
identical base-vs-candidate everywhere): fork 87.5→134.5 / 318.0→369.5 /
363.0→376.6; stock 93.8→132.7 / 210.3→219.6 / 191.9→200.0. The short-leg
prefill deltas are 0.2–0.35 s workloads and read as noise; long/ctx move ~+4–16% fork, ~+0–4% stock. Q4 medians sit inside the known bimodal bands
documented in `receipts/2026-09-10-main-parity-12-matrix` (stock Q4 decode
took the low mode this session; digests unaffected).

## 4. Numerical error vs RNE(f64) — the decisive measurement

Method: root-cause receipt conventions (`rne_bf16`, bit distance = |int
bits − int truth|). Inputs: the native-fixed capture (72 real tensors,
`capture.json` sha256 `2defcc6a…`) for decode/prefill q/k/v/o through the real
nn.Linear modules (bias included), plus patterned synthetic sets at gate/down/
lm_head shapes (real weights, no bias) and a deep-cancellation set. Four
cells: {base,candidate} x {fork,stock}; full table in `ulp/truth.json`.

**Real decode projections (896x896, 128x896, 896x896 o):**

| op | base vs RNE(f64) | cand vs RNE(f64) | cand vs base bits | native (Metal) vs RNE(f64) |
|---|---|---|---|---|
| decode q_proj | 0 mism / 0 ULP | 0 mism / 0 ULP | identical | 106 mism, max 20 |
| decode k_proj | 0 / 0 | 0 / 0 | identical | 13, max 7 |
| decode v_proj | 0 / 0 | 0 / 0 | identical | 46, max 14328 raw (v[50] → 0x0000) |
| decode o_proj | 0 / 0 | 0 / 0 | identical | 0 |

Bias-cancellation elements specifically: decode.v_proj |truth|<1e-3 (n=4, the
regime where Metal rounds to +0) — 0 mismatches on BOTH kernels;
decode.o_proj |truth|<1e-3 (n=148) — 0 on both. **The candidate is
bit-identical to the sequential kernel and both are RNE(f64)-exact exactly
where Metal is least exact.** The "native-order" kernel never reproduces
Metal's reduced-precision product rounding — that is what the original
receipt's synthetic "0 mismatches vs native" hid: on cancellation-free
synthetic data Metal is also exact, so agreement there proved nothing.

Prefill q/v: base == cand identically (31/1 and 4/2 — the recorded RNE tie
noise; unchanged from the root-cause receipt).

Synthetic large-N sets (f32 accumulation-order tie flips at N up to 151936):
base misrounds 23–45/151936 lm_head elements per trial (max 51 bits);
candidate 6–22 (max 8) — the new order is generally CLOSER to RNE(f64) than
the old vec kernel, never farther on these samples. gate/down diffs vs base:
1–2 elements, ≤2 bits. cancel896x128 (truth ~1e-6 of partial scale): 2/1 on
both kernels, bit-identical to each other. (cancel4864x896 was dropped by a
capture-script bug fixed after the window; the cancellation regime is covered
by the 896 set and the real v_proj/o_proj cuts.)

**Answer to the ticket question:** matching macOS bit-for-bit is not
delivered by this kernel and is not achievable with any f32-FMA GEMV —
Metal's deviation comes from reduced-precision product rounding
(`receipts/2026-09-10-bf16-rootcause`, confirmed here at kernel level). It
would require emulating bf16-product accumulation: a deliberate accuracy
regression vs RNE(f64) and non-portable across Metal versions. The candidate's
2.3–2.7x speedup comes at **no measured f64-accuracy cost** (neutral to
slightly better on the ops it newly covers), but it changes two digests to
forbidden values and splits the drivers, so the contract rejects it.

## 5. Suites

| suite | llvmpipe (x64 dev, ALLOW_NON_APPLE) | M1 (fork driver) |
|---|---|---|
| omarchy_matmul_family_tests | 19/19 cases, 20,785,711 assertions, PASS | 19/19 cases, 82,654,252 assertions, PASS |
| omarchy_runtime_tests | 41/41 cases, 22,691 assertions, PASS | 41/41 cases, 22,681 assertions, PASS |

## Provenance

- Branch `wave/Bf16DecodeGemvRebased`: kernel commit a70ed0dc (= 6f1612c2
  cherry-picked onto origin/main 7aa8cf0f; patch verified byte-identical),
  harness commits 4a856151, 06e131a4, 08ca48f, ed4d378c, plus receipt commit.
- Candidate wheel `mlx_omarchy-0.32.2.dev202609102224+e340ce3-cp314-cp314-
  linux_aarch64.whl`, sha256 `cdcc50ae527c4229315be86d85d98d550247bbf8956cf65166317606509c59be`,
  plain stamped release build (DEV_RELEASE=1, no diagnostics) to match the
  baseline wheel's build mode. Baseline wheel = committed canonical
  b6d662a8 wheel, sha256 `98821134f4bcf306ab1d64d6885ddf3d3e8e932311283bbd11f97aecf6a8e439`
  (verified before install).
- Host jwm1-linux (Apple M1 G13G B1, kernel 7.1.6-1-1-ARCH). Matrix window
  17:31:38–17:46:43 under `flock /tmp/m1-gpu.lock` (single top-level
  acquisition; one discarded warmup matrix; fork/stock/base/candidate
  interleaved; `bench_matrix --mode run` with pins and provenance gates —
  every measured leg `verified=match`, clean/AC gates green, 0 validation
  problems in `summary.json`).
- Drivers (`drivers-fresh.txt`): fork Honeykrisp api 1.4.359 driverVersion
  26.2.99 (installed package; base cells reproduce every canonical Linux
  digest, so the canonical per-driver values stand); stock Mesa 26.1.7
  Honeykrisp api 1.4.354 via `stock-icd.json`. The bf16_vec selection requires
  subgroupSize==32 — true on both M1 drivers, false on llvmpipe (which
  correctly keeps the sequential paths; the guard's fallback is the general
  tile kernel).
- Native capture `/tmp/bf16chain3-native-fixed/import-2/capture` (sha256
  `2defcc6ad88268f0…`) reused for real-input ULP measurement.

## Files

- `summary.json`, `summary-canonical.txt`, `matrix/r{1,2,3}-{fork,stock}-{base,cand}/` — digest+perf evidence
- `ulp/truth.json` — full per-set, per-cell ULP statistics incl. vs-native and cancellation cuts
- `m1-suites/`, `llvmpipe-*.log` — suite logs; `m1-logs/`, `window.log`, `drivers-fresh.txt`, `commit.txt`, `candidate-wheel.sha256` — provenance
- Harness: `m1_window.sh`, `m1_build.sh`, `m1_matrix.sh`, `m1_suites.sh`, `m1_ulp_capture.py`, `m1_ulp_truth.py`, `ulp_common.py`, `summarize_requal.py`
