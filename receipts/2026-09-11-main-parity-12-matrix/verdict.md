# Closing canonical 12-matrix parity verdict on current main, both drivers

Date 2026-09-11. Source: `main` commit `711327ce` (harness commit recorded
per leg: `711327ce`; `origin/main` on GitHub resolved to exactly
`711327ceff1b5fc54db98626fbae5d44b216fd96` before checkout). Wheel built
for this receipt:
`mlx_omarchy-0.32.2.dev202609111202+711327ce-cp314-cp314-linux_aarch64.whl`,
sha256 `95748504822f0b75044280198954c85b708e1be78beb2e8bc68fe1995d1a868a`,
size 7573167 bytes, version stamp `711327ce` matches main. Upstream MLX
pin `mlx.lock`: 0.32.2, commit
`1f8e74e3f12f31365464a6867c6579f0e9b29d85`, archive sha256
`cb988a5bdc38c798918d042b9b1c6edda3ccc5f23a2155138d3aa5c1b2acc301`
(identical pin to the 2026-09-10 matrix).

## Verdict

**PASS.** All twelve legs (Q4 and BF16 at short, 262-token long, and
1053-token 1K context; prefill and pinned decode wall-anchored timed;
generated-ids digest checked) reproduce their current per-driver pins on
both the cooperative-matrix Honeykrisp fork driver and stock Mesa 26.1.7,
across 12 repetitions per driver after a discarded warmup matrix per
driver — 144/144 legs measured, every digest equal to its current pin,
every prompt and generated token count exact. The three native-parity
legs (Q4 short, Q4 1K context, BF16 1K context) still equal the native
macOS digests on **both** drivers (`docs/parity-id-policy.md` rule 2
holds at full strength); stock BF16 short now also equals the native
macOS digest.

## Protocol

`receipts/2026-09-10-prefill-close` canonical_parity_protocol, applied to
the fork/stock pairing, same as the 2026-09-10 matrix: one full
bench_matrix matrix per driver run and discarded as warmup, then 12
measured repetitions per driver, fresh bench_matrix process each,
fork/stock alternating; AC power (`macsmc-ac: online` on every run);
runner contention gate clean on all 26 runs (279–299 processes scanned,
zero model-serving matches); greedy generation (temp 0, seed 0) through
`scripts/bench_decode.py` with 4 warmup tokens per leg, pinned decode
lengths (32/128/32), EOS suppressed, prefill timed separately.
`MLX_DISABLE_COMPILE=1` (manifest). Engine:
`scripts/bench_matrix.py --mode run`; per-leg agreement checks (requested
tokens, decode span n-1, digest n, provenance line) enforced by the
runner and re-asserted by `summarize_closing.py`. One top-level
`flock /tmp/m1-gpu.lock` capped at two hours covered the session
(first warmup 12:06:23 UTC, last rep started 12:20:41 UTC — 15m8s used);
timings are wall-anchored only, device timestamps unused (this driver's
device clocks undercount ~2.07x). Driver package re-verified before the
run: `mesa-honeykrisp-omarchy 26.3.0.devel.hk6f6afc8-1`.

## Digests (identical across all 12 reps per leg; every digest = current pin)

| leg (prompt/gen tokens) | fork digest | stock digest | native |
|---|---|---|---|
| Q4 short (30/32) | `7fd25a869ff21678` | `7fd25a869ff21678` | same (native) |
| Q4 long (262/128) | `4cc08910089477fd` | `4cc08910089477fd` | `254d73fd93164b98` |
| Q4 1K ctx (1053/32) | `7da83f06ec9f001d` | `7da83f06ec9f001d` | same (native) |
| BF16 short (30/32) | `f26175202f3dabe9` | `7fc0f968789b1882` | `7fc0f968789b1882` |
| BF16 long (262/128) | `8690dc83246b39f8` | `46108ad71157cb4d` | `407b7624ed1b3b29` |
| BF16 1K ctx (1053/32) | `ff502900d2a179a5` | `ff502900d2a179a5` | same (native) |

Seven of twelve driver-legs now sit on the native macOS digest. The
cross-driver BF16 divergence moved with main's re-pinned digests: BF16
long remains split (fork `8690dc83246b39f8`, stock `46108ad71157cb4d`,
both new values) and BF16 short split for the first time (fork keeps
`f26175202f3dabe9`, stock moved to the native `7fc0f968789b1882`).

## Performance (medians, min–max over 12 reps; decode tok/s, prefill tok/s)

| leg | fork decode | fork prefill | stock decode | stock prefill |
|---|---|---|---|---|
| Q4 short | 111.37 (105.89–112.25) | 333.3 (333.3–333.3) | 101.75 (100.91–102.97) | 169.5 (169.5–193.5) |
| Q4 long | 108.15 (107.38–108.39) | 966.8 (922.5–974.0) | 81.20 (80.84–81.37) | 309.5 (308.6–310.8) |
| Q4 1K ctx | 96.37 (95.48–97.05) | 1114.9 (1111.9–1120.2) | 52.80 (49.46–53.21) | 362.3 (359.6–363.5) |
| BF16 short | 31.86 (31.80–32.02) | 131.6 (131.0–132.7) | 32.17 (31.70–32.27) | 133.3 (131.6–137.0) |
| BF16 long | 28.82 (28.71–28.91) | 446.7 (444.1–450.2) | 29.12 (29.05–29.23) | 229.0 (228.2–229.8) |
| BF16 1K ctx | 23.11 (22.26–23.19) | 456.3 (448.7–457.4) | 22.97 (21.99–23.24) | 224.4 (224.0–224.7) |

Paired medians: fork Q4 prefill 2.1–3.1x stock (cooperative-matrix
prefill kernels); fork Q4 decode +9.5/+33.2/+82.5% (short/long/1K);
BF16 decode parity within ±1% (the two drivers share the BF16 decode
path); BF16 prefill fork leads stock by 95–103% on long/1K, ±1% on short.

## Fraction of the committed base-M1 native macOS baseline

Baseline: `receipts/2026-09-10-native-macos-metal-baseline/
committed-base-m1/native-baseline-baseM1.json` (Apple M1, 8 GPU cores,
macOS 14.8.9, native Metal MLX 0.32.2). Fractions are medians/current
baseline; 1.00 = at native.

| leg | fork decode | fork prefill | stock decode | stock prefill |
|---|---|---|---|---|
| Q4 short | 0.740 | **1.133** | 0.676 | 0.576 |
| Q4 long | 0.737 | 0.797 | 0.553 | 0.255 |
| Q4 1K ctx | 0.686 | 0.606 | 0.376 | 0.197 |
| BF16 short | 0.565 | 0.566 | 0.570 | 0.573 |
| BF16 long | 0.517 | 0.443 | 0.523 | 0.227 |
| BF16 1K ctx | 0.424 | 0.276 | 0.421 | 0.136 |

**At or above native:** exactly one cell — fork Q4-short prefill at
1.133x (13.3% above the 294.1 tok/s native baseline). Everything else is
below native. Largest deficits on the fork driver: Q4 1K decode −31.4%,
BF16 1K decode −57.6%, BF16 1K prefill −72.4%; on stock the deficits run
to Q4 1K prefill −80.3% and BF16 1K prefill −86.4%.

## Delta versus the previous canonical matrix (b6d662a8, 2026-09-10)

| leg | fork Δdecode | fork Δprefill | stock Δdecode | stock Δprefill |
|---|---|---|---|---|
| Q4 short | −0.7% | +0.6% | −0.8% | −0.8% |
| Q4 long | −0.7% | −0.4% | −0.8% | −0.2% |
| Q4 1K ctx | +0.3% | +0.2% | −0.2% | +0.2% |
| BF16 short | +172.3% | +53.1% | +174.1% | +59.1% |
| BF16 long | +156.1% | +40.6% | +157.9% | +8.7% |
| BF16 1K ctx | +130.4% | +25.1% | +126.9% | +2.0% |

Q4 is unmoved (all cells within ±1%, i.e. session noise). BF16 decode
roughly doubled to 2.3–2.7x the previous medians and BF16 prefill gained
2–59% — the measured effect of the commits main gained since b6d662a8
(producer-direct KV cache writes, allocation-independent quantized-matmul
route, native-order dense BF16 decode GEMV, BF16 prefill
cooperative-matrix restructure). The fork Q4 prefill bimodality recorded
on 2026-09-10 did not appear this session: all 12 fork reps sat in the
fast mode (long-leg minimum 922.5 tok/s versus 761.6 last session);
digests identical in every rep, as before.

## Provenance

- Every measured leg printed `verified=match harness=711327ce` for
  `mlx-omarchy 0.32.2.dev202609111202+711327ce`
  (`libmlx.so=sha256:a6df71a07918be44`,
  `core.cpython-314-aarch64-linux-gnu.so=sha256:4ad850da16c300ea`) — the
  loaded binary matches the installed wheel's RECORD
  (`scripts/mlx_provenance.py` gate). The core binding hash is unchanged
  from the 09-10 wheel; `libmlx.so` moved, consistent with main's
  GPU-side commits landing in the native library.
- Models pinned and cache-verified: Qwen2.5-0.5B-Instruct-4bit
  `a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3`; Qwen2.5-0.5B-Instruct-bf16
  `56d07e766edd7159fbe12ed12d9cf114bf38bf1e`. Optional 7B/14B manifest
  entries skipped offline (not part of the 12 legs).
- Drivers (`drivers.json`, re-captured 2026-09-11): fork = installed
  Honeykrisp Mesa 26.3.0-devel (`git-6f6afc8968`), api 1.4.359,
  `VK_KHR_cooperative_matrix` advertised (vulkaninfo count 2); stock =
  Mesa 26.1.7 (`DRIVER_ID_MESA_HONEYKRISP`), api 1.4.354, no cooperative
  matrix (count 0), private ICD
  `/home/joshuawarren/stock-mesa/stock-icd.json`. Both on Apple M1
  (G13G B1), kernel 7.1.6-1-1-ARCH, Omarchy, Python 3.14.7.
- Exact commands: `run_matrix_canonical.sh` (this directory); build
  command and full log in `build-wheel.log`; machine-readable verdict
  with all 144 per-rep rows in `verdict.json`; summary table in
  `summary-canonical.txt`.

## Files

- `canonical/` — 26 run JSONs + per-run logs: 12 measured fork reps,
  12 measured stock reps, and the 2 discarded warmup matrices
  (`fork-warmup`, `stock-warmup`).
- `verdict.json`, `summary-canonical.txt`, `drivers.json`,
  `build-wheel.log`, `run_matrix_canonical.sh`, `summarize_closing.py`.
