# Canonical 12-matrix parity verdict on current main, both drivers

Date 2026-09-10. Source: `main` commit `b6d662a8` (harness commit recorded
per leg: `b6d662a`). Wheel built for this receipt:
`mlx_omarchy-0.32.2.dev202609101916+b6d662a-cp314-cp314-linux_aarch64.whl`,
sha256 `98821134f4bcf306ab1d64d6885ddf3d3e8e932311283bbd11f97aecf6a8e439`,
version stamp `b6d662a` matches main. Upstream MLX pin `mlx.lock`:
0.32.2, commit `1f8e74e3f12f31365464a6867c6579f0e9b29d85`, archive sha256
`cb988a5bdc38c798918d042b9b1c6edda3ccc5f23a2155138d3aa5c1b2acc301`.

## Verdict

**PASS.** All twelve legs (Q4 and BF16 at short, 262-token long, and
1053-token 1K context; prefill and pinned decode timed; generated-ids
digest checked) reproduce their canonical per-driver values on both the
cooperative-matrix Honeykrisp fork driver and stock Mesa 26.1.7, across
12 repetitions per driver after a discarded warmup matrix — 144/144 legs
measured, every digest equal to canonical, every prompt and generated
token count exact. No leg moved versus v0.4.0/v0.4.1/v0.4.2. The three
native-parity legs (Q4 short, Q4 1K context, BF16 1K context) still equal
the native macOS digests (`docs/parity-id-policy.md` rule 2 holds).

## Protocol

`receipts/2026-09-10-prefill-close` canonical_parity_protocol, applied to
the fork/stock pairing: one full bench_matrix matrix per driver run and
discarded as warmup, then 12 measured repetitions per driver, fresh
bench_matrix process each, fork/stock alternating; AC power; runner
contention gate clean (297 processes scanned, zero matches) on every run;
greedy generation (temp 0, seed 0) through `scripts/bench_decode.py` with
4 warmup tokens per leg, pinned decode lengths (32/128/32), EOS
suppressed, prefill timed separately. `MLX_DISABLE_COMPILE=1` (manifest).
Engine: `scripts/bench_matrix.py --mode run`; per-leg agreement checks
(requested tokens, decode span n-1, digest n, provenance line) enforced
by the runner and re-asserted by `summarize_canonical.py`.

## Digests (identical across all 12 reps per leg)

| leg (prompt/gen tokens) | fork digest | stock digest | canonical source | native |
|---|---|---|---|---|
| Q4 short (30/32) | `7fd25a869ff21678` | `7fd25a869ff21678` | v0.4.0 table | same (native) |
| Q4 long (262/128) | `4cc08910089477fd` | `4cc08910089477fd` | v0.4.0 table | `254d73fd93164b98` |
| Q4 1K ctx (1053/32) | `7da83f06ec9f001d` | `7da83f06ec9f001d` | v0.4.0 table | same (native) |
| BF16 short (30/32) | `f26175202f3dabe9` | `f26175202f3dabe9` | v0.4.0 table | `7fc0f968789b1882` |
| BF16 long (262/128) | `ad964232ee67fecd` | `c5be9207833d2a26` | stock-prefill-digest verdict | `407b7624ed1b3b29` |
| BF16 1K ctx (1053/32) | `ff502900d2a179a5` | `ff502900d2a179a5` | v0.4.0 table | same (native) |

The known cross-driver BF16-long divergence persists exactly as recorded
at v0.4.1/v0.4.2 (fork `ad964232ee67fecd`, stock `c5be9207833d2a26`).
Both equal their canonical per-driver values; nothing moved on main.

## Performance (medians, min–max over 12 reps; decode tok/s, prefill tok/s)

| leg | fork decode | fork prefill | stock decode | stock prefill |
|---|---|---|---|---|
| Q4 short | 112.2 (96.5–112.9) | 331.5 (265.5–337.1) | 102.5 (92.8–104.3) | 170.9 (166.7–193.5) |
| Q4 long | 108.9 (62.6–109.3) | 970.4 (761.6–1035.6) | 81.9 (63.3–82.0) | 310.1 (246.7–331.6) |
| Q4 1K ctx | 96.1 (64.3–99.8) | 1112.5 (845.8–1181.8) | 52.9 (37.8–53.5) | 361.7 (256.7–364.7) |
| BF16 short | 11.70 (9.95–11.74) | 86.0 (76.1–97.1) | 11.73 (10.65–11.77) | 83.8 (79.6–94.0) |
| BF16 long | 11.25 (10.02–11.30) | 317.8 (300.1–332.9) | 11.29 (10.67–11.35) | 210.8 (178.7–210.9) |
| BF16 1K ctx | 10.03 (7.61–10.22) | 364.8 (250.5–365.4) | 10.12 (8.39–10.28) | 220.1 (166.1–220.5) |

Paired medians: fork Q4 prefill 2.1–3.1x stock (cooperative-matrix prefill
kernels); fork Q4 decode +9/+33/+82% (short/long/1K); BF16 decode parity
within ±1% (the two drivers share the BF16 decode kernels). Stock medians
sit within ~4% of the v0.4.2 public-wheel stock matrix, as expected for
the custom-kernel commits main gained since b3e977b.

Fork Q4 prefill remains bimodal
(`receipts/2026-09-10-prefill-close`): this single protocol session
captured both modes (rep08 at 761.6/845.8 tok/s long/1K versus the ~970/
~1112 medians; min columns above). Digests are identical in both modes —
the mode changes speed only, never arithmetic. A pre-protocol 3-rep
screening session (`screening-3rep/`) hit the fast mode 3/3 with the same
digests.

## Provenance

- Every measured leg printed `verified=match harness=b6d662a` for
  `mlx-omarchy 0.32.2.dev202609101916+b6d662a`
  (`libmlx.so=sha256:9bdfaca562639796`,
  `core.cpython-314-aarch64-linux-gnu.so=sha256:4ad850da16c300ea`) — the
  loaded binary matches the installed wheel's RECORD
  (`scripts/mlx_provenance.py` gate).
- Models pinned and cache-verified: Qwen2.5-0.5B-Instruct-4bit
  `a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3`; Qwen2.5-0.5B-Instruct-bf16
  `56d07e766edd7159fbe12ed12d9cf114bf38bf1e`. Optional 7B/14B manifest
  entries skipped offline (not part of the 12 legs).
- Drivers (`drivers.json`): fork = installed Honeykrisp Mesa 26.3.0-devel
  (`git-6f6afc8968`), api 1.4.359, `VK_KHR_cooperative_matrix` advertised;
  stock = Mesa 26.1.7 (`DRIVER_ID_MESA_HONEYKRISP`), api 1.4.354, no
  cooperative matrix, private ICD
  `/home/joshuawarren/stock-mesa/stock-icd.json`. Both on Apple M1
  (G13G B1), kernel 7.1.6-1-1-ARCH.
- Exact commands: `run_matrix_canonical.sh` (this directory); build
  command and full log in `build-wheel.log`; machine-readable verdict
  with all 144 per-rep rows in `verdict.json`; summary table in
  `summary-canonical.txt`.

## Files

- `canonical/` — 24 run JSONs + per-run logs for the 12 measured pairs and
  the 2 discarded warmup matrices, plus the driver script log.
- `screening-3rep/` — pre-protocol 3-pair session (36 legs, digests
  identical to canonical).
- `verdict.json`, `summary-canonical.txt`, `drivers.json`,
  `build-wheel.log`, `run_matrix_canonical.sh`, `summarize_canonical.py`.
