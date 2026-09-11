# 2026-09-11 — M1 Max native macOS MLX baseline, second gated session
# (digest oracle + performance denominator confirmation before the Asahi wipe)

Date 2026-09-11 (UTC). Host 16m1mbp (Apple M1 Max, applegpu_g13s, 32-core
GPU, macOS 26.6.2 / 25G83). The machine is about to have Omarchy/Asahi
installed, after which no native macOS MLX number can ever again be
produced on this die. This receipt banks a second independent measurement
session so the same-die denominator and the digest oracle survive with
cross-session evidence, not a single day's word.

## The delta this receipt fills (vs the 2026-09-10 capture)

`receipts/2026-09-10-native-macos-metal-baseline/native-baseline-16m1mbp/`
(att7) already committed, on this same die:

- all six canonical legs, 12 reps each, 10-11 clean per leg under the
  contention gate, with a discarded warmup matrix;
- per-rep digests, all equal to the reference native digests;
- decode/prefill medians with spread;
- the base-M1 same-chip denominator (committed ea09eefa) and the
  cross-chip gap table.

What that capture left missing, and what this receipt adds:

1. **Cross-session reproducibility of the digest oracle.** All 2026-09-10
   evidence was one session on one day. An oracle that is not shown to
   reproduce across sessions is a one-day claim. This receipt is a full
   second gated session ~24h later: every leg reproduces its digest
   exactly, and decode medians drift at most +1.60% (see
   `reproducibility-2026-09-11.json`).
2. **Denominator stability across days.** The Linux M1 Max side (after
   the install) will report native fractions against these medians; two
   independent sessions agreeing within ~1.6% is what makes the
   denominator defensible rather than anecdotal.
3. **A documented protocol variant for a resident-state change** (below):
   on 2026-09-10 `ollama ps` was empty at gate time; on 2026-09-11 a
   resident embed-class model was loaded and resident services could not
   be touched in the capture window.
4. **An explicit missing list** (bottom of this file), including the fact
   that the harness records neither TTFT nor peak memory, and that the
   M1 Max row of `docs/chip-capability-axes.json` is a post-install
   Linux-driver measurement that this native capture can only feed, not
   replace.

Not re-measured (already committed, unchanged): the base-M1 native
denominator, the Linux canonical verdict, the M1 Ultra rows, the gap
table arithmetic.

## Protocol

Identical to the committed matrix (`receipts/native-baseline-2026-09-06`
protocol, engine = `scripts/bench_decode.py` verbatim at harness commit
`b6d662a`), except where the 2026-09-11 variant note says otherwise:

- Six legs: Q4 short/long/long-context (30/32, 262/128, 1053/32) and
  BF16 short/262-token/1K-context (30/32, 262/128, 1053/32), prompts
  byte-identical to `scripts/bench_matrix.json` expansion
  (`harness/prompts.json`), per-leg `prompt_tokens` asserted
  (30/262/1053) or the run refuses.
- Models pinned: `mlx-community/Qwen2.5-0.5B-Instruct-4bit`
  `a5339a41...`, `mlx-community/Qwen2.5-0.5B-Instruct-bf16`
  `56d07e76...`; local snapshot dir must equal the pin or the run
  refuses.
- Greedy temp 0 seed 0, EOS suppressed, pinned generated counts 32/128/32,
  4 warmup tokens per leg, prefill timed separately, decode rate over the
  n-1 inter-token gaps, digest = sha256(exact generated ids)[:16],
  `MLX_DISABLE_COMPILE=1`, fresh process per leg-run.
- Software: upstream PyPI `mlx==0.32.2`, `mlx-lm==0.31.3` (same venv as
  the 2026-09-10 session; versions recorded in the final JSON).
- One full 6-leg warmup matrix discarded, then 12 measured repetitions.
- Gates: AC power; standalone-model-server persistence scan; no loaded
  generation models; per leg-run CPU watch; >= 3 clean reps per leg.
- Engine provenance: `harness/bench_decode.py` and `harness/prompts.json`
  are byte-identical (md5-verified at deploy time against the committed
  2026-09-10 copies, which are themselves verbatim `b6d662a`).

### 2026-09-11 protocol variant — multi-pid watch (the only delta)

At gate time `ollama ps` showed one loaded embed-class model (a resident
embedding model, idle). Stopping or reconfiguring resident services was
not permitted in this capture window, so two refusals of the committed
gate would have fired on resident state rather than on measurement risk:

1. the "ollama holds loaded models" refusal, and
2. the standalone-server persistence scan, which matches the ollama
   daemon's own `llama-server` model-runner process.

The variant replaces exactly those two refusals with strictly wider
direct coverage, leaving every classifier, threshold and statistic
unchanged:

- WATCH_PIDS (5 pids): the resident router-inference leg (w1), the
  ollama daemon (w2), its loaded-model runner (w3), the resident
  document-service (w4), and `fileproviderd` (w5, which was bursting at
  100% CPU shortly before the run).
- Every leg-run samples cumulative CPU of all five pids across the
  window; `watch_pid_cpu_delta_s` (the classifier input, threshold 0.10s)
  is the max across pids; per-pid deltas are stored per row in
  `watch_all_cpu_delta_s`. The 70%-of-best-band decode guard is
  unchanged and still catches unwatched GPU contenders.
- A loaded ollama model is still refused unless it is embed-class and
  every ollama daemon/runner pid is in WATCH_PIDS. Non-embed loaded
  models still refuse outright, as does any genuinely standalone
  persistent server.
- A recorded 60s idle baseline over the watch set precedes measurement
  (`resident_idle_baseline_deltas_s` in the gate section).

ssh 16m1mbp
cd ~/src/mlx-bench-20260911
WATCH_PIDS=$(pgrep -x <resident-inference-service>),$(pgrep -x ollama),\
$(pgrep -f 'ollama/llama-server' | head -1),$(pgrep -f <resident-cpu-service> | head -1),\
$(pgrep -x fileproviderd) \
  ~/src/mlx-bench-20260901/venv/bin/python run_native_matrix.py 12 \
  native-baseline-16m1mbp-20260911
```

(w1 is the router leg, matching the 2026-09-10 `WATCH_PID` semantics.)

## Results (decode median tok/s / prefill median tok/s, clean-rep min-max;
12 reps, 11-12 clean per leg)

| leg (prompt/gen) | 2026-09-10 (att7) | 2026-09-11 (this receipt) | digest (both sessions) |
|---|---|---|---|
| Q4 short (30/32) | 286.96 / 1517.55 | 291.55 (283.40-299.43) / 1517.06 (1342.62-1789.38) | `7fd25a869ff21678` |
| Q4 long (262/128) | 296.25 / 5937.91 | 296.09 (294.43-300.31) / 5902.56 (5670.47-5999.59) | `254d73fd93164b98` |
| Q4 1K ctx (1053/32) | 283.79 / 8048.42 | 284.07 (283.27-286.02) / 8072.52 (7828.82-8177.60) | `7da83f06ec9f001d` |
| BF16 short (30/32) | 217.88 / 1192.74 | 218.47 (216.69-221.45) / 1266.54 (1167.77-1450.71) | `7fc0f968789b1882` |
| BF16 long (262/128) | 214.72 / 5281.12 | 214.23 (213.46-217.90) / 5297.83 (5186.18-5352.61) | `407b7624ed1b3b29` |
| BF16 1K ctx (1053/32) | 207.95 / 7751.35 | 207.87 (203.65-209.98) / 7750.83 (7513.46-7908.01) | `ff502900d2a179a5` |

## Digest oracle verdict

**All six legs are reproducible digest oracles on this die.** Evidence
(`reproducibility-2026-09-11.json`, generated by
`harness/cross_session_delta.py` over the committed per-rep rows of both
sessions):

- Within this session: one digest value per leg across all 12 reps
  (including the contended ones — contention changed speed, never ids).
- Across sessions: the 2026-09-11 digest set equals the 2026-09-10 set
  equals the reference native digests of the committed Linux canonical
  verdict, for all six legs.
- Decode median drift between sessions: -0.23% to +1.60%. Prefill median
  drift: -0.60% to +6.19% (bf16_short prefill is a 30-token prefill and
  was equally wide within the 2026-09-10 session: 1133-1444).

## Contention record (reported, not dropped)

- Gate idle baseline (60s, pre-run): the router leg (w1) burned 1.14s
  CPU during the baseline window (active traffic at gate time), then
  stayed quiet for the entire matrix: max w1 delta across all 72
  measured leg-runs = 0.02s.
- Rep 12: `fileproviderd` (w5, an unwatchable-by-name system sync daemon
  included in the watch precisely for this) burned 0.39-1.07s during
  four leg-run windows (q4_longctx, bf16_short, bf16_long,
  bf16_longctx). Those four reps are excluded from the medians by the
  unchanged classifier; q4_short and q4_long rep 12 were under threshold
  and stay included. Clean reps per leg: 12 (q4_short, q4_long), 11
  (other four). No leg fell below the >= 3 clean reps requirement; no
  median is contended.

## Same-die read-across (context, not a gate)

Against the committed base-M1 native denominator
(`receipts/native-baseline-2026-09-06`, 5 reps, same protocol, same
digests): M1 Max decode medians are 1.94x/2.02x/2.02x (Q4
short/long/1K-ctx) and 3.87x/3.85x/3.81x (BF16) the base-M1 medians.
Direction and magnitude match the 2026-09-10 read-across: Q4 decode at
0.5B is kernel-overhead-bound and nearly saturates at M1 Max, while BF16
keeps scaling with memory bandwidth. Per-chip numbers, unnormalized;
these ratios are cross-chip context for the future Linux M1 Max rows and
never a parity divisor (the same-chip M1 Max comparison will be native
vs Linux on THIS die after the install).

## What is still missing (honest list)

- **TTFT and peak memory per leg**: not captured — `bench_decode.py`
  records decode_tps, prefill, prompt_tokens, digest, ids, device, and
  nothing else; no new instrumentation was added for this receipt, per
  the capture instruction. The 2026-09-10 capture lacks them for the
  same reason, so the committed protocol has no TTFT/peak-mem columns
  on any host.
- **M1 Max row in `docs/chip-capability-axes.json`**: a Linux
  Honeykrisp-driver measurement (capability report, arithmetic probes,
  suites, per-driver-build digests). Impossible before the Asahi
  install; this receipt is the native denominator that row's benchmark
  legs will be measured against.
- **Native M2 Max matrix**: still missing entirely (2026-09-10 attempts
  were contended by router traffic; that host is out of scope here).
- **This session's residuals**: none withheld — raw per-rep rows, watch
  deltas, warmup discard and gate state are all committed under
  `native-baseline-16m1mbp-20260911/`.

## Files

- `native-baseline-16m1mbp-20260911/` — raw tree: discarded warmup
  matrix, per-rep per-leg `.log` + `.json` (with per-pid watch deltas),
  final `native-baseline-16M1MBP.json`, and
  `reproducibility-2026-09-11.json`.
- `harness/` — byte-identical `bench_decode.py`, `prompts.json`,
  `mlx_provenance.py`, `resummarize.py`; the variant
  `run_native_matrix.py` (multi-pid watch); `cross_session_delta.py`
  (reproducibility evidence generator, pure post-processing).

## Redaction

Receipts in this repository carry chip, OS/build and host alias only.
Redacted before commit, by class: battery hardware id in the pmset gate
output (`id=<redacted>`, matching the 2026-09-10 pattern). Never
recorded at all: process command lines and service names (the watch set
is stored as opaque labels w1-w5 plus a count; the runner's JSON stores
no argv), addresses, ports, serials. The only stderr content in any
captured leg log is an mlx deprecation notice.
