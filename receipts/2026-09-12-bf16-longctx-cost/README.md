# BF16 1024-context decode critical-path attribution

## Result

The paired eager BF16 run falsified the hypothesis that added host CPU work fully explains the 1024-context decode penalty. No source candidate was selected.

The uninstrumented median increased from 31.2338 ms/token for the short prompt to 40.0091 ms/token at 1024 context, an 8.7753 ms/token penalty. Process CPU increased by 4.1547 ms/token. Even if all added process CPU were exposed on the critical path, 4.6206 ms/token, or 52.65% of the median wall penalty, remains outside that explanation. Using the adverse observed extrema leaves a 3.4921 ms/token non-CPU remainder, or 41.07% of the conservative wall delta.

The instrumented wall delta was 8.7136 ms/token. Its -0.0618 ms/token difference from the uninstrumented delta is small relative to the attributed penalty.

## Identity and source reconciliation

- Harness source: `e4d079c883b8606dddef7720bd302c5756be0692`.
- Qualified wheel source: `6b1ac0296ba65a8e0075171ca9451e222ddff06b`.
- Wheel: `0.32.2.dev202609130004+6b1ac029`, SHA-256 `fb4b96b0e52de3a07d62fea6766baa24f409b0b6b32afef1fd5f365e246e1c50`.
- `source-reconciliation.txt` proves that all eight grouped-BF16 implementation and test paths are identical between those commits. The 95 intervening paths contain no non-ANE Omarchy runtime source change.
- Benchmark model: `mlx-community/Qwen2.5-0.5B-Instruct-bf16@56d07e766edd7159fbe12ed12d9cf114bf38bf1e`.
- Hardware: `jwm1-linux`, Apple M1 (G13G B1), Mesa Honeykrisp `26.3.0.devel.hk6f6afc8-1`, kernel `7.1.6-1-1-ARCH`.
- Execution model: `openai-codex/gpt-5.6-sol`. No model fallback was observed.

The active native reference is the five-repeat 2026-09-06 receipt from the same physical M1 and model snapshot. The 1024-context digest matches exactly, and current Linux reaches 24.9943 tokens/s versus native MLX at 54.55 tokens/s, or 45.8191% of native. The native short digest differs from the current short digest, so the receipt does not use the native short result or compute a cross-context native ratio.

## Controlled measurements

| Mode | Prompt | Wall ms/token, median | Observed range | Process CPU ms/token, median |
|---|---:|---:|---:|---:|
| Uninstrumented | short | 31.2338 | 0.2782 | n/a |
| Uninstrumented | 1024 context | 40.0091 | 0.1289 | n/a |
| CPU clock | short | 31.3718 | 0.0759 | 16.0546 |
| CPU clock | 1024 context | 40.0854 | 0.0541 | 20.2094 |

Each cell contains three counterbalanced repetitions. Every repetition generated 32 tokens, reported `verified=match`, loaded the pinned `+6b1ac029` wheel, and matched its prompt digest: short `f26175202f3dabe9`, 1024 context `ff502900d2a179a5`.

The window acquired `/tmp/m1-gpu.lock` at `2026-09-13T04:24:58Z` and released it at `2026-09-13T04:26:49Z`, before the fixed `2026-09-13T04:33:27Z` deadline. The quiet gate recorded three consecutive 0.00 one-minute loads. The maximum sampled one-minute load during the run was 0.35. The exit path recorded `CHILD_PIDS_CLEAR`, a successful device reopen, and `rc=0`.

## CPU profile

The decode-scoped `cpu-clock:u` profiles lost zero samples. Estimated long-context minus short-context sampled self-time was distributed across eager-fusion analysis (+0.6450 ms/token), Python runtime (+0.8212), heap and memory work (+0.1993), dispatch recording (+0.1797), and driver code (+0.1607). The remaining reported and unclassified symbols contributed +2.5766 ms/token. No narrow host function accounts for the wall penalty, and these self-time estimates are not added to GPU wait.

Raw decode-scoped stacks and flat reports are under `window-final/`. The larger raw `perf.data` files remain on the measurement host; `window-final/perf-data-manifest.txt` records their names, byte sizes, and SHA-256 digests.

## Contract and outcome

Eager execution used `MLX_DISABLE_COMPILE=1` and `HF_HUB_OFFLINE=1`. The compiled BF16 control exited with code 1 and the named `Compiled tape bfloat16 is refused` error. Output digests, source provenance, GPU device identity, and the refusal contract remained intact.

No speculative source change was made. The next discriminating measurement is the repository's existing compile-time `MLX_OMARCHY_GPU_PROFILING` harness with `MLX_OMARCHY_GPU_PROFILE`. It records per-dispatch GPU ticks, submit clocks, and completion waits. A paired short/1024 eager BF16 run can locate the residual by kernel group before source work is selected.

## Invalid attempts retained

- `attempt0-false-contender/`: the original broad `pgrep -f` guard matched a dormant Bash monitor and exited before measurement.
- `attempt1-transfer-name/`: staged filenames did not match checked paths and preflight exited before measurement.
- `attempt2-scanner-self-match/`: the first Python classifier interpreted an arbitrary MLX path as work and detected itself. The earlier false claim that its live local scan passed was corrected immediately; the observed scan exited 1 after also matching unrelated worker watchers.

None of these attempts contributes performance data. Each receipt records child cleanup and lock release.

## Reproduction

```bash
python3 receipts/2026-09-12-bf16-longctx-cost/analyze.py
```

This regenerates `verdict.json` from the retained raw logs, scoped profiles, native reference, and control measurements while checking the token, digest, provenance, deadline, refusal, and cleanup invariants.
