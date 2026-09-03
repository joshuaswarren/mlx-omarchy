# 2026-09-03 — decode A/B across three wheels, affinity sweep, deep-collector verification (jwm1, 8 cores)

Exclusive-session follow-up to `receipts/2026-09-03-eight-core-remeasure.md`.
Machine: jwm1-linux, M1 T8103, kernel 7.1.6-1-1-ARCH, all 8 cores online
(`nproc` = 8), 1-minute load average between 0.0 and 0.9 across the session
(the load was this session's own single benchmark process). Every timed run
used the standing protocol: `MLX_DISABLE_COMPILE=1`,
`--max-tokens 32 --temp 0 --seed 0`, pinned model snapshots
(`a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3` 4-bit,
`56d07e766edd7159fbe12ed12d9cf114bf38bf1e` bf16), same prompts, chat
template default, 5 runs per leg, median reported. Raw logs:
`jwm1:~/benchq/logs/*.log`.

## Wheel inventory and sha256

| name | path | bytes | sha256 |
|---|---|---|---|
| `dev20260903` (local 06:06 rebuild; the wheel behind the README 8-core column) | `~/src/mlx-omarchy/dist/mlx_omarchy-0.32.2.dev20260903-cp314-cp314-linux_aarch64.whl` | 2872489 | `6e54ab2b8e808b78d052eae0d5dc76e4f4111f2afac3e572e9eec243d8c5f42c` |
| `dev202609030928` (ceae628-era build, from `mlx-omarchy-decomp` at HEAD `ceae628`) | `~/src/mlx-omarchy-decomp/dist/mlx_omarchy-0.32.2.dev202609030928-cp314-cp314-linux_aarch64.whl` | 6796874 | `63eece27f9901a73e7acfc62cde7e8feeead3f87b9d06740cc00322fbd6fcd9d` |
| v0.3.1 GitHub asset `dev20260903` | release download | 6789056 | `60d274fbc2c51d61f12b056fa7930b78f6badc551d8f16dbcf9fe925a24b0afd` |
| v0.3.2 GitHub asset `dev202609030512` (the published Latest) | release download | 6789976 | `61424114d983c8f3c02b6ae83125be9fe709a13a2fcb8b27ff0b0d02fb64e370` |

Verification: the v0.3.1 asset hash matches the receipt line in
`receipts/2026-09-02-release-v0.3.1.md` byte for byte. No build receipt line
exists on the box for either of the other two local wheels; their provenance
is the build directories themselves (`mlx-omarchy-decomp` HEAD is
`ceae628`; the `dist/` files carry the hashes above from today).

Byte-identity finding: the installed `mlx/` trees of the local `dev20260903`
wheel and the ceae628-era `dev202609030928` wheel are **identical in all 332
non-pyc files** (same `lib/libmlx.so`, same `core*.so`; only `__pycache__`
mtime artifacts differ; the wheel files differ in packaging size, 2.87 MB vs
6.80 MB, not in payload). The wheel the 8-core README column measured IS a
ceae628 build.

## Verdict 1: decode regression (ceae628-era vs released `dev20260903`)

**No.** A third leg was added because the two assigned wheels turned out to
be byte-identical; the published v0.3.2 asset is the only wheel with
different code. 8 cores, unpinned, medians:

| leg | bf16 prefill | bf16 decode | 4-bit prefill | 4-bit decode |
|---|---|---|---|---|
| `dev20260903` local (README wheel) | 17.420 | 2.002 | 18.666 | 3.847 |
| `dev202609030928` ceae628-era | 16.838 | 2.072 | 18.319 | 3.864 |
| v0.3.2 asset `dev202609030512` | 15.461 | 1.837 | 17.498 | 3.529 |

- `dev20260903` vs ceae628-era decode: bf16 **-3.4%**, 4-bit **-0.4%** —
  noise, as the byte identity predicts. There is no decode regression
  between the ceae628-era wheel and the released-as-used `dev20260903`
  wheel.
- The recorded 1-core decode figures (3.56 / 6.46) are therefore not
  reproducible from any surviving ceae628 artifact: today's rebuild of
  `ceae628` matches the 8-core re-measure (2.0 / 3.9), not the old numbers.
  Whatever produced the recorded rows was a build (flags, toolchain, or
  driver state) that no longer exists. The 40% "decode regression" is
  retired as a wheel-vs-wheel claim.
- The **published v0.3.2 asset is the real mover**: decode bf16
  **-11.3%** and 4-bit **-8.7%** against the ceae628-era binary; prefill
  bf16 -8.2%, 4-bit -4.5%. Anyone reproducing the README table from the
  GitHub release gets 5-11% less than the table says, on both rows.

## Verdict 2: affinity sweep (released-as-used `dev20260903` wheel)

Prefill medians, per-run lists in `~/benchq/logs/S0606-*.log`:

| affinity | bf16 prefill | vs 8-core | 4-bit prefill | vs 8-core |
|---|---|---|---|---|
| unpinned (8 cores) | 17.420 | — | 18.666 | — |
| `taskset -c 0` (1 E-core) | **21.835** | **+25.3%** | 18.207 | **-2.5%** |
| `taskset -c 0,1` (2 E-cores) | 17.617 | +1.1% | 19.004 | +1.8% |
| `taskset -c 0-3` (E-cluster) | 17.487 | +0.4% | 18.768 | +0.5% |
| `taskset -c 4-7` (P-cluster) | 18.310 | +5.1% | **19.708** | +5.6% |
| `taskset -c 4` (1 P-core) | 18.233 | +4.7% | 19.083 | +2.2% |

Decode observed at the same affinities (same runs): bf16 2.00-2.27,
4-bit 3.78-4.19 — flat within noise on every affinity except a mild
P-cluster gain (+8-13%), consistent with decode not tracking core count.

- The single-core prefill advantage is **bf16-only** (+25.3%, reproducing
  the earlier 22.162 data point); 4-bit prefill does **not** improve pinned
  (-2.5% on one core; best is the P-cluster at +5.6%).
- The discriminator point `taskset -c 0,1` collapses the win: +25.3% on one
  core becomes +1.1% with one extra E-core. The cost is
  per-additional-busy-core, not E-vs-P cluster type. Source facts from the
  parent (one CompletionDispatcher thread that polls then cv-sleeps 100 ms;
  no host thread pool) mean this is cross-core handoff/latency exposure on
  the bf16 host path, not a growing busy-spin.
- Consequence: a runtime-wide pin-to-one-core default is not justified (it
  costs 4-bit prefill), and a library-side affinity knob cannot reproduce
  the bf16 win because it needs every host thread, including Mesa's, on one
  core. The honest lever is a **documented run configuration**: for bf16
  host-bound prefill measurement (or single-model bf16 serving), run under
  `taskset -c 0` and recover ~25%. Decode is affinity-insensitive; pick
  affinity for prefill, not decode.

## MLX_MAX_OPS_PER_BUFFER and thread-count knobs

Not measured: the knob does not exist in the installed package. `strings`
on the installed `core.cpython-314-aarch64-linux-gnu.so` finds
`MLX_MAX_OPS_PER_BUFFER` 0 times and `OMP_NUM_THREADS` 0 times, against
`MLX_DISABLE_COMPILE` 2 times as a positive control. Upstream source
(pinned 1f8e74e3, archive sha256 matches `mlx.lock`) consumes
`env::max_ops_per_buffer` only in `mlx/backend/metal/device.cpp`; the
omarchy Vulkan backend never reads it. Setting it on this stack is a no-op,
so the best/worst-affinity knob legs were voided by evidence rather than
run.

## Deep collector, end to end with a real wheel

- Clean clone of origin/main on jwm1:
  `git clone --depth 1 https://github.com/joshuaswarren/mlx-omarchy.git`
  at `7c3c98eb28690a3a9724fa58058932ffe3916ba8`, 0 dirty files.
- Ran `scripts/collect_deep.py` under the venv holding the released
  `dev20260903` wheel. Sections confirmed available: `correctness`
  (6/6 probe ops pass) and `benchmark` (matmul sweep, 0.040/0.161/0.191
  TFLOP/s at n=256/512/1024). `profile` correctly recorded unavailable
  (no diagnostics build).
- Submission from that clone **failed**: endpoint answered
  `HTTP 422 schema_invalid`, rejecting the payload fields `cpu_present`,
  `hotplug_control`, `ane_dt_node`, `ane_dt_compatible`, `boot_chain`,
  `cmdline`, `core_shortfall` (introduced in `c03a6c0`). Origin/main's own
  `services/community-data/schema/payload-v1.schema.json` allows them; the
  **deployed worker predates `c03a6c0`** and needs a redeploy. Parent owns
  deploys; reported.
- To prove the remaining path, the same clean clone was checked out at
  `115ea2c582ca51d66252bab136b4e8805935640a` (the commit before the field
  additions) and the run repeated against the same venv:
  archive `deep-2026-09-03.tar.gz`, 5721 bytes,
  sha256 `8c083d2dc77af5b21e89b1764172dfe219cd3162d03a6977d52856e27044e7cc`,
  public receipt:
  **https://mlx-omarchy-community-data.joshua-s-warren.workers.dev/v1/results/8c083d2dc77af5b21e89b1764172dfe219cd3162d03a6977d52856e27044e7cc**
  (deduplicated=false, verified serving, `source_commit` recorded,
  `repo_dirty=false`, `cpu_online=8`).

## Exact commands

Venvs (`python3 -m venv`, then wheel, then `pip install --no-deps
mlx-lm==0.31.3`, then the support freeze of `venv-coreaudit` minus the mlx
lines): `~/venv-bqm1-0606`, `~/venv-bqm1-0928`, `~/venv-bqm1-0512`.

```
MLX_DISABLE_COMPILE=1 taskset -c <SPEC> <venv>/bin/python -m mlx_lm generate \
  --model <snapshot-dir> --prompt "$(cat prompt-file)" \
  --max-tokens 32 --temp 0 --seed 0        # SPEC: -, 0, 0,1, 0-3, 4-7, 4
```

`Prompt:` and `Generation:` lines parsed per run; medians above. Harness:
`jwm1:~/benchq/{run_leg.sh,parse_legs.sh,parse_legs.py,decode_legs.sh,sweep_legs.sh}`.

## Not measured here

- `MLX_MAX_OPS_PER_BUFFER` best/worst legs — knob provably absent from the
  installed package (evidence above).
- The diagnostics-wheel event stream for the worst case — deferred in favor
  of the dispatcher-poll-interval decode A/B that took priority mid-session;
  the affinity table above is the input that decision needed.
- The recorded 1-core ceae628 decode rows (3.56 / 6.46) remain unreproduced
  by any surviving build; they stay in the README with their condition
  named, and the build that produced them is no longer recoverable from
  surviving artifacts.
