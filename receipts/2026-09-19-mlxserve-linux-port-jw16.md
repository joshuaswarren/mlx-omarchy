# mlx-serve Linux port: clean-checkout verification + four-leg serve bench — 2026-09-19

Lane: ServeDocsMlxServe. Host: jw16mbp1-linux (Apple M1 Max, T6001, 10
cores / 62 GB). GPU work inside `flock /tmp/m1-gpu.lock` windows with
`llm-inference.service` stopped before and restarted + verified after
(200 + real completion, see post-state). Sibling agent
`MesaRegressionBisect` shared the same lock by the flock contract.

## What was verified

Branch `joshuaswarren/mlx-serve:linux-vulkan-port` builds and serves
from a **clean checkout** on jw16. This took three attempts, and the
two failures are real port findings, now fixed on the branch:

1. **`lib/opencode2-mlx-serve` submodule was not initialized.**
   `lib/opencode2_plugin.zig` `@embedFile`s files from it; a clean
   clone dies mid-compile with `error: failed opening
   "opencode2-mlx-serve/LICENSE": FileNotFound`. Fixed by a configure-
   time check in `verifyMlxStageLinux` (commit `1b12f6f`).
2. **ELF rpath was dead.** `addMlxLib` emits `@loader_path/...`
   rpaths — Mach-O syntax the ELF loader does not expand. The Linux
   binary linked, then failed at startup: `libmlxc.so: cannot open
   shared object file`. Everything before this commit only "served"
   with `LD_LIBRARY_PATH` papering over it. Fixed by mirroring the
   rpaths in `$ORIGIN` form for the Linux graph (same commit range);
   verified `readelf -d` shows both entries and the binary runs from
   any cwd.
3. Also committed: `src/model.zig` defaults the 1-D f16→bf16 narrowing
   off Darwin (`673e440`). The Omarchy backend capability-gates bf16
   ops and hard-throws "No GPU kernel exists for it" on a shape miss;
   Metal keeps the bf16 default, `MLX_SERVE_F16_NARROW_1D=1` forces
   the old behavior.

Clean-build receipt: fresh clone from the branch, submodules, MLX
staged from `mlx-omarchy` `prepare-mlx.sh` output (Vulkan backend,
`MLX_BUILD_OMARCHY=ON`), mlx-c `56b2d39` + two patches, jinja static
lib, `zig build --release=fast`. Log:
[`build-clean-checkout.log`](2026-09-19-mlxserve-linux-port-jw16/build-clean-checkout.log)
(`CLEAN_BUILD_OK` at the end). Verified binary:
`zig-out/bin/mlx-serve` sha256-16 `331318ebb60c9166`, `--version`:

```
mlx-serve 26.9.5-dev
mlx 0.32.3
mlx-c 56b2d39fc831f2c0eb5bb94d82ef7191f7b31fa6
nax off (requires M5-class GPU)
ggml unavailable (no embedded llama.cpp)
```

## Four-leg bench — Qwen2.5-7B-Instruct-4bit

Method: identical to the run-7 two-leg bench of
[`2026-09-19-serve-options-bench-jw16.md`](2026-09-19-serve-options-bench-jw16.md)
— all servers up simultaneously on distinct ports, one warmup (16 tok)
each, then 8 interleaved rounds, greedy (temperature 0), `max_tokens`
128, prompt "Explain why the sky is blue." (37 prompt tokens on every
leg this time), wall-clock per request over `usage.completion_tokens`.
Runtime pins fatal-checked at startup: mlx-omarchy wheel
`0.32.3.dev202609190758+50eeb29` (tag v0.7.1), `libmlx.so` sha256-16
`df3d4e74c597956c`. Leg C/D used the clean-build binary above,
launched as `mlx-serve --model <snapshot> --serve` (port 11234) and
the same plus `--no-pld` (port 11235).

| leg | server | median tok/s | min–max | n |
|---|---|---|---|---|
| A | `mlx_lm.server` 0.31.3 | **10.36** | 8.15–13.10 | 8 |
| B | oMLX 0.6.4 | **33.5** | 28.74–34.01 | 8 |
| C | mlx-serve 26.9.5-dev Linux (PLD on, default) | **5.65** | 5.21–5.66 | 8 |
| D | mlx-serve 26.9.5-dev Linux (`--no-pld`) | **5.70** | 5.51–5.70 | 8 |

Every request on every leg finished `finish=length` (128/128 tokens)
and produced **identical greedy text**: "The sky appears blue during
the daytime because of a process called Rayleigh sca…". Zero errors.

Reproducibility against the earlier same-day two-leg run 7 (same
method, A+B only): A 9.33 → 10.36, B 33.6 → 33.5 medians. B agrees to
0.3%; A's whole distribution sits higher this run for reasons not
pinned down (a stray resident 0.5B mlx-serve process from the earlier
lane was alive during parts of the day and was killed before this
run; run 7 also observed late-compile pressure on A). Use the
four-leg numbers for the comparison — all four legs shared one window
— and both receipts are linked from docs/serve.md.

### The 48.7 tok/s claim, reconciled

The earlier lane's "mlx-serve serving end-to-end on jw16 at 48.7
tok/s" does not survive same-conditions measurement, and the number's
provenance explains it: 48.7 is the **Qwen2.5-0.5B** decode figure
that appears throughout the ane-linux-experiments decode-fusion
receipts (e.g. `qwen25-0.5b-4bit:longctx-1024-decode-32` median
48.74), and the stray process the earlier lane left on jw16 (PID
430201, port 8954) was serving **Qwen2.5-0.5B-Instruct-4bit**, not the
7B bench model. On the standard docs bench model (7B 4-bit, 128
greedy tokens, 37-token prompt) the Linux port decodes at **5.65
tok/s** — 1.8× slower than `mlx_lm.server` and 5.9× slower than oMLX
on identical weights and prompts.

### Why mlx-serve is the slowest of the three

- The Linux build links the standard MLX op set over the Vulkan
  backend (`mlx 0.32.3` fork + `mlx-c`). The Metal fast paths the
  macOS build uses (custom kernels, the bf16 narrow-1d table format)
  are unavailable or default-off here.
- oMLX's lead (33.5) comes from its own batched-LLM-engine kernels on
  the same libmlx — its engine, not the backend, is the difference.
- Prompt Lookup Decoding (mlx-serve's speculative decoder, on by
  default) made no measurable difference on this prompt: 5.65 (on) vs
  5.70 (off), within noise. Greedy single-stream on a short factual
  prompt gives PLD few draft hits; do not expect it to close the gap.

### Rough edges in the Linux path (all observed, not hypothetical)

- Source build only: no Linux release asset upstream (v26.9.4 ships
  macOS artifacts). Needs zig 0.17 nightly, cmake, Vulkan headers,
  libwebp, avahi `libdns_sd`, and an `mlx-omarchy` `prepare-mlx.sh`
  staged tree.
- Both required submodules must be checked out
  (`lib/mlxc-src`, `lib/opencode2-mlx-serve`); now a loud configure-
  time error instead of a mid-compile FileNotFound.
- MLX safetensors only — the embedded llama.cpp (GGUF), ds4, and ANE
  engines are compile-time stubs on Linux (same surface as the iOS
  build).
- CLI sharp edge: `--model-dir <dir>` is a discovery ROOT, not a
  selection — `--model-dir <snapshot> --serve` boots an empty server
  that answers `503 no_model` to everything. The working invocation
  is `mlx-serve --model <dir> --serve`.
- `/v1/models` reports an internal hash id, not a model name; requests
  work with `"model": "default"` (or empty) when `--model` was given.
- Context is pinned to the model card maximum (4096 for
  Qwen2.5-7B-Instruct-4bit); the Python servers negotiate longer.
- Log file interleaving: mlx-serve writes NUL padding into inherited
  stdout under some launchers (cosmetic; logs remain greppable with
  `grep -a`).

## Post-state

`llm-inference.service` active, `/health` 200, real completion:
content "Hi", `finish=stop` (44 completion tokens — the model reasons
first; a 16-token probe spends the budget in `reasoning_content`).
The stray 0.5B mlx-serve (PID 430201) is dead and intentionally not
restored. Window protocol deviation worth remembering: the service
must be restarted **after** releasing the lock — its
`flock --nonblock` ExecStart fails while the window holder still
holds `/tmp/m1-gpu.lock` (this run's first restart attempt sat
inactive for exactly that reason).

## Artifacts (this directory)

- `bench3.py` — four-leg harness (run-7 harness extended; pins
  fatal-checked; per-leg warmup + failure isolation).
- `run3.sh` — GPU-window wrapper (stop service → lock → bench →
  restart + verify).
- `results3.json` — raw rounds, pins, per-leg summaries.
- `server-C_mlxserve.log` — leg C server log (startup banner, route
  table, PLD notice, request log).
- `build-clean-checkout.log` — the clean-checkout build through
  `CLEAN_BUILD_OK`.

Verdict: **docs/serve.md updated** — mlx-serve now has a Linux build +
measured numbers and joins the comparison as a real (if slow) third
option; the macOS-only claim is corrected to a from-source Linux port
on the fork with the PR upstream.
