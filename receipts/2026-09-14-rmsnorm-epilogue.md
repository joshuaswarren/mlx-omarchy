# Fused residual-Add + RMSNorm prologue: digest-preserving on jw16, tok/s rise not established

Date: 2026-09-14
Task: fold the 49 FastRmsNormF16 decode dispatches (one workgroup each, a
prologue, per `receipts/2026-09-14-decode-gap-attribution.md`) into the
producing residual Add, or prove the pin cannot be preserved.

## Verdict

**The fold is digest-preserving — both pins match exactly — but the tok/s
rise is not established on this session's legs, so per the acceptance rule
this does NOT land.** The branch `rmsnorm-epilogue` (workstation
`mlx-omarchy`, commits `4853e18e` + `a3e9f486`) is left unmerged.

One locked leg per cell, same wheel, fold toggled with
`MLX_OMARCHY_FUSED_RMSNORM` (default on):

| leg | fold off | fold on | pinned native digest | digest result |
| --- | ---: | ---: | --- | --- |
| short 30/32 decode | 164.9322 tok/s | 167.9240 tok/s | `7fd25a869ff21678` | **match** (on) |
| ctx1053/32 decode | 144.6366 tok/s | 143.5705 tok/s | `7da83f06ec9f001d` | **match** (on) |

- short: **+3.0 tok/s (+1.8%)** — rises.
- ctx1053: **−1.1 tok/s (−0.7%)** — does not rise; below single-leg noise.

Both on-legs reproduce the pinned generated-ID digests bit-exactly, so the
original question is answered: **the RMSNorm prologue fold CAN preserve the
pin.** The mechanics: the fused kernel rounds the sum to the storage dtype
(`unpackHalf2x16(packHalf2x16(...))`, the fused_chain.comp round trip)
before squaring, uses the identical 256-thread strided accumulation and
shared-memory stride-128..1 tree as `fast_norm.comp`, and multiplies
`sum * norm * weight` in the same order. The residual-sum buffer is written
byte-identically to the standalone Add.

Why the tok/s gain is now marginal: since the 2026-09-13 baseline
(147.5 / 86.3 tok/s on this host) the environment moved — the off legs
alone measure 164.9 / 144.6 tok/s. The ~10.8 us/dispatch term-A cost the
fold priced (−0.775 ms/token estimate) no longer matches this session's
driver state; 49 saved dispatches bought +1.8% at short context and nothing
measurable at 1053. A multi-repeat median A/B is required before any land
decision; this receipt deliberately does not fake one.

## Design

Planned from the eager tape in `EagerFusionScope` (after GEMV planning, so
a GEMV-epilogue Add is never stolen). Pattern: a float16 `Add` whose output
is a single row, shape-equal to a float16 `RMSNorm` node reading that Add
output directly, with a resident weight (parameter, not a tape node).
Either claimed node firing dispatches once
(`dispatch_rmsnorm_add_pair`, `FastRmsNormAddF16`, five bindings, one
workgroup) and writes both the sum and the normalized row; any runtime
contract failure un-plans both nodes onto the per-node path. Gated by
`MLX_OMARCHY_FUSED_RMSNORM` (default on) under `MLX_OMARCHY_FUSED_GEMV` /
`MLX_OMARCHY_FUSED_CHAIN`.

Files: `shaders/fast_norm_add.comp` (new),
`compute.h`/`compute.cpp` (new kernel enum + SPIR-V table),
`fused_chain.{h,cpp}` (planner + executor + gate),
`primitives.cpp` (`dispatch_rmsnorm_add_pair`).

## Identity

- Branch: `rmsnorm-epilogue` at `a3e9f486`, parent `7f8786b0` (HEAD at
  session start), on checkout `b28deb1c`-lineage tree.
- Wheel: `mlx_omarchy-0.32.2.dev202609142212+a3e9f486-cp314-cp314-linux_aarch64.whl`,
  built on jw16mbp1-linux in worktree `/var/tmp/rmsnorm-epilogue`
  (branch push `rmsnorm-epilogue-v2`).
- Host: `jw16mbp1-linux` (Apple M1 Max T6001), Honeykrisp
  (`vulkaninfo` device `Apple M1 Max (G13C C0)`), `mx 0.32.2.dev202609142212+a3e9f486`,
  `mlx-lm 0.31.3`, Python 3.14.7 venv `/var/tmp/rmsnorm-epilogue/venv-test`.
- Model: `mlx-community/Qwen2.5-0.5B-Instruct-4bit` snapshot `a5339a41…`.
- Protocol: `scripts/bench_decode.py --tokens 32 --temp 0.0 --seed 0
  --warmup-tokens 4`, `MLX_DISABLE_COMPILE=1`, fresh subprocess per leg,
  `/tmp/m1-gpu.lock` held per leg with `flock -w 60` (never stolen,
  never unlinked). Prompts: `short` = "Hi"; `ctx1024` expanded by
  `bench_matrix.prompt_text` (1053 chat-template tokens).
- Raw logs: jw16 `/var/tmp/rmsnorm-epilogue-runs/{off,on}-{short,ctx1053}.log`.
- ANE, driver, DTS, llama, and other hosts untouched. jwm1 never used.
- Host deviation recorded: jw16 `openblas 0.3.34-1` moved headers to
  `/usr/include/openblas/`, which upstream MLX `find_path()` does not
  search; compatibility symlinks were created for
  `cblas.h f77blas.h openblas_config.h lapack.h lapacke.h lapacke_config.h
  lapacke_mangling.h lapacke_utils.h` in `/usr/include` (pointing at the
  packaged copies). This was required to build any wheel on the host today,
  including a baseline.
- Agent model: `zai/glm-5.3-flash`.

## Next step if this is picked up again

Re-run the same A/B as a 5-repeat median per cell; if the ctx1053 median
still does not rise, the fold's win is driver-era dependent and the branch
should stay parked, not merged.
