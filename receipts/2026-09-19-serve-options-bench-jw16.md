# Serve options on the v0.7.1 runtime: mlx_lm.server vs oMLX vs ddalcu/mlx-serve — 2026-09-19

Lane: ServeOptionsDocs. Host: jw16mbp1-linux (Apple M1 Max, T6001, kernel
7.1.6-1-1-ARCH, 10 cores / 62 GB). All GPU work inside `flock
/tmp/m1-gpu.lock` windows with `llm-inference.service` stopped before and
restarted + health-checked (200) after, per the standing GPU protocol.

## Runtime pin (fatal-checked inside the bench)

- Wheel: published v0.7.1 release asset
  `mlx_omarchy-0.32.3.dev202609190758+50eeb29-cp314-cp314-linux_aarch64.whl`,
  sha256 `e536056bcb23d8d93121edc662cebb0c4b14b8670233c8f018ad54150b9b8d49`
  (verified after download), installed into a fresh venv
  (`/tmp/servebench/venv`, Python 3.14.7).
- `mlx-omarchy 0.32.3.dev202609190758+50eeb29` (= tag v0.7.1);
  `libmlx.so` sha256 prefix `df3d4e74c597956c` — matches the pin other
  v0.7.1 lanes used. The bench asserts both at startup (exit 3 on
  mismatch).
- Servers: `mlx-lm 0.31.3` (PyPI, the installer pin) and `omlx 0.6.4`
  (source install from the jundot/omlx tree; **not on PyPI** —
  `pip index versions omlx` → no matching distribution, checked
  2026-09-19).

## Headline

1. **ddalcu/mlx-serve does not run on this stack, at all.** It is a
   native Zig server against Apple's Metal MLX; upstream's own
   building doc says "The server itself is macOS / Apple Silicon only."
   Latest release v26.9.4 ships exactly two assets
   (`mlx-serve-bin-macos-arm64.tar.gz`, `MLXCore.dmg`) — no Linux
   artifact. It needs macOS 26.2+, Xcode + Metal Toolchain, and brew.
   On omarchy/Linux (Vulkan mlx-omarchy) it is not installable; the
   closest Linux-side substitutes are the two servers measured below.
   On a Mac it is `brew install mlx-serve && mlx-serve serve` on port
   11234, speaking OpenAI + Anthropic + Ollama APIs.
2. **oMLX is ~3.6× faster than `mlx_lm.server` on identical weights**
   (Qwen2.5-7B-Instruct-4bit, single-stream greedy): median 33.6 vs
   9.33 tok/s. oMLX's engine pool (batched LLM engine, its own custom
   kernels) decodes well above the memory-bound rate; mlx_lm.server's
   serve path decodes at roughly the memory-bound expectation for 7B
   4-bit on this part (the 143.6 tok/s single-stream control in the
   stall-fix receipt was the 0.5B model, ~17× smaller).
3. **`mlx_lm.server` 0.31.3 cannot serve Ministral-3-8B** on this
   runtime; oMLX serves it (VLM engine) but slowly, 3.3–3.4 tok/s.

## A/B bench — Qwen2.5-7B-Instruct-4bit (run 7, completing)

Method: both servers up simultaneously on distinct ports; one warmup
(16 tok) each; then 8 interleaved rounds A,B,A,B,…; greedy
(temperature 0), `max_tokens` 128, prompt "Explain why the sky is
blue." (37 prompt tokens on both servers); HTTP wall-clock per request
over `usage.completion_tokens`. Client timeout 120 s. Harness:
`bench.py` in this directory; raw responses in `results.json`.

| leg | server | median tok/s | min–max | n | all finish=length |
|---|---|---|---|---|---|
| A | mlx_lm.server 0.31.3 (batched) | **9.33** | 8.21–10.44 | 8 | yes |
| B | oMLX 0.6.4 (batched LLM engine) | **33.6** | 33.07–34.08 | 8 | yes |

Load-to-ready was not the differentiator (both serve first token within
seconds of start); the gap is decode rate itself. A's rounds 1–4 ran
~8.2 tok/s and rounds 5–8 ~10.3 (late compile/pressure effects at the
margin); B was flat at 33–34.

Model verification (rule: verify before pinning; checked 2026-09-19):
`mlx-community/Qwen2.5-7B-Instruct-4bit` on HF — exists, not gated,
Apache-2.0, last modified 2024-11-06, ~17.9k downloads. It is pinned
here as the **verified-loadable bench/default model**, not as current
SOTA: the September-2026 8B-class leaders per web search are Qwen3-8B,
Gemma 4 (7B-class), and DeepSeek-R1-Distill-8B; the docs' dated SOTA
paragraph already covers the current generation.

## Ministral-3-8B-Instruct-2512-4bit

HF: exists, not gated, Apache-2.0, last modified 2025-12-06.

| leg | result |
|---|---|
| B: oMLX 0.6.4 | **serves it**, median 3.35 / 3.37 / 3.33 tok/s across three independent runs (runs 3, 4, 5; n=8 each, every request `finish=length`, spread ≤0.15 tok/s within a run). oMLX routes it to its VLM engine and reports 543 prompt tokens for the same short prompt the A leg counts as 37 — the decode number therefore carries a much larger prefill. |
| A: mlx_lm.server 0.31.3 | **cannot serve it.** Failure ladder, each stage reproduced on a fresh server process: (a) request model-field hub resolution — the server hub-resolves the request's `model` string, and a registry-style name gets HF's 401 masking for nonexistent repos; (b) with the real repo id the load proceeds (tokenizer inits, hub fetch 200) and the request returns HTTP 200, but the generation thread dies: `AttributeError: 'TokenizerWrapper' object has no attribute '_detokenizer'` (`mlx_lm/server.py:788` → `tokenizer_utils.py:459`) in the mistral3/tekken path; (c) the same wheel serves Qwen2.5-7B end-to-end (run 7), isolating the bug to the mistral3 serve path. |

The (b) traceback is in `server-A-key.log` (run 6 section). This is an
upstream mlx-lm 0.31.3 serve bug on the deployed PyPI wheel, not an
mlx-omarchy regression — flag for the next mlx-lm pin bump.

## ddalcu/mlx-serve verification detail

- Upstream `docs/building.md` (fetched 2026-09-19): "The server itself
  is macOS / Apple Silicon only." Prerequisites: macOS 26.2+ Apple
  Silicon, Xcode 26.2+ with Metal Toolchain, brew (cmake, libwebp),
  Zig 0.17 nightly. Vendored mlx is built by `scripts/build-mlx.sh`
  with Metal/NAX kernels.
- GitHub releases API, latest v26.9.4: assets are
  `mlx-serve-bin-macos-arm64.tar.gz` and `MLXCore.dmg` only. No Linux
  build exists or is claimed.
- Usage (on a Mac): `brew install mlx-serve`; `mlx-serve run <model>`
  / `mlx-serve serve`; endpoint `http://localhost:11234/v1` (OpenAI),
  Anthropic Messages, and the Ollama API on the same port. Documented
  in `docs/serve.md` §"Which server?" as the Mac-only option.

## Honest notes

- Runs 1–6 of this bench burned windows on harness-side and upstream
  bugs before run 7 completed: (1) my client sent a model name the
  servers treat differently (see A ladder above); (2) my first server
  env set `HF_HUB_OFFLINE=1`, which breaks the offline snapshot
  resolution for the same lookup; (3) a run-2 server survived its
  cleanup and squatted port 8081 for two later windows (fixed with
  process-group kill + `fuser -k` port scoping); (4) run 6's B-leg
  warmup hung past its client timeout and the window was aborted via
  SIGINT (rc 130). None of these affect the run-7 numbers, which were
  produced with the fixed harness end-to-end.
- While my window was held, `llm-inference.service` crash-looped
  (auto-restart against my flock, never reaching the GPU); it settled
  to active/health-200 on release. Post-state confirmed `active`,
  `SubState=running`, `/health` 200 after every window, with
  `reset-failed` applied at the end.
- oMLX's 543-token prompt accounting on the Ministral VLM path (vs 37
  on Qwen2.5's LLM path) is reported as observed; prefill is included
  in the wall-clock tok/s for all legs.

## Artifacts (this directory)

- `bench.py` — the harness (runtime pins fatal-checked; interleaved
  rounds; per-request error capture; fuser-scoped port cleanup).
- `results.json` — run 7 raw responses + computed summary.
- `server-A-key.log` — A-server log, `[rtmod]` noise filtered: the
  run-6 detokenizer traceback and run-7's clean 200s.
- `server-B_omlx-last300.log` — tail of the B-server log.

Verdict: **serve.md updated** with the three-server comparison; the
Mac-only status of ddalcu/mlx-serve documented with upstream citations;
oMLX documented as the faster-but-source-install option; mlx_lm.server
stays the default bundled path.
