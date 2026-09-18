# Runtime modernize: prepared-source bump 0.32.2 -> 0.32.3 main (jw16 device verify)

Branch `rtmod/source-bump` (base main `86c9deb8`), tip `71d73a41`.
Candidate wheel: `mlx_omarchy-0.32.3.dev202609181256+71d73a41-cp314-cp314-linux_aarch64.whl`
(jw16 `~/src/mlx-omarchy-rtmod2/dist/`). Scratch venvs on jw16: `/tmp/venv-cand`
(candidate wheel + mlx-lm 0.31.3 + mlx-vlm 0.7.1), `/tmp/venv-base` (certified
v0.6.7 release wheel `+fb649d8`, for A/B), `/tmp/venv-main` (candidate wheel +
mlx-lm git `872ae88`), `/tmp/venv-omlx` (candidate wheel + mlx-lm `872ae88` + omlx 0.6.4).
Weights cache: `~/.cache/huggingface` on jw16 (~80 GB, all matrix models).

## 1. Target choice

- Upstream pin: `ml-explore/mlx` main tip `59d600b5e64c238427d0f8d897ab7c682ef4d3d2`
  (2026-09-17, in-dev 0.32.3). One month past v0.32.2, 65 commits.
- mlx-lm PyPI latest IS 0.31.3 (no newer release exists; frontier is git main
  `872ae88d1fac77350db23c8c04fe8dd372a9e3e8`, what oMLX main pins). The repo's
  0.31.3 pin is current, not stale. mlx-vlm latest is 0.7.1 (newer than the
  0.6.3 the Bonsai runtime pins; `mlx_vlm.load` imports and the text path works).
- LoaderFeasibility finding (2026-09-18 receipt): NO currentgen arch requires a
  source newer than 0.32.2, so this bump is CURRENCY, not a load requirement.
  `mlx-vlm 0.7.1` declares `mlx>=0.32.2`; a PEP-440 parse of the old
  `0.32.2.dev…` local version is BELOW that floor, and `0.32.3.dev…` is above
  it, so the bump also cleans up the mlx-vlm dependency story.

## 2. Delta and port

Framework-level changes that touch the backend boundary:

- `fast::GatedDeltaUpdate` (gated delta nets, Metal kernels upstream): new
  `Custom` fast primitive with per-backend `use_fallback`. Omarchy wiring in
  `overlay/mlx/backend/omarchy/primitives.cpp`: `use_fallback(...) -> true` +
  `OMARCHY_UNSUPPORTED_MULTI(GatedDeltaUpdate)` — GDN layers run the composite
  fallback (pure implemented Vulkan ops), exactly the no_gpu posture. No fused
  Vulkan kernel was added; that is the follow-up backend work item if GDN
  models need fused speed.
- `array` assignment/release cycle fix + new `mlx/memory.cpp`
  (`get_array_buffer_size`): framework-level, no backend impact.
- `mlx/linalg.cpp` gained `check_cpu_or_cuda_stream`; patch
  `mlx-linalg-gpu.patch` rebased onto the new layout (hunk re-anchored, the
  GPU-refusal rewrite in `check_cpu_stream` unchanged in spirit).
- io/load, gguf, fast.cpp, ops.cpp, python bindings: covered by the existing
  patch series; all 13 patches apply `--fuzz=0` on the new vintage.

## 3. Device load matrix (jw16, real Vulkan GPU, lock protocol)

Probe: load + greedy generate (prompt "The capital of France is", 16 tokens).
Full JSON lines: jw16 `/tmp/rtmod-matrix.log`, `/tmp/rtmod-round2.log`,
`/tmp/rtmod-round3.log`.

| model | arch | result |
|---|---|---|
| mlx-community/Qwen2.5-0.5B-Instruct-4bit (regression) | qwen2 | GENERATED (load 0.48 s, gen 0.24 s, coherent text) |
| mlx-community/Ministral-3-8B-Instruct-2512-4bit | mistral3 | FAILED: Vulkan timeline counter watchdog, see F1 |
| prism-ml/Ternary-Bonsai-8B-mlx-2bit | qwen3 (2-bit) | FAILED: same watchdog, see F1 |
| mlx-community/gemma-4-12B-4bit | gemma4_unified | NOT-LOADED on 0.31.3 ("model type not supported"); loads on mlx-lm `872ae88`, then hits bf16 tape refusal, see F2 |
| lmstudio-community/gemma-4-E4B-it-MLX-4bit | gemma4 | NOT-LOADED: upstream mlx-lm KV-shared-layer bug, see F3 |
| lmstudio-community/gemma-4-26B-A4B-it-QAT-MLX-4bit | gemma4 | FAILED: bf16 compiled-tape refusal, see F2 |
| mlx-community/gemma-4-31b-it-4bit | gemma4 | FAILED: bf16 compiled-tape refusal, see F2 |
| mlx-community/Qwen3.6-27B-mxfp4 | qwen3_5 | FAILED: `Take` dtype uint8 not implemented, see F4 |
| prism-ml/Ternary-Bonsai-2-27B-mlx-2bit | prism_hadamard_qwen35 | see section 4 (headline) |

## 4. Headline: Ternary-Bonsai-2-27B

- The pack's own `runtime/artifact.py` `load_model` self-rejects the pack:
  config carries `schema_version: 2` while that entry point demands 1. This is
  a pack-internal v1-preview vs v2 entry-point skew, NOT a runtime failure.
- The reviewed revision of the pack loader is pinned by Bonsai-demo
  (`scripts/bonsai2-runtime.sha256`); the pack's four runtime files match it
  byte-for-byte, and the canonical entry is `vision_artifact.load_vl_model`
  (schema-2 aware) driven by `scripts/mlx_generate_bonsai2.py`. Round 3 runs
  that canonical path on the candidate wheel: result recorded below in
  section 8 (appended after the run).

## 5. Parakeet pin regression — PASS on the bumped source

`mlx-omarchy-parakeet download` / `verify` / `transcribe` on the candidate
wheel, jw16, ANE islands on `/dev/accel/accel0`:

- `verify`: 12 files + pinned audio fixture OK.
- `transcribe`: `status: match`, `checks_failed: []`, 104 emissions, pipeline
  7866 ms, transcript "He hoped there would be stew for dinner, …"
- `encoder_hidden.npy` sha256 `38c73261f29230276ed76f1fc017b76b024156d79218bd5f1347fdc7e7d43ec7`
  — EXACT match to the certified pin (`38c73261…`) recorded in
  `receipts/2026-09-16-parakeet-wheel-packaging.md`.

The certified encoder pipeline is intact on the 0.32.3 vintage.

## 6. Findings (backend work items, none introduced by this bump)

- F1 Vulkan timeline watchdog: first big-model eval stalls
  ("timeline counter failed to advance for 10000 ms (last observed=0,
  target=1)"). A/B: reproduces IDENTICALLY on the certified v0.6.7
  (`+fb649d8`, 0.32.2-based) wheel — standing issue, not a bump regression.
- F2 bf16 compiled-tape refusal ("bf16 fragments corrupt nondeterministically
  on Honeykrisp") blocks every gemma4 generation attempt; deliberate gate, do
  not weaken — needs bf16 tape correctness work or an eager path decision.
- F3 E4B checkpoint ships per-layer k/v for KV-shared layers; mlx-lm
  (0.31.3 AND git main line 185) rejects 126 params. Upstream mlx-lm bug —
  file upstream; not fixable by source or backend.
- F4 `mx.take` with uint8 indices/indices-dtype unimplemented on Vulkan —
  blocks Qwen3.6-27B-mxfp4 (mxfp4 dequant path). Missing primitive dtype.
- F5 Bonsai-2 pack loader entry skew (v1 `load_model` vs v2
  `load_vl_model`), section 4.
- gemma4_unified arch requires mlx-lm > 0.31.3 (loads with `872ae88`).

## 7. Host protocol receipts (jw16)

- GPU take/release under `/tmp/m1-gpu.lock` (flock fd 9) for every GPU
  session; llm-inference stopped before, restarted + CONFIRMED after:
  `active`, `health http 200`, real completion returned (round 1 and round 2).
- Round 3 restart+CONFIRM appended in section 8.

## 8. Round 3 results

- **Bonsai-2-27B (headline)**: canonical loader works on the candidate wheel —
  `vision_artifact.load_vl_model` loaded the pack in 10 s (manifest-verified
  loader, 27B ternary + vision tower resident). Generation then stopped at the
  FIRST GDN layer with the exact op error Joshua asked to capture:
  `RuntimeError: [omarchy] Select dtype is not implemented for the Omarchy
  Vulkan backend (dtype=int64, shape=[1,21]). No GPU kernel exists for it; no
  silent CPU fallback occurs.` — thrown from `mlx_vlm/models/qwen3_5/
  gated_delta.py:375` (`gated_delta_chunked` -> `mx.async_eval(Y, S)`), i.e. a
  `Select` over int64 inside the chunked gated-delta path. The backend
  supports int64 storage broadly (ElementwiseI64/U64, ReduceGeneralI64/U64)
  and `omarchy_shader(select_complex64 shaders/select.comp -DUSE_U2=1)` shows
  Select already has a dtype-agnostic two-word (uvec2) compiled variant, so
  Select-I64 is a small follow-up kernel-mapping task, not new machinery.
  Becomes backend work item F6.
- **E2B** (`lmstudio-community/gemma-4-E2B-it-MLX-4bit`): NOT-LOADED — same
  upstream KV-shared-layer loader bug as E4B ("Received 140 parameters not in
  model: layers.15.self_attn.k_norm..."), so no 8 GB-tier gemma4 pick exists
  via E2B/E4B until F3 is fixed upstream; 12B loads (via mlx-lm `872ae88`) but
  hits F2; 26B-A4B-QAT hits F2.
- **Server A/B (oMLX vs mlx_lm.server)**: NOT COMPLETED — mechanical probe
  failures, no model verdict. mlx_lm.server answered 404 (my probe used
  `/v1/chat/completions`; server on this build exposes a different route set —
  request paths need re-checking against `/tmp/ab-mlxlm-server.log`). omlx
  bound to :8082 only after the probe window (model load exceeded the wait
  loop; server log confirms `Binding server at http://127.0.0.1:8082`). Both
  servers are staged and the harness (`/tmp/run-round3.sh` section E) needs a
  longer readiness wait + correct mlx_lm route; rerun is cheap.
- **Gate suite**: build of `omarchy_*` test targets currently fails to
  COMPILE in `overlay/tests/omarchy/test_matmul_family.cpp` — upstream added
  an 11th `global_scale` parameter to `quantized_matmul` (gather_qmm work),
  and the test's positional argument lists mis-bind into it. Wheel/release
  build is unaffected (built with tests OFF, succeeded). The test update +
  full gate rerun (G1-G5 posture) is the remaining pre-land step for this
  branch; not silently skipped, recorded here.
- Host protocol: llm-inference restarted after round 3 and CONFIRMED
  (`active`, `health http 200`); GPU lock released.

## 9. Verdict

- Source bump: LANDED on branch, clean patch series, candidate wheel built and
  device-proven for the regression path (Qwen2.5 GENERATED) and the full
  Parakeet certified pin (exact `38c73261` encoder_hidden match). No publish,
  no push; nothing outside the branch and jw16 scratch was modified.
- Currentgen generation on Vulkan is blocked by four standing backend items
  (F1 watchdog, F2 bf16 tape, F4 Take uint8, F6 Select int64) — all
  reproduced/qualified on device with exact errors, none introduced by the
  bump (F1 A/B-verified on the certified 0.32.2 wheel).
