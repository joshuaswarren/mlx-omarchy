# F7 — Bonsai-2-27B GDN value corruption: root-caused, fixed, pushed

2026-09-18 · lane F7GdnCorrectness · hosts: jw16 (T6001), jw14m2 (T6021)

## Verdict

**F7 was never silicon, cache-state, or horizon — it is a fused-chain leaf-addressing
bug in the omarchy Vulkan runtime**, fixed at
[`da43969e`](https://github.com/joshuaswarren/mlx-omarchy) (origin/main):

`leaf_mode_for` (overlay/mlx/backend/omarchy/fused_chain.cpp) selected the DivLast
leaf-addressing mode from `data_size == count / last_dim` alone. For the GDN k leaf,
shape `(N, 1, L)` (mlx_vlm qwen3_5 `state * k[..., None, :]`), DivLast addresses
`leaf_flat[index / last_dim]` — indexing the leaf by the OUTPUT's second-to-last axis
instead of the leaf's own last axis. Every elementwise product of the GDN recurrent
update read `k[r, dv]` instead of `k[r, dk]`; the state update stopped converging and
amplified ~100x per step.

Failure signature chain: post-prefill state at layer 44 = 2.0e30 → x100/step through
decode → 2.6e36 at step 4 → fp32 overflow to inf → NaN logits from step 4. The
signature is identical on T6001 and T6021 because the bug is in the shared runtime.

## Fix

Refuse fusion for leaves the two packed addressing modes cannot express:

- DivLast now requires `shape.back() == 1` (column-vector broadcast, e.g. kv cache
  rows — the legitimate per-row-affine case keeps fusing).
- ModLast now requires all outer dims == 1.
- Anything else (the GDN `(N,1,L)` leaf included) falls back to per-node dispatch.

`+28/-3` fused_chain.cpp, `+27` regression test
`overlay/tests/omarchy/test_fused_chain.cpp`
("strided (B,1,L) broadcast leaf refuses fusion and matches eager").

## Evidence chain

1. Repro (round14, jw16/T6001): cache-bearing greedy NaN at step 4, top5
   `[248319, 82753, 82781, 82780, 82779]`.
2. `mx.metal.is_available() == False` on BOTH silicons → the mlx_lm/mlx_vlm GDN metal
   kernels never run; everything executes the mx-ops fallback. Driver-kernel
   translation exonerated; arm (a) horizon still run: jw16 no-cache NaN at step 4 too
   (len 21), so on T6001 cache is not the differentiator.
3. **Silicon exonerated**: the custom gdn_sink mlx_vlm (the jw16 pack-lane variant)
   installed on jw14m2/T6021 reproduces the identical NaN@4 signature; stock mlx_vlm
   0.7.1 (chunked GDN, different algorithm) runs finite on both.
4. Suspect elimination, all on T6021 custom-vlm: fused decode linears OFF → still
   NaN; `MLX_OMARCHY_NO_BUFFER_CACHE / TAPE_FULL_BARRIERS / TAPE_NO_REUSE /
   TAPE_SYNC_EVERY / FUSED_GEMV=0` → still NaN (allocator/tape lifecycle exonerated);
   `MLX_OMARCHY_FUSED_CHAIN=0` → **clean** (step 3 top5 changes to the correct
   trajectory).
5. Op-level pin: decoder-layer isolate → first-NaN layer = 44; GDN step probe at
   (layer 44, step 4): q/k/v/g/beta/state_in all finite, state_in absmax 2.6e36.
6. Captured t=3 prefill tensors (`/tmp/f7-t3-inputs.npz`, jw14m2): numpy f64 replay of
   the same formulas = 90.01; standalone eager mx = 90.01; in-graph compiled =
   10,229 — and the compiled values match the `k[r,dv]`-misindexing model to 5e-6.
7. Micro-op sweep: exp/softplus/sigmoid/rms_norm/dot products all within fp32 norms on
   the runtime — the fault is confined to the fused-chain composition.
8. Minimal repro outside the model: `_gated_delta_step_ops` (mx.compile) on saved
   tensors: eager 90.01 / compiled 10,229 / compiled with `MLX_OMARCHY_FUSED_CHAIN=0`
   90.01. 20-line repro, deterministic.

## Fix verification (wheel d268daaf+fix, 8b2ac9380d775a3cb2d9e2ea9d3d8505d03d3b68a77310f5ad03dc921dcaeaa6)

- **T6001 (jw16, /tmp/venv-f7fix, libmlx md5 e10898ba30c7)**: Bonsai-2-27B cache-bearing
  greedy 8 steps finite=1.0 throughout — token trajectory identical to T6021's fixed run
  (top5 760/6511/314/9338/369...; absmax 19.23827 vs 19.23826, fp16-noise level). The
  pre-fix cross-silicon step-0 divergence is explained by chip-dependent fused-chain
  misindexing and disappears with the fix. Full generation 96 steps:
  **`'The capital of France is **Paris**.<|im_end|>'` — coherent, zero NaN,
  1.45 tok/s** (raw greedy decode incl. post-EOS turn padding; pre-fix 2.45-3.8 tok/s
  figures were NaN-terminated early stops, not comparable). Service
  stop/restart+CONFIRM discipline held (llm-inference active, health=200).
- **T6021 (jw14m2, /tmp/f7-venv-cv)**: Bonsai-2-27B greedy 96 steps:
  **`'The capital of France is **Paris**.<|im_end|>'` — coherent, zero NaN,
  1.87 tok/s**.
- Cross-run of Bf16CompiledTape's corrupt matrix (Qwen3.5-9B, gemma-4-31b-it-4bit,
  Ministral-3-8B) with fusion ON on the fixed wheel: queued with that lane — their
  bf16 fence is expected to lift if digests hold (same mechanism family).
- Wheel identity: sha256 8b2ac938... (d268daaf base + fix), pre-built to
  avoid colliding with the bf16 lane's in-tree build; /tmp/f7-build (jw16).

## Landed

- origin/main `da43969e` "fused-chain fix: strided (N,1,L) broadcast leaves refused
  from DivLast - F7 root cause" (force-with-lease over seconds-old a7940da0, which
  held only the test: the shared checkout was on bf16-tape-gate-lift and that lane's
  fused_chain.cpp revert ate the first add; corrected immediately, lane tip restored).
- Push gate: `git merge-base --is-ancestor 63c1d3cf HEAD` exit 1 = PASS, both pushes.

## Parked (not done in this lane-pass, pointers intact)

- AC/ACO matrix + certified Parakeet E2E on a main+fix wheel: runner of record is
  /var/tmp/ParakeetE2EJw16/fused_e2e.py + arm wrappers, identity pre-gate via
  venv-identity-guard.py, pins db50a8c / 38c73261 / ef6afd13 / 5b54f4a9. The fix
  wheel must be rebuilt from main@da43969e (the verified wheel is d268daaf+fix) and
  run through the standard gates before a release cut.
- oMLX A/B mechanical rerun: unchanged, mlx_lm server hang remains the separate known
  issue; document-only per lane scope.
- Abliterated pack tok/s: adapter at jw16:/tmp/ablit (adapter.safetensors +
  refusal_dir), base 3f926b41 verified 129 sites cos=1.0 — run after the T6001 leg.
- Lift candidate for the bf16 fused-chain fence (compiled.cpp): if Bf16CompiledTape's
  corrupt matrix passes with fusion ON on this wheel, their fence can lift per
  docs/known-defects.md process.
