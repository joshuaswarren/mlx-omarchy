# Parakeet runtime ships in the wheel: installed product packaging (2026-09-16)

Branch `agent/parakeet-wheel-packaging` (base origin/main `03648747`), tip
`33084356`. Verdict: **LAND.** The installed product runs the pinned public
Parakeet reference end to end from a clean install: host packaging gates
green on this host (llvmpipe), and the aarch64 wheel passed the clean-HOME
hardware gate on jwm1 — 3x warm transcribe fully green and identical,
transcript `db501a8c…`, `encoder_hidden` `38c73261…` (npy basis), ANE
1 submit / 0 timeouts, `cpu_tensor_events` 0, kill-switch refusal explicit.

## What landed

1. **Runtime modules in the wheel.** New `overlay/tools/coreml/CMakeLists.txt`
   installs the Core ML front-end tree as `<prefix>/coreml/` (excluding
   `__pycache__` and the Swift `capture/` harness); `patches/mlx-build.patch`
   wires `tools/mlx-omarchy-parakeet` and `tools/coreml` into the build next
   to the existing info/ane-worker/coreml subdirs. The installed
   `mlx/bin/mlx-omarchy-parakeet` resolves imports exactly like the in-repo
   tools (parent-dir + `coreml/` on `sys.path`).
2. **CLI.** `mlx-omarchy-parakeet` keeps `download`/`verify` and gains
   `transcribe AUDIO [-o OUT] [--deadline-ms N] [--keep-scratch]` — the full
   E2E port of the pinned harness (`receipts/2026-09-15-tdt-gpu-loop/
   derivation/fused_e2e.py`, sha `0e38e7b1…`): FLAC decode, Vulkan mel,
   ANE island encoder via `AneIsland`+`EncoderRunner`, decoder/joint fused
   steps, GPU-resident TDT, detokenization. `download` now also fetches and
   hash-verifies the pinned audio fixture; `verify` checks it too.
3. **ANE worker spawn contract — resolved by shipping the standalone
   fd-protocol worker.** aarch64 wheel builds pass
   `-DMLX_OMARCHY_ANE_DEVICE=ON` (`scripts/build-wheel.sh`, `uname -m`
   gated), so the wheel ships the serve-CLI worker
   (`overlay/tools/mlx-omarchy-ane-worker`, the `f171a61e…` reference
   implementation) instead of the runtime-inherited-lock worker whose
   contract answers `expected STAGING_BYTES`. The vendored
   `libane/ane.h` is the canonical omarchy-ane `6fa243a…` header (fetched
   from jwm1, `f0f6848b…`), so clean wheel builds need **no omarchy-ane
   checkout**; `OMARCHY_ANE_INCLUDE_DIR` remains as an override.
4. **Pinned assets under a versioned share dir**
   (`share/mlx-omarchy/parakeet-1/`, installed only on aarch64):
   bundles `island-attn-a-kt` / `island-select-8head` / `island-pv`
   (mil-hwxc `b61de468`, byte-identical to
   `receipts/2026-09-14-encoder-split-plan/bundles`; per-file sha in the
   pin), `libane-strict.so` (omarchy-ane `6fa243ac…` +
   `LIBANE_CONFIG_STRICT_BIND`, sha `56b46234…`, fetched from jwm1 and
   verified), and `parakeet-runtime-pin.json`.
5. **Pin manifest** (`mlx-omarchy.parakeet-runtime-pin.v1`): sha256 of every
   shipped asset plus the frozen e2e expectations — transcript
   `db501a8c…` (text + sha), `encoder_hidden` `38c73261…`, 104 emissions
   with native token ids/frame indices/durations, `cpu_tensor_events` 0,
   decode control `gpu-loop`, mel sha. transcribe verifies assets before any
   execution and verifies outputs after, refusing on any mismatch.
6. **Bundle alias retired in code.** `RESIDENT_BUNDLES` and the island-B
   submit now use the canonical `island-select-8head` (the repo renamed the
   bundle in the split-plan receipt; the runner still used the staging-only
   `…-scratch417` alias). One name, one bundle.
7. **Refusals (explicit, named, no CPU/GPU-only encoder fallback):**
   missing runtime assets (non-aarch64 wheel), any asset hash mismatch,
   `MLX_OMARCHY_ANE_DEVICE=off` kill switch, missing `/dev/accel/accel0`
   (`MLX_OMARCHY_ACCEL_DEV` relocates) or non-char device, `ane` module not
   loaded, cache not verifying, non-pinned audio, drifted encoder source
   (the mil_adapter emit is cached per revision and hash-pinned to
   `ac8e9526…`), missing `numpy`/`protobuf`, and any diverged output pin.
   Island mode defaults to `resident-batch` (the green-battery
   configuration); an exported `ANE_ISLAND_MODE` wins.
8. **Docs**: `docs/parakeet.md` gains "Installed product (wheel)";
   `docs/install-omarchy.md` documents the shipped binaries and share dir.

## Host-side proof (x86_64, llvmpipe — this host)

- Wheel builds green three times (adding the new install rules did not
  disturb the release build): final dev wheel
  `mlx_omarchy-0.32.2.dev202609161804+3648747-cp311-cp311-linux_x86_64.whl`,
  8 610 905 B. aarch64-gated assets correctly absent from the x86_64 wheel
  (worker, `share/mlx-omarchy/`), coreml tree + parakeet CLI correctly
  present; installed layout asserted in a fresh venv.
- `mlx-omarchy-parakeet download` fetched and verified all 12 pinned model
  files + the fixture into a cold cache (receipt schema
  `mlx-omarchy.parakeet-download-receipt.v1`, `verified` true for every
  file, `audio_fixture.cached` true); `verify` re-hashes clean.
- `transcribe` on x86_64 refuses exactly as specified (exit 1, names the
  missing surface).
- **Partial E2E on llvmpipe (LLVM 15.0.6 lavapipe), installed runtime
  modules, pinned encoder capture standing in for the ANE islands:**
  mel tensor bytes **bit-exact** vs the golden capture
  (`bcbaa3ca…`); TDT + detokenization reproduce **104/104 emissions, token
  ids, frame indices, durations, and the pinned transcript `db501a8c…`**
  exactly. The gpu-loop control check does not pass on llvmpipe (the loop's
  designed fallback engaged with a reason); the strict `gpu-loop` pin is
  asserted only where it is certified — the Apple-GPU gate.
- Pin-manifest correction found by that run: the receipt's mel
  `5b54f4a9…` is the **.npy file** hash; the pin now records the canonical
  float32 tensor-byte hash `bcbaa3ca…` with the basis named in the pin.
- `tests/coreml/test_parakeet_runtime_pins.py` 8/8 PASS (asset↔pin drift
  guard, bundle-name coupling, pin consistency, CLI refusals);
  `overlay/tests/omarchy/coreml/test_encoder_parity.py` 7/7 PASS.

## jwm1 hardware gate (Main-cleared window, 2026-09-16): **PASS**

Branch tip `33084356`; wheel
`mlx_omarchy-0.32.2.dev202609161842+33084356-cp314-cp314-linux_aarch64.whl`
(8 307 372 bytes, sha256
`5e9b8696a578618704d9a13bd02dbcc05a70561ae7814410cf1e2ec7ea288d8e`), built
on jwm1 with `scripts/build-wheel.sh` — **no omarchy-ane checkout needed**
(vendored `ane.h`), no GPU lock held during the build.

`gate-jwm1.sh` results (fresh venv, `CLEAN_HOME` pre-seeded with the model
cache per Main's authorization and re-verified inside the gate):

- Installed surface complete: CLI, worker, `coreml/` tree, pin manifest,
  libane, all three bundles; runtime-path provenance — every path the run
  touched (libmlx, core, audio, encoder source, worker, libane) lives under
  the installed prefix or the clean HOME; no staging references.
- `download` verified the pinned cache (offline path against the seeded
  HOME; the seed itself was a cold network fetch of all 12 files +
  fixture).
- **3x warm transcribe, identical and fully green on every run:**
  status `match`; mel, `encoder_hidden`, emissions, token ids, frame
  indices, durations, transcript, `cpu_tensor_events` 0, decode control
  `gpu-loop` with fallback `null`, NaN/Inf 0 — all pins pass. Totals
  8225.6 / 8036.5 / 8191.8 ms (encoder_ane 7077.2 / 6876.7 / 7027.8 ms,
  tdt_decode ~831 ms — same family as the fold-reland receipt's 8436 ms
  median).
- ANE per run: **1 batch submit, 1 worker start, 0 timeouts**; the
  worker is the **wheel-built fd-protocol CLI**
  (sha `5c657d6c14d949f07650e3e8e67bbed3615315c212e73edbea66316b38753f74`)
  — the installed binary, not the dev-staged `f171a61e…`; libane
  `56b46234…`; mil `ac8e9526…`.
- Refusal arm: `MLX_OMARCHY_ANE_DEVICE=off` → exit 1, "ANE disabled by
  MLX_OMARCHY_ANE_DEVICE=off; the installed product has no non-ANE
  encoder path, so transcribe refuses instead of falling back".
- Lock `/tmp/m1-gpu.lock`: `flock -w 900` on every gated invocation,
  never stolen, never unlinked, free after the gate.

### The divergence that wasn't (recorded for the next agent)

The first gate arm reported `encoder_hidden_sha256` FAIL with actual
`f4dbfff3…`. Diagnosis chain: kill-switch run diverged too (not the
fold); the dev-staged `f171a61e…` worker swapped into the install still
diverged (not the worker); the original `fused_e2e.py` harness driven
entirely from the wheel's runtime reproduced **`38c73261…` exactly** (not
the environment, not the runner, not libmlx). Root cause: the receipt pin
`38c73261…` is the SHA-256 of the **`.npy` file** (header + payload),
while the checker hashed the **float32 tensor bytes** — whose hash is
`f4dbfff3…` for the very same tensor. The runtime was pin-exact from the
first run; the pin record now carries both bases (tensor-byte hash as the
checked value, `.npy` file hash for cross-receipt verification), and the
mel pin was corrected to the tensor basis the same way (`bcbaa3ca…`;
its llvmpipe reproduction on this host is bit-exact against the golden
tensor).

### jwm1 artifacts

Gate run: `/var/tmp/parakeet-gate-out/` (gate.log, run-1..3, killswitch,
refworker, harness + the collector `/var/tmp/jwm1-collect.py`); wheel
worktree `/var/tmp/parakeet-wheel-gate` at `33084356`; seeded HOME
`/var/tmp/parakeet-gate-home`. Scratch venvs `/var/tmp/parakeet-gate-venv2`
(note: its installed worker was swapped to `f171a61e…` for the isolation
arm — not the wheel binary) and the gate's own mktemp venvs were removed
with their WORK dirs.

## Rules honored

`63c1d3cf` not merged, not touched. No release cut; branch lands via Main.
GPU lock untouched on this host (no lock exists here; all jwm1 runs wait
for the window).
