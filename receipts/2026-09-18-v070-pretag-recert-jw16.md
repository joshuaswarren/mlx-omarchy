# v0.7.0 pre-tag recertification — jw16 (M1 Max T6001) — 2026-09-18

Lane: V070Recert. Verdict: **READY-TO-TAG from main bytes.** No tag, no
publish, no upload in this lane (both-host gate on jwm1 and the publish
itself remain with the owner after the divisor lane returns jwm1).

## Wheel (the tag candidate)

Built on jw16 from a clean detached worktree of `origin/main @ b283a16f`
(latest main at build time; carries `064b7301` bf16 compiled-tape lift and
`da43969e` fused-chain DivLast fix):

- file: `mlx_omarchy-0.32.3.dev202609182317+b283a16-cp314-cp314-linux_aarch64.whl`
- sha256: `2def345c00a60c41d2f18018096720611d40ff5a3dc50845ae3e19dc59794e53`
- size: 8371886 bytes
- libmlx16: `38adf6efb80ee850` (not on the certified list before this gate;
  candidate line `v0.7.0-candidate-b283a16` appended to the gate copy only)
- path on jw16: `/var/tmp/v070-jw16/dist/`
- `63c1d3cf` non-ancestry asserted at fetch, at the worktree, and before
  this receipt (exit 1 = PASS each time).

## Pre-tag gate — GATE GREEN (run 6, 18:50:55–18:53:51 CDT)

Script: `/tmp/v070-jw16-gate.sh` (skeleton: v068 gate). Status file:
`/var/tmp/v070-jw16.status`; artifacts: `/var/tmp/v070-gate-jw16/`.

| gate | result |
| --- | --- |
| artifact checks | PASS — version identity filename/dist-info/METADATA, `b283a16` stamped in libmlx, no `MLX_OMARCHY_GPU_PROFILE`, `MLX_DISABLE_COMPILE` control present, WHEEL tag matches platform, runner bytes == `overlay/tools/coreml/vulkan_encoder.py` at b283a16f |
| packaging clean-HOME ×3 (gate-jwm1.sh) | PASS — transcript `db501a8c080380ea…a0790` exact on all three runs |
| E2E serve-default (wheel bytes) | PASS — match, subs=1, 104/104, bounds PASS, cpu_tensor_events 0, timeouts 0, transcript/mel/hidden (`38c73261…`) pins exact |
| E2E launch arm (ANE_ISLAND_MODE=launch) | PASS — match, subs=48, same pins exact |
| E2E placed-AC (MLX_OMARCHY_PLACED=AC) | PASS — match, subs=1, hidden `38c73261…` exact |
| E2E placed-ACO | PASS — hidden `ef6afd137f1610901c1bce9cf4c9e199edc430c37bd4aeb27caa4622692d5e88` exact; see ACO provenance below |
| venv-identity-guard | PASS — fresh `/var/tmp/V070REL-venv` certified against the candidate line, `PYTHONPATH` unset |
| decode legs (one flock hold) | PASS — `longctx-1024-decode-32` = `7da83f06ec9f001d`, `short-decode-32` = `7fd25a869ff21678`, provenance names the wheel libmlx16 |
| hwcap probe + KATs (deployed libmlx) | PASS — `HWCAP_SHA256=1`, 13/13 FIPS 180-4 KATs, open-probe 1783.1 MB/s |

### ACO provenance finding (recorded, not blocking)

The wheel ships only the base islands (`island-attn-a-kt`, `island-pv`,
`island-select-8head`) — no `island-oproj-L*` bundles. With pure wheel
bytes, `MLX_OMARCHY_PLACED=ACO` silently runs the o-proj on GPU and returns
the AC hidden (`38c73261…`) with every other pin holding — the first gate
run caught exactly this. The passing ACO arm above uses a merged bundle dir
(wheel base islands + `island-oproj-L*` staged from
`/var/tmp/jw16-encoder-islands/bundles`, the certified measurement source);
runner/pkg/worker/libane stay wheel bytes. ACO remains an opt-in
measurement configuration, consistent with the v0.6.2 R4 decision.

## C++ suites from main bytes (`.work/build-tests` @ b283a16f)

| battery | result |
| --- | --- |
| omarchy_primitive_tests | **103/103** (2,700,952 assertions) |
| omarchy_runtime_tests | **41/41** (22,694 assertions) |
| capability sim | **6/6** — registry unit OK + 5/5 profiles rc=0 (`m1-honeykrisp-fork`, `m1-stock-no-coopmat`, `subgroup-size-64`, `small-shared-memory`, `no-cooperative-matrix`). Note: the binary invoked with no args prints usage and exits 2 after running the registry unit — profiles must be listed individually; the storm-era `run-gates.sh` no-arg loop records rc=2 regardless of health |
| omarchy_compiled_tape_tests | **12/12** (2,096 assertions, bf16 bit-exact set incl. shapeless reuse) |
| omarchy_fused_chain_tests | 33/34 — the single failure IS the fc dispatch-count pin anomaly, A/B'd against stock below (documented, non-blocking) |

## Model matrix — compile ON vs eager (candidate wheel b283a16)

mlx-lm 0.31.3, greedy, fixed prompt "Explain photosynthesis in one
sentence.", max 96 tokens, digest = sha256(generated ids)[:16].

| model | mode | reps | n | digest | tok/s |
| --- | --- | --- | ---: | --- | --- |
| Qwen3.5-9B-MLX-4bit | compiled | 2 | 96 | `910abe30d4305271` (×2) | 11.28 / 11.27 |
| | eager | 1 | 96 | `910abe30d4305271` | 11.08 |
| gemma-4-31b-it-4bit | compiled | 1 | 96 | `9f1fe40101db3a4b` | 2.37 |
| | eager | 1 | 96 | `9f1fe40101db3a4b` | 2.45 |
| Ministral-3-8B-4bit | compiled | 1 | 1 (EOS) | `d4735e3a265e16ee` | — |
| | eager | 1 | 1 (EOS) | `d4735e3a265e16ee` | — |
| Ternary-Bonsai-8B-2bit | compiled | 1 | 96 | `25dc382d3170a80c` | 5.74 |
| | eager | 1 | 96 | `25dc382d3170a80c` | 5.64 |
| Ternary-Bonsai-2-27B (mlx_vlm gdn_sink pack, F7 lineage) | gen (compile ON default) | 1 | 96 steps | coherent: "The capital of France is **Paris**…" (`nan_at` null) | **1.44** |

**Compile ON changes no generated id on any model** (compiled == eager
everywhere; Qwen3.5 deterministic across two compiled reps). Bonsai-2-27B
is coherent at 1.44 tok/s on the main wheel — the F7 fix holds on the
candidate bytes (T6001).

Pin drift, recorded honestly: Qwen3.5 (`910abe30…`) and Ministral
(`d4735e3a…`) reproduce the 2026-09-18 bf16-receipt pins exactly. gemma
(`9f1fe4…` vs receipt `20b1cc9f…`) and Bonsai-8B (`25dc382d…` vs receipt
`66a0f9e9…`) differ from those pins while being internally consistent —
compiled == eager on both wheels tested (b283a16 and the probe wheel), and
gemma's digest is identical across fused ON, fused OFF (`MLX_OMARCHY_FUSED_CHAIN=0`),
eager, and both wheels. The drift is therefore environmental (venv-era
tokenization stack), not a compile or fusion effect. Mechanism not
investigated further in this lane; the standing release pins (decode legs,
Parakeet transcript/mel/hidden) all held exactly.

## Bf16CompiledTape open item — corrupt matrix with fusion ON

Probe wheel: `0.32.3.dev202609182351+ad89b32`,
libmlx16 `e0bcb4b23e27eeb3`, built on jw16 in throwaway worktree
`/var/tmp/v070-probe-wt` from b283a16f + exactly one local TEMP commit
removing the `node.dtype() != bfloat16 &&` condition from the tape
`try_add` fast path in `compiled.cpp`. **Measurement only — never
published, never tagged.**

| model | fusion ON (probe wheel) | digest | vs eager |
| --- | --- | --- | --- |
| Qwen3.5-9B | 11.43 tok/s, coherent | `910abe30d4305271` | identical — CLEAN |
| Ministral-3-8B | EOS artifact, as eager | `d4735e3a265e16ee` | identical — CLEAN |
| gemma-4-31b | 2.34 tok/s | `9f1fe40101db3a4b` | identical to its own eager and fusion-off digests — CLEAN |

Verdict: **the in-model fused-bf16 corruption is gone on the full corrupt
matrix with fusion ON.** `da43969e`'s DivLast leaf refusal removed the
mechanism, exactly as F7GdnCorrectness predicted. Per release scope the
fence is NOT lifted in v0.7.0 (the `known-defects.md` entry stays); **the
fence is liftable in its own commit next release** with a fresh digest
sweep like this one.

## fc dispatch-count pin anomaly — A/B vs stock (documented, non-blocking)

Case: "nonidentity broadcast keeps the per-node fallback (shapeless)",
f32, EAGER leg, `omarchy_fused_chain_tests`.

- candidate b283a16: `CHECK_EQ(eager, 1)` got **2** — battery 33/34.
- stock c136912f (throwaway worktree `/var/tmp/v070-stock-ab`, same
  build-tests flow): battery **33/33**, same case `CHECK_EQ(eager, 1)`
  SUCCESS → **1**.

Deterministic A/B: candidate 2, stock 1 — matching the bf16 receipt's
record. The eager leg does not execute `eval_compiled_tape`, so the
mechanism remains unexplained; it is documented and does not block the
release.

## jw16 window discipline

Every GPU window followed: `sudo systemctl stop llm-inference.service` →
`flock -w 900 /tmp/m1-gpu.lock` (inode 12, never unlinked, verified in
each take) → work → release → restart → `is-active` + `/health` 200
confirmed. Hand-back receipts: gate run 6 (18:53:51, health 200), C++
suites (19:0x, 200), final window (19:26-19:29, one 503 warm-up observed
then 200), window 4 (19:2x, 200 re-confirmed after the load). The window
was handed over by F7GdnCorrectness via hub after their own health-200
confirm; no collisions.

## Asset map (jw16)

- candidate wheel: `/var/tmp/v070-jw16/dist/`
- gate: `/tmp/v070-jw16-gate.sh`, `/var/tmp/v070-jw16.status`,
  `/var/tmp/v070-gate-jw16/`
- suites + matrix + probe logs: `/var/tmp/v070-cxx-suites/`,
  `/var/tmp/v070-final/` (`*.jsonl`, `bonsai2-27B-gen.log`,
  `capsim-*.log`, `stock-fc*.log`, `cand-fc-pin.log`)
- probe wheel (do not publish): `/var/tmp/v070-probe-wt/dist/`
- matrix venv `/tmp/v070-matrix-venv` (wheel + mlx-lm 0.31.3 closure),
  probe venv `/tmp/v070-probe-venv`, stock A/B tree
  `/var/tmp/v070-stock-ab`

## Not done in this lane (by scope)

- No tag, no GitHub release, no upload, no `verify-release-assets` against
  uploaded bytes.
- jwm1-side both-host arms (both-host gate) run after the divisor lane
  returns jwm1 to Asahi.
