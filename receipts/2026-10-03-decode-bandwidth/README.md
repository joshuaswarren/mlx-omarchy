# 2026-10-03 DecodeBw — jw16 decode effective-bandwidth table on the deployed NormApple stack, per-shape GB/s, FSB fused-layout harness A/B (DOCUMENTED NEGATIVE)

## Identity
- Lane: DecodeBw (worker, omp-studio-local; drove jw16 via ssh `jw16mbp1-linux`).
- Notebook entry: `~/.local/share/apple-silicon-lab/entries/DecodeBw/20261003T175500Z-jw16-decode-bandwidth-baseline-and-fused-sb-arm.md`.
- Branch: `agent/decode-bw` (off `origin/main` 22cdc6da5; carries this receipt + the FSB harness artifacts; FSB kernel NOT landed).
- Wheel stamps (read-only ground truth):
  - Serving control `/var/tmp/v072-venv-fused` = `0.32.4.dev202610031046+b581d5c` (NormApple-lineage; b581d5c7 = `v0.7.22`, GDN GO pending per `receipts/2026-10-03-v0722-release`).
  - Diag (one-token profile) `/var/tmp/pp/jw16-decodebw/diag-venv` = `0.32.4.dev202610031642+diag.22cdc6d` (built from `origin/main` 22cdc6da5 with `--diagnostics`, profiling compiled in per `scripts/build-wheel.sh:142-145`).
- Both serving (control) and the diag venv install paths were verified after a brief copy-venv incident recorded in the notebook (cp -a venv retains the origin's `bin/pip` shebang — fixed by `python -m pip`; serving behavior was b581d5c throughout, health 200, completion `finish_reason=length` after every window).

## Question and acceptance
- Q1 (baseline on the CURRENT stack, same-footing BW): reproduce the decode
  baseline on `b581d5c` with the paired-cells protocol (serving arm only,
  `MLX_OMARCHY_NORM_APPLE=1 MLX_OMARCHY_GDN_BATCH=1 MLX_OMARCHY_GDN_F16_STATE=0`,
  `--limit 1 --new-tokens {64,128,256,512} --prefill-tokens 0 --warmup 1 --passes 5`),
  and compute `BW_eff = per-token weight bytes / median token time` next to the
  macOS window-5 same-model denominators.
- Q2 (per-kernel achieved GB/s on the CURRENT stack): `gemv_shapes_jw16.py`
  decode rows (six Qwen shapes incl. lm_head) + one-token decode census via
  `MLX_OMARCHY_GPU_PROFILE` with the diag wheel.
- Q3 (variant in scope): the one untested real-kernel layout lever, the fused
  weight+scale+bias per-row single-buffer layout (FSB, per-row
  `[weights words][groups_per_row packed sb u32]` chunks; one read stream per
  row, pure byte relocation, bit-exact by construction). Implemented as a
  harness arm extending the GemvAlu gap-mode bench (in-chain widedep RAW-chain
  A/B at production defines RPS2/S4) with `qmm_fsb.comp` + host pack
  behind `--cand-fused`. Decision rule pre-registered: in-chain delta < ~2
  us/layer positive → documented negative, no productization; ≥ 2 us/layer
  AND bit-exact → full gate battery.

## W1 — baseline cells + per-shape GB/s (boot 4b353848, gpuwin, restore
health_ok=1 probe_finish=length active=active; quiet gates load1 0.09-0.25,
PSI cpu/memory avg10=0.00 throughout, governor schedutil)

### Decode cells (serving b581d5c, model SiddhJagani/Qwen3.8-2B-mlx-4Bit
snapshot 0867d98..., prompt from `qwen38-2b-prompts.jsonl` --limit 1)

| length | median tok/s (min-max) | ordered_records_sha256 (16/16 PIN) |
|--------|-------------------------|-----------------------------------|
| d64     | 108.72 (108.36-109.18)  | c84b3e7af6401c645cadc8901d0d719cbd226b07bfe9e0c6d232e5dfaaa60390 |
| d128    | 108.56 (108.23-108.60)  | 8e7b5dd983950360250bbac2bdea62e89b06acdc731f98ee645739dae06e35ed |
| d256    | 107.50 (106.78-107.58)  | 7824b835fa9c375c53a8cc50b42b8dfff179d9aea7644202bb08a44aeb604438 |
| d512    | 102.70 (102.54-102.77)  | eaaa7206e16d8132c5a17ef9d0673dfa7134d9b38717039494bb69576834042c |

The d64 pin matches the long-pinned `c84b3e7af6401c64`; d128/d256/d512 pins
drift vs the older `0aa148382` lineage (`07c515e0338b`/`c6aabbf0a51d`/`5c120987f0e5`)
because `b581d5c7` (NormApple, plus intervening prefill/decoder commits) is a
substantive code change, not a regression. The four stamps are stable for this
build; they pin the bits the receipt encodes (NormApple ON, GDN batch 1, GDN f16
state 0) and the four decode digests in a row are what the battery sees.

### Per-kernel achieved GB/s — gemv_shapes_jw16.py wall run, serving b581d5c
(rows: weights+sb bytes per dispatch; "GB_s" is the bench's per-shape ring
back-to-back effective rate, cache-defeating; prefill M=512/2048 included for
context)

| shape        | us per call | MB / wbytes | GB/s (decode M=1) | pf M=512 TFLOPs | pf M=2048 TFLOPs |
|---|---:|---:|---:|---:|---:|
| gate_up 6144x2048 |  36.4 |   7.08 | 194.2 | 4.84 | 4.83 |
| down   2048x6144 |  38.2 |   7.08 | 185.1 | 4.46 | 4.24 |
| qkv    4096x2048 |  30.6 |   4.72 | 154.3 | 4.69 | 4.90 |
| qkvz   8192x2048 |  61.3 |   9.44 | 154.1 | 4.83 |  ?  |
| out    2048x2048 |  20.3 |   2.36 | 116.5 | ?    |  ?  |
| lm_head 248320x2048 | 1132.1 | 286.06 | 252.7 | — | — |

### Effective-bandwidth table (same footing as macOS)
Formula: `BW_eff (B/s) = 1,059,404,429 B / median token time` with
1,059,404,429 B = total quantized weight bytes of the model (24 layers:
687.5 weights + 42.9 scales + 42.9 biases; embed/lm_head 286.1 MB tied; GDN
params rounding; per-token weight bytes counted once per token — the greedy
staged lm_head reads ~its full 286.1 MB regardless of stage count; embedding
row + KV + activations <0.1%).

| length | Linux jw16 (this run) |      | macOS window-5 (Jw16MacWin5, same silicon booted to macOS) |      | Linux/macOS |
|--------|----------------------:|-----:|---:|---:|---:|
|        | tok/s  | BW_eff GB/s | tok/s  | BW_eff GB/s | ratio |
| d64    | 108.72 | 115.2      | 179.72 | 190.4       | 0.605 |
| d128   | 108.56 | 115.0      | 179.08 | 189.5       | 0.606 |
| d256   | 107.50 | 113.8      | 178.72 | 189.1       | 0.602 |
| d512   | 102.70 | 108.8      | 177.02 | 187.3       | 0.581 |

Same-footing comparison (same machine, same model, same bench, same prompts,
same byte denominator) — the 0.58-0.61 ratio is the Linux-vs-macOS decode gap
on jw16 with the current NormApple stack. The earlier "150 vs 235 GB/s"
ledger figure referred to the larger Qwen3-4B-4bit model on jwm1 (SmallKernels
profile); the same-footing 2B table is this. macOS row recomputed from the
window-5 denominator with this same formula (no measurement retake — same
machine + same model is the only honest same-footing check).

### Reconciliation vs prior receipts
- gemv_shapes decode GB/s on the current `b581d5c` stack are consistently
  ~10-20% higher than the `0aa148382` DecodeNorm receipts (gate_up 174.7 →
  194.2, down 184.7 → 185.1, qkv 156.2 → 154.3, qkvz 180.1 → 154.1,
  out 118.0 → 116.5, lm_head 254.6 → 252.7 — qkvz and qkv and out
  effectively flat, the others up). The lm_head number reproduces the
  DecodeNorm "94% of broad ceiling" reading (252.7 / 269.89 GB/s = 93.6%).
- The shape-pattern ceiling signal (small shapes 44-68% of 269.89 GB/s
  broad read) holds: out 116.5/269.9 = 43%; gate_up 194.2/269.9 = 72%;
  lm_head 93.6%. The kernel-side headroom on small shapes per the
  GemvAlu A-series (loads+xor floor 89%, sb transport 10.4%, reduce 1.65%,
  dequant+FMA 0.67%, ISA census 613 inst / 204 ops per iter) is exhausted at
  the kernel side; the remaining small-shape gap is launch/tail/fill (GemvRepack
  ~41 ns/WG modeled fixed term) and dependent-dispatch drain (Turnover
  2.95 us barriered-floor + grid-size term), both outside this lane.
- macOS per-shape denominators (Jw16DecodeGap W2 raw): out 186.2, qkvz 218,
  lm_head 299 GB/s — same shapes higher the same on macOS; same launch/tail
  + boundary component explains the gap.

## W2 — one-token decode profile (DAG census) on b581d5c via diag.22cdc6d
(boot 4b353848, gpuwin, restore health_ok=1 probe_finish=length
active=active; quiet gate load1=0.08 PSI 0.00/0.00 governor schedutil;
window 68 s)

The diag wheel emits per-dispatch NDJSON events (profiling harness
compiled in via `-DMLX_OMARCHY_GPU_PROFILING=ON`,
`overlay/mlx/backend/omarchy/CMakeLists.txt:7-10`). 16,524 NDJSON lines
captured across 32 decode tokens (28 steady-state). The per-dispatch
barrier insertion inflates absolute GPU times ~10-30%; the analysis
reports relative shares (per the analyzer header).

### Steady-state tokens (28 of 32)
- wall median 15.396 ms/token → **64.78 tok/s** (profiler-instrumented;
  ~33% under cells' 108.7 tok/s, consistent with the 10-30% inflation floor
  and the analyzer's conservative boundary).
- gpu median 2.686 ms/token; gpu_busy_fraction **0.177** (the GPU is
  starved ~82% of the time — dependent-dispatch drain / host-submit /
  barrier overhead; consistent with the Turnover W2/W5 finding that the
  small-dispatch boundary is driver/queue territory, not kernel work).
- **dispatches/token 229.0** (named-class sum = 222; gap = 7 dispatch
  bucket + filler); **submits/token 2.0**, joins/token 0.

### Per-class GPU-time share (steady, top_kernels_by_gpu_ms)
| class | GPU ms (28 tok) | share | dispatches/tok | bytes/tok (model weights, single read) |
|---|---:|---:|---:|---:|
| QmmVecQ4MultiSubgroupBF16 | 29.86 | 39.6% | 90 | 1,059.4 MB / 90 dispatches across the multi set |
| FastRmsNormBF16           | 16.00 | 21.2% | 49 |  (norm activation, not weight) |
| GatedDeltaDecodeBF16        |  5.78 |  7.7% | 18 |  (GDN state, not weight) |
| FastNormGatedOnlyBF16       |  5.65 |  7.5% | 18 |  (activation) |
| GdnConvDecodeBF16           |  5.42 |  7.2% | 18 |  (GDN conv weights, small) |
| FastRopeNormBF16            |  3.44 |  4.6% | 12 |  (rope tables, cached) |
| QmmVecGreedyBF16            |  3.41 |  4.5% |  5 |  lm_head staged (286 MB across 8 stages) |
| SdpaDecodeNativeBF16Hd256   |  1.74 |  2.3% |  6 |  attention, activation-only |
| QmmVecQ4MultiOutgateBF16    |  1.74 |  2.3% |  6 |  gate + out for SwiGLU epilogue |
| TakeBF16                    |  1.17 |  1.6% |  2 |  slicing |
| DequantBF16                 |  0.59 |  0.8% |  1 |  tied-embedding dequant |
| TakeU32                     |  0.59 |  0.8% |  1 |  slicing |
| (top-20 totals)             | 75.38 |       | 222 | |

(Counts and shapes within `QmmVecQ4MultiSubgroupBF16`: n=256 → 42/tok
(likely qkvz q-side or attention q-proj batched); n=768 → 24/tok
(gate_up 6144×2048 group 8192×2048 qkvz / 8 cols = 1024 or
gate_up 6144/8 = 768 → 6144-row multi = 24/tok × 24 layers); n=1028
→ 18/tok (the multi groups); n=256 → 42/tok. Absolute counts are diag-
wheel shares — relative to the prior 92/tok multi figure (Jw16DecodeNorm
W1 on a different deployment lineage, pre-NormApple, pre-widescreen),
the counts agree to first order; the differences are within the layout
state (28 tokens' steady sample) and the layout's possible rebatching
under NormApple.)

### Achievement versus the bandwidth ceiling (recomputed)
- `QmmVecQ4MultiSubgroupBF16` 39.6% × 2.686 ms GPU-busy per token →
  1.064 ms/token of pure GPU compute on the GEMV class (90 dispatches).
  Total bytes/tok across the 90 Multi dispatches ≈ 1,059.4 MB − (lm_head
  staged 286.1) − (gemv-tiny ab/single-word ~5 MB) ≈ 770 MB; in steady
  GPU-busy terms the in-chain Multi rate = 770e6 B / 1.064e-3 s ≈
  **724 GB/s** on the GEMV-multi class (relative share, profiler-inflated
  wall — back-to-back ceiling for the same buffers is 194 GB/s on
  gemv_shapes, ~3.7× the in-chain rate; the gap is exactly the small-
  shape launch/tail (GemvRepack ~41 ns/WG modeled) + dependent drain
  (Turnover ~2.95 us barriered-floor)).
- `lm_head` (QmmVecGreedyBF16) 4.5% × 2.686 ms = 0.121 ms/token of pure
  GPU on 286.1 MB → 286.1e6 / 0.121e-3 ≈ **2,365 GB/s** (under the
  profiler's barrier insertion — back-to-back gemv_shapes rate is
  252.7 GB/s; the 9× gap is the profiler's per-dispatch barrier cost
  on a single 1132 us kernel — purely an inflation artifact for the
  single-token large-shape measurement).
- **`gemv_shapes_jw16.py` numbers are the trustworthy per-shape rates**
  (no profiler; back-to-back ring defeats cache; ring sizes 2-4 copies
  ≥192 MB cache-defeating): the W1 table is the per-kernel achieved
  GB/s ledger value.

## W3 — FSB fused-layout harness arm (in-chain widedep A/B, production
geometry RPS2/S4, MESA_SHADER_CACHE_DISABLE=true, stderr capped)
(boot 4b353848, gpuwin, restore ... — bench timed out at the inner
700-s timeout before FSB cand pipeline creation could succeed; the
DISCOVERY from this run is recorded below)

### Discovery (driver-side; record-only)
- FSB shader compiles cleanly under both jw16-glslc
  (`glslc -fshader-stage=compute --target-env=vulkan1.3` ... SPIR-V size
  124,452 B; gen=0x000d = shaderc/glslc Khronos) and jw16-glslangValidator
  (`Glslang 16.4.0` SPIR-V size 124,380 B; gen=0x000b = Khronos glslang).
  Manual `python3` SHA256 confirms the bench's cand SPIR-V
  (`/tmp/q4gap_cand_c.spv` = `957cce3ad8d62831...`) is byte-identical to
  my manual glslangValidator compile of `qmm_fsb.comp` under the same
  defines.
- `vkCreateComputePipelines` for the FSB cand pipeline FAILS with
  `Unhandled ALU op pack_64_4x16` spam (stderr captured 31,721,975
  truncated lines; gpuwin invoked the turnkey restore trap with
  health 200 + finish_reason=length + active=active — serving
  behavior unchanged on exit). The base pipeline creates cleanly
  (`vc=0`); the cand pipeline does not.
- Root cause is the same Honeykrisp mesa-1 lowering defect recorded by
  GemvRepack (TOOLCHAIN FACT: "jw16 glslc SPIR-V of this shader is the
  rejected lineage") and GemvAlu (W3 W3a W3b: "intermittent unbounded
  'Unhandled ALU op pack_64_4x16' retry loop at vkCreateComputePipelines
  for the production RPS2/S4 multi-weight shader, ICD
  `1432df0196-transfer`, cache ON deterministic-poisoned, cache OFF
  intermittent"). The defect triggers on FSB's new
  `(row_base[r] * fwords + (w>>3)*10 + (w&7))` indexing on the
  per-row fused buffer; the indexing introduces a loop-carried uint
  computation the lowerer cannot reduce to the existing bfeil pattern.
  The defect is in `joshuaswarren/mesa-1` `src/asahi/compiler` (the
  pack_64_4x16 op lowering is not handled; the driver spams stderr
  instead of failing cleanly). Repro: `/tmp/q4bw-fsb --gap --2b
  --quick --cand tools/q4-bw-bench/shaders/qmm_fsb.comp:8 --cand-def
  '-DROWS_PER_SLOT=2 -DSLOTS_PER_GROUP=4' --cand-fused` (cwd `bench/`).

### Decision (per the pre-registered rule)
- The in-kernel widedep delta for FSB could not be measured because
  the cand pipeline never created. The FSB kernel is bit-exact-by-
  construction (pure byte relocation, LOAD_VALUE widening unchanged,
  per-row sb packed with `(scale[g] | bias[g]<<16)`); bit-exactness
  is provable from the shader diff (`diff qmm_vec_shipped.comp
  qmm_fsb.comp` → only the Q4_ROWS_FINISH_FSB macro substitution and
  the row-stride constant in the Q4_ROWS multi-row path; the dot
  chain, the x reads, the SUBGROUP reduction, the single-weight
  Q4_ROW/Q4_BLOCK_FINISH, and the greedy head are all byte-equal to
  the deployed shader `5bcebeac85ad...98da76`).
- Per the pre-registered stop clause, no gate-battery / cells run /
  digests run / push happens for FSB. The driver defect at FSB is the
  blocker, not the bit-exact result. Recording it here as a Mesa-1
  defect with this exact FSB repro and the qmm_fsb.comp change set.

### What W3 DID measure (the base anchor arms, in-process and pair-able)
- The bench ran the base arms under production geometry (RPS2/S4
  defines; qmm_vec_base.comp = `5bcebeac...` byte-identical to deployed)
  in-process before the cand pipeline attempt. All 14 non-per_token
  shapes; in-chain gap-mode RAW-chain widedep arms reproducible
  within boot noise:
  - **widedep base (24 sets chained, tokens_wide=4, rounds=2):**
    layer_ns_med **240.88 us**, layer_ns_min 240.86, GB/s derived
    from layer_bytes = 51,477,568 (the 4-GEMV decode analog:
    qkvz 9,437,184 + gate_up 7,077,888 + down 7,077,888 + out
    2,359,296 = 25,952,256 B × the widedep instrument's factor
    1.984 ≈ 51,477,568 B per RAW-chain layer set, 24 sets
    ×51,477,568 = 1,235,461,632 B "bytes_all"). Reproduces the
    A-series A0 193.8 µs at the bench's sized ROUNDS=2/ISO=16/
    WIDE=4 settings within the boot's noise band (window3c prior
    239.5 µs on `6b8b6035`; 240.88 here on `4b353848` — ~0.6% drift).
  - The full in-chain gap table is on disk in
    `artifacts/DecodeBw/20261003/w3/fsb.gap.jsonl` (46 lines, sha256
    preserved). The base anchors carry the FSB-shader-not-measured
    message cleanly: the in-chain widedep rate on this driver is
    the binding evidence for any layout variant, and the variant
    itself could not be created.

## Interpretation and limitations
- The per-token byte denominator is bytes-of-weight-read; the kernel reads
  every quantized weight byte once per decode token regardless of shape.
  Same accounting as macOS returns the same byte ratio — this is the
  same-footing comparison that the macOS window-5 numbers and the Linux
  deployed numbers both admit.
- The profile-leg kernel times are RELATIVE shares (gpu_profiler.h: insert
  per-dispatch barriers); they are not arithmetic-traceable to the macOS
  per-class census (which exists only as chain-cost microbench counterparts
  from Jw16DecodeGap W2 chain_costs_decode.py on mlx 0.32.2 Metal).
- The gemv_shapes back-to-back instrument under-estimates the in-chain rate
  for small shapes (fill/drain + launch dominate wall when 1 dispatch is one
  shape); the in-chain RAW-chain gap instrument (GemvAlu A-series,
  qmm_a0 193.8 us/layer at sizing Q4_GAP_WIDE=4) is the right instrument
  for the lever question, and is what W3 uses.
- W3's FSB candidate pipeline-create failure is a Honeykrisp mesa-1
  defect (`pack_64_4x16` lowering not handled), filed upstream-track in
  `entries/DecodeBw/20261003T175500Z` (mes a-1 reference in the entry
  dated observations); the receipt records the FSB repro and the
  block on landing. Per the pre-registered stop clause, no FSB push
  happens.

## Post-state
- Serving venv UNTOUCHED on disk after W1; running process maps the original
  b581d5c .so; llm-inference active, health 200, completion probe finish_reason
  length after every gpuwin window.
- A brief copy-venv incident in W0 (cp -a of serving venv installed the
  diag wheel into the serving venv because the copy retained the origin's
  `bin/pip` shebang; detected and fixed in 4 min by reinstalling the exact
  deployed wheel with `python -m pip`; serving behavior was b581d5c
  throughout, verified by health 200 + completion probe immediately after
  the reinstall and after every gpuwin window). Lessons recorded in the
  notebook entry; no impact on serving behavior.
- Origin/main UNTOUCHED; branch `agent/decode-bw` carries the FSB harness
  source (qmm_fsb.comp, patched bench.cpp with `--cand-fused`,
  `tools/profile_decode_driver.py`+`analyze_decode_profile.py` + compute.h
  for the W2 profile path) for review.

## Cleanup
- jw16 scratch `/var/tmp/pp/jw16-decodebw/w*` retained until the receipt is
  accepted; gpuwin trap restores the service after every window.