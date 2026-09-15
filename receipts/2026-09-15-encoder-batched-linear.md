# Encoder leftover linears batched: pin-exact, encoder wall 22.6 s to 19.4 s (2026-09-15)

Verdict: the 194 leftover encoder linears now run as **one batched fp32 matmul
per linear plus one custom reduce kernel**, replacing the 5688f8bd chunk loop.
`encoder_hidden` is **bit-identical to the pin**
`38c73261f29230276ed76f1fc017b76b024156d79218bd5f1347fdc7e7d43ec7`, the ANE
E2E stays **104/104 with transcript sha `db501a8c`** and the frozen contract
bounds pass, and the encoder stage wall drops **22641.2 → 19386.0 ms
(−3255 ms, −14.4%)** on the same command and machine.

## Change

`overlay/tools/coreml/vulkan_encoder.py` `linear` handler (branch tip
`agent/encoder-batched-linear`, copy in
`receipts/2026-09-15-encoder-batched-linear/vulkan_encoder.py`, sha256
`0c68c1ff58dff77b2de6a592cf970b4463d2a95c9d6642efb60382309b99aa25`):

1. K is reshaped to `[K/16, 16]` on both operands; one batched fp32 matmul
   produces every 16-wide block's partial `[K/16, M, N]`.
2. `_leftover_chain_kernel` (`mx.fast.metal_kernel`, one thread per output
   element) applies the landed rounding in one dispatch: each block's partial
   rounds fp32→fp16, then accumulates in fp16 ascending — the same strictly
   sequential fp16 sum the chunk loop performed dispatch by dispatch.

Byte identity rests on two probes plus the real run:

- Batched-GEMM per-block partials are byte-identical to the loop's per-chunk
  `xs @ wsT` partials on every probed shape (`blocks_identical=True`).
- Handler A/B driving the real `apply` code paths on jwm1 over the real MIL
  shape mix (96×1024→1024, 49×4096→1024, 48×1024→4096, 1024→640; with and
  without bias; 2 trials): **ALL-IDENTICAL**.
- Real run: `encoder_hidden.npy` sha256 = the pin. Control (landed chunk
  loop) reproduced the pin in the same session before the batched arm ran.

## Why a stock single reduce could not do it

- `mx.sum` over the blocks agrees with the ascending fp16 chain on only
  0.104–0.198 of elements across three layouts (axis 0 on `[B,M,N]`,
  axis −2 on `[M,B,N]`, axis −1 on `[M,N,B]`) — it accumulates in a different
  order/precision and rounds once.
- `mx.cumsum` has no Vulkan kernel on this backend (fp16 scan raises).

## Why the explicit chain was rejected (measured, not tasted)

Standalone jwm1 runs, islands ABC, same basis (reports in this dir):

| arm | encoder wall | vk compute dispatches | encoder_hidden |
| --- | ---: | ---: | --- |
| chunk loop (5688f8bd control) | 22605 ms | 158100 | `38c73261…` ✓ |
| batched GEMM + explicit fp16 chain | 26947 ms | 50144 | `38c73261…` ✓ |
| batched GEMM + chain, groups of 8 | 27159 ms | 60280 | `38c73261…` ✓ |
| batched GEMM + `mx.sum` (diagnostic, bytes wrong) | 20095 ms | 6884 | `fa1cc6ef…` |
| **batched GEMM + kernel reduce (landed)** | **19484 ms** | **6690** | **`38c73261…` ✓** |

The explicit chain serialized ~21.6k dependent fp16 adds into the graph and
cost more wall than the chunk loop it replaced (26.9 s vs 22.6 s); grouping
the chain did not help. The `mx.sum` diagnostic isolated the batched matmul
itself as fast in-graph (20.1 s with the wrong bytes), so the reduce moved
into one kernel.

## E2E result (jwm1-linux, real pipeline)

Same command as `receipts/2026-09-15-encoder-fp16-leftover.md` with
`--encoder-runner /var/tmp/ParakeetE2EBatchedLinear-stage-vulkan_encoder.py`
and fresh out/scratch dirs. Lock `/tmp/m1-gpu.lock`, `flock -w 900`, never
steal, never unlink.

| quantity | value |
| --- | --- |
| status | **match** |
| emissions actual / native | 104 / 104 |
| matching_prefix_length | 104 (first_divergence null) |
| e101 | **7892** |
| durations_match / frame_indices_match | true / true |
| transcript_match | true, both sha256 `db501a8c080380ea027ffa50a4b4956c39df77cb692c4fb78e556311a11a0790` |
| encoder_max_abs_err | 0.175354 ≤ 0.3 PASS |
| encoder_mean_abs_err | 0.003755 ≤ 0.02 PASS |
| encoder_rel_l2_err | 0.023044 ≤ 0.1 PASS |
| nan / inf | 0 / 0 PASS |
| encoder_hidden sha256 | **`38c73261f29230276ed76f1fc017b76b024156d79218bd5f1347fdc7e7d43ec7`** (pin, unchanged) |
| mel sha256 | `5b54f4a9a2ba3434cd69b6e48e6780d3bcb6c635d9ce85cda3d85c60f2455bde` (unchanged) |
| ANE | islands ABC, 72 submits, 72 worker starts, 0 timeouts, exec 5057 ms |
| vk compute dispatches (standalone basis) | 158100 → 6690 (−95.8%) |
| stage walls ms | audio 183.3, mel 256.6 (warm; the pin run paid a cold 9744.8), **encoder 19386.0** (was 22641.2), decode 2880.8, pipeline 22843.0 |

Identical encoder bytes ⇒ identical decode: the measured encoder error stats
are bit-for-bit the pin run's figures.

## Identity

- encoder runner (this change) sha256 `0c68c1ff58dff77b2de6a592cf970b4463d2a95c9d6642efb60382309b99aa25`, staged on jwm1 at `/var/tmp/ParakeetE2EBatchedLinear-stage-vulkan_encoder.py`
- control runner = landed 5688f8bd file, sha256 `974fc2a966e38719449a791088449ab09bcc3442ca7effdf8819903bd76fec19`
- e2e runner `/var/tmp/ParakeetE2E/parakeet_e2e.py` sha256 `1623707880831c963514eb07b533a3e707bfda6ce05208932a0a4522b10f12a7`
- decoder `vulkan_decoder.py` `637f077ece00025e5be3c341db256c8933ae19655804d3439490709d9ebc60ef`, LUT `fused_lut_2026-09-15.npz` `5a7591b89185b060ccf4f9135e6460a7429fe9988fad265b035e1dc3137b4cb7` (pkg `/var/tmp/ParakeetE2EAneBnns/pkg`)
- MIL `ac8e9526154ac8b8d4c08b83e44dbc1a2208d7ec33e42a7838d57b698b14f133`, audio `30885601173f96b0d8ddd020dc959b055c6c1582b85a33e3fcab8c4b08ed94c2`
- worker `762dd1de…`, libane `1ab9d95d…`, bundles `island-attn-a-kt` / `island-pv` / `island-select-8head-scratch417`
- host jwm1-linux, aarch64, Python 3.14.7 (venv-agxgen + MelFrontendPerf venv-cache overlay)

## Artifacts

`receipts/2026-09-15-encoder-batched-linear/`:
`e2e-report.json` (`ab48804c…`),
`run-report-control-loop.json` (`fae410f4…`),
`run-report-batched-kernel.json` (`56e74811…`),
`vulkan_encoder.py` (`0c68c1ff…`).

## Not claimed

- No perf claim beyond the encoder stage: mel was warm in this run and cold in
  the pin run, so pipeline totals are not comparable (22843 vs 35624 ms).
- The fp32-GEMM pre-correctness form (18.2 s, wrong bytes) is still faster
  than any byte-exact form measured; ~1.3 s of this change's remaining cost
  over that baseline is the price of the per-block fp16 rounding contract.
- Generalization beyond this pinned fixture is untested.

resolved_model: this session (`zai/glm-5.3-flash`).
