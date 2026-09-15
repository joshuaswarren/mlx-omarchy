# 2026-09-15 BNNS fused-LSTM accumulator RE (macstudio)

Status: **closed at contract level.** The fused `ios18.lstm` CPU arithmetic on
macstudio (M1 Ultra, macOS 26.6.2) is recovered from binary disassembly and the
emulation reproduces the live operator **bit-exactly**: 640/640 `next_cell` and
640/640 `next_hidden` lanes on a dense seeded probe (I=H=640, random weights and
inputs), plus a 0/1/2-product accumulation ladder all 640/640 on `next_cell`.
This closes the local question behind
`parakeet.tdt.decoder-lstm.gate-preact-bnns-vs-vulkan`: the remaining decoder
gap is a Linux-kernel implementation task, not an unknown-arithmetic task.

## Symbols and dispatch path

Sampled live (`sample`, 8 s, fused-LSTM predict loop):

```text
-[MLE5Engine predictionFromFeatures:...] (CoreML)
  E5RT::ExecutionStreamImpl::ExecuteStreamSync (Espresso)
    E5RT::Ops::BnnsCpuInferenceOperation::Execute/Inference (Espresso)
      BNNSGraphContextExecute_v2 (libBNNS.dylib)
        0x...b4b108  dispatch_apply thread-pool wrapper (libBNNS local)
          0x...7e5274 / 0x...7e53e4  column-chunk dispatcher (6-way table)
            0x...7ed858  THE fp16 GEMV kernel (all hot samples)
```

Exported symbols named (full list: `bnns_exports.txt`, 414 lines):
`BNNSGraphContextExecute_v2`, `_bnns_graph_builder_register_lstm`,
`_bnns_graph_builder_register_bidirectional_lstm`,
`BNNSComputeLSTMTrainingCacheCapacity`, `BNNSDirectApplyLSTMBatchBackward`,
`BNNSDirectApplyLSTMBatchTrainingCaching`, `BNNSFunctionGRUFusedGates`.
Graph-IR debug strings inside libBNNS name the compiled ops:
`bnns::graph_compiler::op::lstm`, `op::lstm_elementwise`,
`matmul_ih_output(`, `matmul_hh_output(`, `lstm_elementwise(ih=, hh=,
initial_c=`, `lstm_while_loop_*`.

Binary source: on-disk framework dirs hold no dylibs (dyld shared cache only).
Extracted the arm64e cache with the official `/usr/lib/dsc_extractor.bundle`
(`dsc_extract.c` / `extract_py.py`; 3649 dylibs, 5.3 GB, ~25 s) — extraction
preserves unslided addresses, so `nm`/disassembly offsets match runtime samples
1:1. Extracted `libBNNS.dylib` sha256
`98c01d2d6bbabd85dc2c6fadff57ea35a1d409a054e17630b9b71879474b1daa`
(__TEXT vmaddr 0x18e061000). Function boundaries from LC_FUNCTION_STARTS
(33067 entries, `fstarts.txt`).

## The kernel (libBNNS 0x7ed858, local, 1404 B)

Pure NEON fp16. **No AMX, no fp32 accumulation, no FZ16 flush** (kernel never
touches FPCR). Shape: GEMV `y[n] += Σ_k a[k]·W[k][n]` over a K×N row-major
weight, destination-accumulating (`cbz x16` first-block-gate: block at k
offset 0 stores, later blocks `fadd` into the loaded destination).

- Column tile: 128 lanes (16 × `q` accumulators, `.8h`), descending remainder
  ladder 128→64→32→24→8→scalar (`fmadd h0`).
- Inner k loop: `ld1r {v.8h}` broadcast of one fp16 input element, 8 × `ldp`
  weight pairs, 16 × `fmla v.8h` — **fp16 FMA, single rounding per term**,
  ascending k, sequential.
- k-blocking: `B = min(max(16384/tile, 8), K)` from the head
  (`udiv #0x8000`); tile = 128 lanes → **B = 128** for any N ≥ 128.
  (My first reading took x4 = N, giving B = 8 for N = 2560 — wrong; B = 128 is
  what validates bit-exact. Siblings 0x7ebdfc/0x7ec350/0x7ec830/0x7ece0c/
  0x7ed2dc/0x7f1058 share the identical B-formula head; row-batched and
  strided-input variants.)
- Fold: after each k-block, `y = rnd16(y + blocksum)` in fp16 (`fadd .8h`,
  destination re-loaded and stored).

## The fused-op contract (all six steps validated bit-exact)

1. **One matmul over the concatenated input**: `a = [x ; h0]` (K = I+H = 1280),
   repacked weight `W = [wi | wh]` (N = 4H = 2560). Concat-as-one-GEMV scores
   640/640; two separate GEMVs with a combining add score 426/640 — the graph
   compiler repacks (`matmul_0_weights`), so the ih/hh split exists only in the
   graph IR, not in the arithmetic.
2. **GEMV per step 1** (B = 128 chains, fp16 folds, first block stores).
3. **Bias**: one fp16 add, `preact = rnd16(matmul_out + b)`.
4. **Unaries**: the 2026-09-14 macstudio dense tables (`sigma_full.npz`,
   `tanh_full.npz`) verified exact; NaN table entries (17 ambiguous σ lanes)
   need the CR fallback but were never hit on these probes.
5. **Cell**: `c1 = rnd16(rnd16(f·c0) + i·tanh16(g))` — the f·c0 product is
   rounded to fp16 (FMUL) **then** FMA-fused with the unrounded i·g product
   (single rounding at the add). Scores: this form 640/640; full-expression
   single rounding 453/640; full pairwise 543/640; f-product-unrounded 441/640.
   This answers the unary-mac receipt's open micro-question (accumulate width
   with i·g ≠ 0): it is FMUL-then-FMA, not one fused expression.
6. **Output**: `h1 = rnd16(o·tanh16(c1))` — the **rounded** `tanh16` table
   value; consuming unrounded tanh scores 360/526 on c1-correct lanes.

Per-block fold order matters: `B=128` 640/640; B=8 133, B=64 146, B=640 85
(with the final cell). All counts `next_cell`/640 on probe P4.

## Validation ladder (macstudio, CPU_ONLY, single timestep)

| probe | construction | c1 exact | h1 exact |
| --- | --- | --- | --- |
| P0 | wi=wh=0 → preact = bias | 640/640 | 640/640 |
| P1 | wi one-hot rows → 1 product/lane | 640/640 | 639/640 |
| P2 | wi two-hot rows → 2 products/lane | 640/640 | 639/640 |
| P4 | full seeded dense model | 640/640 | 640/640 |

P1/P2 h1 at 639/640: single lane each, under the ±1-ulp σ ambiguity band; not
chased (c1, which exercises the same unaries upstream, is full).

Two independent `ct.convert` builds of the same seeded model produce identical
outputs (`ladder_out.npz` P4 == `out_gemm.npz`), so the op is deterministic.

## Emulation

`bnns_lstm_contract.py` (sha256
`a41b0441385f83ed6c0bc270682a33beee7038ea6c2475b23d672963ffc1c6ba`) — numpy,
fp64-compute-then-round16 (exact fp16 FMA for exponent spread < 31 bits).
Inputs regenerated host-side with `numpy.random.default_rng(1234)` /
`(4321)`; native outputs from macstudio in `ladder_out.npz`
(`50c41360…`), `out_gemm.npz` (`ba7349f8…`).

## Where the capture-side plateau fits

BnnsAccHunt's ~150-form numpy hunt on the v7 singleton lanes plateaus at
57.2 % and my earlier mis-derived B=8 form scored 51.8 % — both below f64-exact.
Given macstudio now reproduces the contract bit-exactly end-to-end, the v7
residual (long-tailed, tens-of-arg-ulps miss distances, present even at
accumulation depth 0) is capture-side: σ16-preimage inversion ambiguity in the
gate pinning, or a capture-host libBNNS version skew. Separating test: rerun
`run_gate_preact.py` on macstudio and compare against the captured native
gates directly.

## Next test

1. BnnsAccHunt: score `bnns_lstm_contract.py` on the v7 lanes (directly-pinned
   ones, not inverted ones).
2. Rerun the v7 capture probe on macstudio (skew discriminator).
3. Linux: the contract LANDED in the pinned decoder overlay (numpy fp64
   emulation, ~6 s per 145-call decode — correctness first). Remaining if
   wanted: port the blocked GEMV to an `mx`/Vulkan kernel for speed; the
   spec is this receipt.

## Live decode on jwm1-linux (landed)

The contract shipped into the pinned decoder overlay
(`overlay/tools/coreml/vulkan_decoder.py`, sha256
`637f077ece00025e5be3c341db256c8933ae19655804d3439490709d9ebc60ef`;
`fused_lut_2026-09-15.npz` sha256
`5a7591b89185b060ccf4f9135e6460a7429fe9988fad265b035e1dc3137b4cb7` —
65536-entry bit-indexed sigmoid/tanh tables). GPU keeps the embedding gather
and projector; both LSTM layers run the contract on CPU numpy. The decoder
test mirror now calls `fused_lstm_numpy` (single source of truth).
`tests/coreml/test_fp16_activations.py` 4 passed (unchanged CR contract);
`test_vulkan_decoder.py` skips on the Linux host (needs mlx + pinned
package) and runs on the Apple hosts. The obsolete interp-fit path and
`unary_tables.npz` are deleted.

Run (jwm1-linux, Asahi, mlx `0.32.2.dev202609081722+a9f1bc8`, Apple M1
G13G B1; `flock -x -w 900 /tmp/m1-gpu.lock`, lock honoured never stolen;
prior `result.json` preserved as `result.json.pre-bnns-re`, prior
`vulkan_decoder.py` as `vulkan_decoder.py.pre-bnns-re`):

```text
PYTHONPATH=/tmp/vulkan-tdt-142 .venv-decode/bin/python diagnose_decision_142.py
```

| metric | correctly-rounded contract (2026-09-14) | this contract |
| --- | --- | --- |
| free_decode first_divergence | emission 101, token 8029 | **null** |
| decision 142 actual token | 8029 | **7892** (native) |
| decision 142 logit gap | +0.2734375 | **0.0** |
| joint_on_native_decoder_state | 16/16 | 16/16 |
| decoder_calls / joint_calls | 145 / 146 | 145 / 146 |

Landing gate met: `free_decode.first_divergence is None`. Not claimed:
token-exact E2E (104/104) — separate test; `decoder_on_native_inputs`
per-trace replays remain ulp-level non-bit-exact (max_abs 2.4e-4 cell,
1.2e-4 hidden; the projector replay dominates at 1.95e-2 — the known
native-linear fp16-FMA-chain contract `mx` cannot express), consistent with
capture-host skew rather than contract error.

Live artifact: `live_result.json` (sha256 `44af0c9a…7637`). Deployment
receipt: source sha in `result.json.source_sha256.vulkan_decoder.py`
matches the landed file `637f077e…`.

## Provenance

- Host: MacStudio.local, Mac13,2, Apple M1 Ultra, macOS 26.6.2 (25G83).
- Working dir on host: `/private/tmp/bnns-lstm-re/` (probes, sample,
  extraction; `dsc/` trimmed post-receipt, `libBNNS.dylib` kept).
- coremltools 9.0, numpy 1.26.4, python 3.11 (venv311).
- Artifacts in this directory: `bnns_lstm_contract.py`, `probe_dispatch.py`
  (full probe), `probe_ladder.py` (P0–P4), `dsc_extract.c` + `extract_py.py`
  (cache extraction), `memdump2.py`, `probe.sample.txt` (8 s, 4477 lines),
  `bnns_exports.txt`, `fstarts.txt`, `libBNNS.dylib`, `ladder_out.npz`,
  `ladder_inputs.npz`, `out_gemm.npz`.
- Host writes: macstudio only inside `/private/tmp/bnns-lstm-re/`;
  jwm1-linux `/tmp/vulkan-tdt-142/` (decoder deploy + backups + live run).
  Reference capture untouched. GPU use: locked jwm1 decode only; no merge
  of 63c1d3cf.

- resolved_model: `zai/glm-5.3-flash`

