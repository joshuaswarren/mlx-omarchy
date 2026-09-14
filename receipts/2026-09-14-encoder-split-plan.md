# Encoder ANE island + Vulkan split (2026-09-14)

Host-only. No ANE, no SSH, no jwm1, no `ane-linux-experiments` edit or
execute. No full-encoder ANEC is claimed. No hardware execution is claimed.

Executing model: `anthropic/claude-opus-5` (session-reported identity). The
assignment named an OpenAI route; the session reports an Anthropic model. The
routing discrepancy is reported to the parent and is not hidden behind the
usual receipt line.

This closes the design half of blocker 2 (H13 encoder compile holes): the
in-envelope ANE work is cut into **three compiled bundles** that pass the
schema-4 loader gate, everything else stays on Vulkan, and every tensor that
crosses the boundary is named with its dtype, dense byte size, channel, and
surface geometry.

## Pins

| item | value |
| --- | --- |
| overlay | mlx-omarchy `origin/main` `030bab6ff6c279a5d76770b9dfc4ab03aa7975d8` |
| compiler source | mil-hwx-compiler `83d486b178e83e602ab894f1154004bb7f2600b9` (`receipts: (32,512,128) is 32 programs; (32,1024,128) chain-overlaps`) |
| `build/mil-hwxc` sha256 | `77521fb36681f8ace6e4aff97ef7ca856b57137e3808680f85dfd4742248de61` (`make -q build/mil-hwxc` → 0, binary current for that source) |
| checker | `mlx-omarchy-info` sha256 `67fa0ff39f00d3b3678b45a528e134fd49ccb1cfa96ee74a5e3b63ec5aa1c02a` (built 2026-09-13 in worktree `main-land2`) |
| encoder MIL | `receipts/2026-09-14-encoder-leftover/model-fp16-allpeels.mil` sha256 `a7f08439c281b8e1e1dfe7dbabbef8bf844a7094c1fe98a00688340fcb8db4dd` |
| encoder | `mweinbach1/parakeet-tdt-0.6b-v3-coreml` revision `b650695c2322ee5281dff48d7345b2f3a58ff018` |
| leftover baseline | `receipts/2026-09-14-encoder-leftover.md` |
| host | `omp-studio-local` Linux 6.17.2-1-pve x86_64 |

The previous pin for this island family (`82f1d4ce`, 2026-09-13) predates
`37fb29e` *h13: swap select operands to Apple cond-true channel*. Every select
program here is recompiled at `83d486b`; the select ANEC is
`d40ec023cfd5b2c5d5cfeb95f4a4673024f495224acd4448e8498bf50114780f`, not the
`860de06c…` of the 2026-09-13 island. The older select program carries the
pre-fix operand mapping. The batched-matmul program is unchanged across the two
pins (`a3aa2fe1333a48eba3e7fcae5b848563e2bfa747a232dadf9ed2d568821c5627`).

## The cut, from the MIL

One conformer layer, source lines from `model-fp16-allpeels.mil` (layer 0;
layers 1–23 repeat the same ops and shapes under other tensor-name
suffixes). `ANE` marks an in-envelope op, `VK` a leftover op that stays on
Vulkan.

```
hidden_states_21_cast_fp16 [1,375,1024]
  VK   8x linear(128,1024)+bias -> _h0_lin.._h7_lin, expand_dims, concat axis 1
  VK   query_states_1_cast_fp16 [1,8,375,128]
  VK   query_1_cast_fp16 = + var_345_to_fp16          (pos bias u)
  VK   query_states_with_bias_v_1 = + var_348_to_fp16 (pos bias v)
  VK   mul_0_cast_fp16 = query_1 * var_7_to_fp16      (1/sqrt(d) scale)
  VK   hidden_states_23_cast_fp16 [1,8,375,128]  (K heads, same 8x linear+concat)
  VK   transpose K -> k_headsT [1,8,128,375]          <- boundary repack
  ANE  attention_scores_1 = matmul(q_v,  pos_kT)   [1,8,375,749]   island A prog 0
  ANE  matmul_0           = matmul(q_s,  k_headsT) [1,8,375,375]   island A prog 1
  VK   attention_scores_3 = conv(padconv, g=8)     [1,8,375,750]   rel-pos shift
  VK   reshape -> [1,8,750,375], slice/concat -> var_367 [1,8,749,375]
  VK   reshape -> matrix_bd_1 [1,8,375,749]
  VK   matrix_bd_3 = slice_by_index last dim       [1,8,375,375]
  VK   matrix_bd_5 = matrix_bd_3 * var_371_to_fp16
  VK   var_373_bool broadcast to [1,8,375,375]                     <- boundary repack
  ANE  attention_mask_9 = select(ninf_rt, matrix_bd_5, cond)       island B
  VK   add_0 = matmul_0 + attention_mask_9         [1,8,375,375]
  VK   softmax_0 = softmax(add_0, axis=-1)         [1,8,375,375]
  VK   hidden_states_25_cast_fp16 [1,8,375,128]  (V heads)
  ANE  attn_output_1 = matmul(softmax_0, v_heads)  [1,8,375,128]   island C
  VK   transpose, concat, out projection, conv module, FFN, layer_norm ...
```

`attention_scores_1` feeds the Vulkan rel-pos shift, whose result
`matrix_bd_5` feeds the select. That data dependency is why the select cannot
join island A: one package binds all of its inputs before the first dispatch.
`matmul_0` and the select are mutually independent but land on opposite sides
of that same Vulkan chain, so island A takes the two matmuls that only need
layer-local Q/K, and the select is its own package.

## Island set (compiled, host)

`build/mil-hwxc --target H13 --format anec`, default `--schedule per-op`.
Compiler packages: `receipts/2026-09-14-encoder-split-plan/packages/<island>/`.
Every MIL compiled for this receipt is kept under
`receipts/2026-09-14-encoder-split-plan/islands/`:

| MIL | kept? | what it measures |
| --- | --- | --- |
| `island-attn-a-kt.mil` | **island A** | rel-pos + scores matmul, both `transpose_y = false` |
| `island-select-8head.mil` | **island B** | select with an 8-head bool cond |
| `island-pv.mil` | **island C** | PV matmul |
| `island-attn-a-rt.mil` | no | same pair with `transpose_y = true` scores (209 TDs, shape/`nchw` mismatch) |
| `island-select.mil` | no | select with the `[1,1,375,375]` broadcast cond (adapter refusal) |
| `island-qk.mil`, `island-relpos-rt.mil` | no | the two island-A programs standalone |
| `island-qk-select.mil` | no | scores + select in one package (2 programs, `dispatchPlan [0,1]`) |
| `island-qk-select-add.mil` | no | the same plus `add_0`: 17581 programs |
| `island-relpos-const-real.mil` | no | rel-pos with the real layer-0 table baked as a constant |

| island | MIL | rc | programs | encoder | TDs | anec bytes | sha256 |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| **A** `island-attn-a-kt` prog 0 | rel-pos matmul `[1,8,375,128]x[1,8,128,749]` | 0 | 1 | `apple-parity-batched-matmul` | 208 | 182400 | `d05e193a620d7edc25ee61f865c164182155f84e7fb925a38c303a9c0db1d8df` |
| **A** prog 1 | scores matmul `[1,8,375,128]x[1,8,128,375]` | 0 | 1 | `apple-parity-batched-matmul` | 208 | 182400 | `a3aa2fe1333a48eba3e7fcae5b848563e2bfa747a232dadf9ed2d568821c5627` |
| **B** `island-select-8head` | select, runtime a, bool cond `[1,8,375,375]` | 0 | 1 | `apple-parity-boolean` | 5 | 9216 | `d40ec023cfd5b2c5d5cfeb95f4a4673024f495224acd4448e8498bf50114780f` |
| **C** `island-pv` | PV matmul `[1,8,375,375]x[1,8,375,128]` | 0 | 1 | `apple-parity-batched-matmul` | 208 | 182400 | `76496b74eff176f8e0358dbb5e285f7fe60c8a7c270c2b5f400b5407195c3fca` |

Island A is one package, `dispatchPlan [0, 1]`, two ordered logical results
(`attention_scores_1`, `matmul_0`), no intermediate. 416 task descriptors.

Per layer: 3 packages, 4 programs, 629 task descriptors. All three packages are
layer-invariant — every shape is identical across the 24 layers and the
per-layer rel-pos table is bound as a runtime input — so the encoder needs
**3 bundles on disk**, 556416 anec bytes in total. One encoder pass submits
them 72 times: 96 program dispatches, 15096 task descriptors.

Layer-invariance is counted, not assumed: the MIL holds 24
`attention_scores_*` matmuls at `[1,8,375,749]`, 24 `matmul_*` at
`[1,8,375,375]`, 24 `attn_output_*` at `[1,8,375,128]`, and 24 selects, all
with the same `a = var_8_to_fp16_rt` and `cond = var_373_bool` function
inputs. Only the 24 `[1,8,128,749]` rel-pos constants differ, and this
contract binds that table as a runtime input.

### Schema-4 bundles

`overlay/tools/ane-export/h13_package_to_bundle.py` → `receipts/2026-09-14-encoder-split-plan/bundles/<island>/`,
then the loader gate. Full output: `receipts/2026-09-14-encoder-split-plan/check-bundle.log`.

| bundle | graph_hash | task_descriptors | payloads | check-bundle |
| --- | --- | ---: | ---: | --- |
| `parakeet-encoder-island-attn-a-kt` | `85325f3059a345dee7c62c53a0072b2e869c0c8116385c992e8963587714ff88` | 416 | 2 | `OK: bundle valid` EXIT:0 |
| `parakeet-encoder-island-select-8head` | `fd9e6c5b64f8b2c58fa2067799589ba0d32dfc24361b6612c9ef94d9e1a6422e` | 5 | 1 | `OK: bundle valid` EXIT:0 |
| `parakeet-encoder-island-pv` | `ed03818974f7e2728daff0c839fcd4486a87282515347e68db7d2219605905c3` | 208 | 1 | `OK: bundle valid` EXIT:0 |

This is structural schema-4 acceptance and declared H13/driver-ABI-1
compatibility. It is not device or numerical qualification.

## Boundary contract

Direction is from the ANE island's point of view. `dense bytes` is what the
caller stages: a plain row-major buffer of exactly that many bytes, in the
`nchw` axis order. The runtime, not the caller, performs tile placement —
`pack()`/`unpack()` in `overlay/mlx/backend/omarchy/ane/worker_libane.cpp:159`
walk dense elements in linear order and place element *i* at
`ane_packed_offset` (`overlay/mlx/backend/omarchy/ane/tile_layout.h:28`),
`plane*plane_stride + row*row_stride + column*element_size`.

### Island A — `parakeet-encoder-island-attn-a-kt`

| dir | tensor | MIL producer / consumer | dtype | shape = nchw axes | dense bytes | surface (plane,row) | ch |
| --- | --- | --- | --- | --- | ---: | --- | ---: |
| in | `q_v` | `query_states_with_bias_v_1_cast_fp16` | fp16 | `[1,8,375,128]` | 768000 | 96000, 256 | 6 |
| in | `pos_kT` | `var_355_to_fp16` (per-layer const) | fp16 | `[1,8,128,749]` | 1533952 | 196608, 1536 | 5 |
| in | `q_scaled` | `mul_0_cast_fp16` | fp16 | `[1,8,375,128]` | 768000 | 96000, 256 | 6 |
| in | `k_headsT` | transpose of `hidden_states_23_cast_fp16` | fp16 | `[1,8,128,375]` | 768000 | 98304, 768 | 5 |
| out | `attention_scores_1` | → `attention_scores_3` padconv | fp16 | `[1,8,375,749]` | 4494000 | 576000, 1536 | 4 |
| out | `matmul_0` | → `add_0` | fp16 | `[1,8,375,375]` | 2250000 | 288000, 768 | 4 |

### Island B — `parakeet-encoder-island-select-8head`

| dir | tensor | MIL producer / consumer | dtype | shape = nchw axes | dense bytes | surface (plane,row) | ch |
| --- | --- | --- | --- | --- | ---: | --- | ---: |
| in | `ninf_rt` | `var_8_to_fp16_rt` (function input, `0xFC00` fill) | fp16 | `[1,8,375,375]` | 2250000 | 288000, 768 | 5 |
| in | `matrix_bd_5` | `matrix_bd_5_cast_fp16` | fp16 | `[1,8,375,375]` | 2250000 | 288000, 768 | 6 |
| in | `cond` | `var_373_bool` broadcast to 8 heads | bool | `[1,8,375,375]` | 1125000 | 144000, 384 | 7 |
| out | `attention_mask_9` | → `add_0` | fp16 | `[1,8,375,375]` | 2250000 | 288000, 768 | 4 |

Program scratch is 2310144 bytes (channel 3). `scratch_bytes` is not a tensor.

### Island C — `parakeet-encoder-island-pv`

| dir | tensor | MIL producer / consumer | dtype | shape = nchw axes | dense bytes | surface (plane,row) | ch |
| --- | --- | --- | --- | --- | ---: | --- | ---: |
| in | `probs` | `softmax_0_cast_fp16` | fp16 | `[1,8,375,375]` | 2250000 | 288000, 768 | 6 |
| in | `v_heads` | `hidden_states_25_cast_fp16` | fp16 | `[1,8,375,128]` | 768000 | 96000, 256 | 5 |
| out | `attn_output_1` | → out-projection transpose | fp16 | `[1,8,375,128]` | 768000 | 96000, 256 | 4 |

### Contract rules

1. **Row pitch is padded to 64 elements.** `375 → 384`, `749 → 768`,
   `128 → 128`, for both fp16 (768 B, 1536 B, 256 B rows) and bool (384 B
   rows). Plane stride is `H × row_stride`. The caller never builds those
   surfaces; it stages dense bytes and the runtime packs.
2. **The staged axis order is the manifest `nchw` order, not the MIL shape.**
   Every binding above is spelled so the two agree. They do not have to: the
   `transpose_y = true` spelling of the scores matmul declares
   `[1,8,375,128]` and binds `nchw [1,8,128,375]`, so a caller that trusts the
   declared shape stages K untransposed and the pack walks the wrong axes.
   `island-attn-a-rt.mil` measures that spelling (rc 0, 209 TDs, program sha
   `6cd906a2faf3762b225e4709e7cfb0e8a9960792e55fafff9225346fa078907d`). It is
   rejected for this contract in favour of the `transpose_y = false` spelling,
   which is self-consistent and one TD smaller.
3. **Broadcasts are materialized by Vulkan.** `var_373_bool` is
   `[1,1,375,375]` in the graph and the program binds 1125000 bool elements.
   The `[1,1,375,375]` spelling compiles to the identical ANEC
   (`d40ec023…`) but its manifest declares `tensors.cond.logicalBytes` 140625
   against a binding of 1125000, and the adapter refuses it:
   `tensor 'cond' binding range exceeds its logical shape`. The 8-head
   spelling is the contract.
4. **`ninf_rt` is a runtime input, not a constant.** H13 refuses every
   constant-`a` select (`h13.select-needs-decoded-encoder`); the frontend
   materializes the `-inf` fill. It is also the only way the select stays a
   5-task program.
5. **`ninf_rt`, `cond` and every island program are layer-invariant.**
   `pos_kT` and `matrix_bd_5` are per-layer. There is no resident read-only
   input in schema 4: `state` and `intermediates` must be bound for both read
   and write (`docs/ane-bundles.md`), so the two invariant buffers are staged
   per dispatch under this contract.
6. **Failure is fail-closed and named.** A short buffer is refused before
   submit: `worker.cpp:114` compares the staged size against the program
   binding's `logical_bytes`. A missing bundle directory is not an error — the
   region stays on Vulkan (`ane/bundle.h`, `AneBundleNotFound`).

### Boundary traffic, measured from the manifests

| leg | to ANE | from ANE |
| --- | ---: | ---: |
| island A | 3837952 | 6744000 |
| island B | 5625000 | 2250000 |
| island C | 3018000 | 768000 |
| per layer | | 22242952 (21.2 MiB) |
| 24 layers | | 533830848 (509.1 MiB) |

Of that, 112.4 MiB per pass is re-staging the layer-invariant `ninf_rt` and
`cond` plus the per-layer `pos_kT`. Whether the split is faster than Vulkan
alone is **not established** here and cannot be from a host compile; it needs
the alternating ANE-on/ANE-off measurement the compatibility plan specifies.

## The Vulkan side: no CPU tensor fallback

Every leftover op has a GPU path in this overlay. Status and test anchors from
`docs/compatibility-matrix.md` at `030bab6f`; `shared-gpu` means MLX's common
GPU layer driving omarchy kernels.

| leftover op | n | MLX primitive | omarchy status | evidence |
| --- | ---: | --- | --- | --- |
| `concat` | 96 | `Concatenate` | shared-gpu | matrix row `Concatenate`, `test_complex_ops.cpp#L418` |
| `transpose` | 73 | `Transpose` | shared-gpu | matrix row `Transpose`, `test_complex_ops.cpp#L227` |
| `slice_by_index` last dim | 24 | `Slice` | shared-gpu | matrix row `Slice`, `test_complex_ops.cpp#L269` |
| `layer_norm` | 120 | `fast::LayerNorm` | native, no guard | matrix row `fast::LayerNorm`, `test_fast_ops.cpp#L338` |
| `silu` | 72 | `Multiply` ∘ `Sigmoid` | native | matrix rows `Multiply`, `Sigmoid` |
| `sigmoid` | 24 | `Sigmoid` | native | matrix row `Sigmoid`, `test_compiled_tape.cpp#L179` |
| `softmax` | 24 | `Softmax` | native | `dispatch_softmax` (`primitives.cpp:1902`) takes any last-axis row length, fp32 accumulation; 375 is not special |
| `conv` | 101 | `Convolution` | partial (`Convolution shapes`) | `Convolution::eval_gpu` (`primitives.cpp:4023`) refuses only spatial rank outside 1–3, rank mismatch, and group/channel mismatch — every encoder form (2-D k3s2 g1, k3s2 g256 depthwise, 1×1 g1, 1×1 g8 padconv; 1-D pointwise 1024→2048 and 1024→1024, 1-D depthwise k9 g1024) is inside that guard |
| `add`, `mul` at encoder shapes | — | `Add`, `Multiply` | native | matrix rows `Add`, `Multiply` |

So the split has no op that lands on the CPU. The CPU schedules, stages, and
copies; it evaluates no tensor primitive.

## What still cannot run on the ANE

Re-probed at this pin with the current binary (`receipts/2026-09-14-encoder-split-plan/reprobe-leftover.json`,
`receipts/2026-09-14-encoder-leftover/probes/*`). Every hole from the
2026-09-14 leftover table still refuses; every in-envelope form still compiles.

| probe | rc | diagnostic |
| --- | ---: | --- |
| `concat-xn-heads` | 65 | `h13.invalid-concat-parameters` |
| `concat-values-tuple` | 65 | `h13.unsupported-concat` |
| `transpose-r3-021`, `transpose-r4-0213` | 65 | `h13.nonfoldable-transpose` |
| `slice-lastdim-375-749` | 65 | `h13.noncontiguous-slice` |
| `layer-norm-encoder` | 65 | `h13.norm-outside-envelope` |
| `softmax-8375375` | 65 | `h13.norm-outside-envelope` |
| `silu-1024-375`, `silu-375-4096`, `sigmoid-1024-375` | 65 | `h13.unary-outside-envelope` |
| `conv-1x1-750` | 65 | `h13.conv-outside-envelope` |
| `select-fp16-cond` | 65 | `h13.boolean-outside-envelope` |
| `matmul-relpos`, `matmul-scores-ty0`, `matmul-scores-ty1`, `matmul-pv` | 0 | compiles |
| `select-rrb-8375375`, `select-encoder-cond-broadcast` | 0 | compiles |

Three further limits bound the shape of any split, not just this one:

1. **No full-encoder ANEC.** `concat` is an ISA hole with no copy encoder, and
   the encoder emits 96 of them. Nothing in this receipt changes that.
2. **No fused island.** `--schedule=chain` refuses both this island family and
   the general case: `h13.chain-outside-envelope` (at most two boundary
   inputs) and `h13.chain-unrepresentable-edge` (`apple-parity-batched-matmul`
   has no decoded post-operation field). Islands are ordered program lists
   inside one package, never one fused program.
3. **Elementwise on ANE is refused by policy, not by envelope.** `add_0` at
   `[1,8,375,375]` does compile: `island-qk-select-add.mil` is rc 0 with
   **17581 programs** (1 batched matmul + 1 boolean + 17579
   `h13-source-qualified` adds) and 118008559 bytes, because the elementwise
   path 64-lane-splits the surface. It is kept on Vulkan and its package was
   not retained.

## Adapter change

`overlay/tools/ane-export/h13_package_to_bundle.py` refused every package
containing a bool surface: the compiler spells `dtype` on the graph tensor for
bool, and the adapter's allow-list had only
`{accumulation, aliasOf, logicalBytes, role, shape}`. The 2026-09-13 island
worked around it by hand-editing a copy of the compiler manifest
(`receipts/2026-09-13-h13-v2-as-schema4.md`), which does not scale to a
per-region export path. `dtype` is now allowed and checked against the program
binding it duplicates; a contradicting dtype is a named refusal.

```text
python3 -m unittest overlay.tests.omarchy.ane.test_h13_package_to_bundle
# Ran 13 tests ... OK
```

Island B converts with no manifest edit.

## Not established

- Hardware execution of any island. No device was opened.
- Numerical agreement of any island against the CoreML/host reference.
- Whether the split is faster than Vulkan alone.
- The `AneRegion` partitioner. `docs/architecture.md` and the compatibility
  plan describe it; this tree has the bundle loader, the runtime, and the
  worker, not the graph-level partition that would select these three regions
  during `eval`.
- Compilation of the remaining leftover with any rewrite. The 8 hole classes
  are re-confirmed refusals, not open questions closed here.
- The const-baked rel-pos alternative is measured but not adopted:
  `island-relpos-const-real.mil` against the real layer-0 blob
  (`weight.bin` offset 8195008) is rc 0, one `apple-parity-batched-matvec`
  program, 208 TDs, 1738880 bytes, sha
  `30985d60132d6ddc649f6c520768bc634294486a39780dc70787b156692de610`. It
  removes 1533952 staged bytes per layer and adds 24 distinct per-layer
  bundles carrying model weights.

## Hardware boundary

No SSH, hardware inference, ANE command, router change, or Mac execution.
No files under `ane-linux-experiments` were edited or executed.
