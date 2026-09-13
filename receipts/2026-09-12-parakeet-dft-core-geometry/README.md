# Receipt: vDSP DFT core geometry, 2026-09-12

This experiment executes the next step recorded in `receipts/2026-09-12-parakeet-mel-stage-isolation.md`: probe every packed complex input position, compare radix-4 and split-radix networks by exact float32 and signed-ULP signatures, then re-run the preserved 3001-frame stage comparison. The implementation source is mlx-omarchy commit `18248be2` (`mel: isolate vDSP DFT core geometry`). The capture tool remains pinned to `parakeet-coreml-swift` commit `75aec2a1c991319657ff4dec5f602c12da6c5012`; the reference model remains `mweinbach1/parakeet-tdt-0.6b-v3-coreml` revision `b650695c2322ee5281dff48d7345b2f3a58ff018`.

## Inputs and execution

`dft_geometry_probe.py make-input` generated a 512 by 512 float32 identity basis. Rows `2*c` and `2*c+1` therefore isolate the real and imaginary components of packed complex index `c`, for every `c` from 0 through 255. The raw input SHA-256 is `fc6ed15324fa8fe5d0b68455503165a6b03a0df03bf86de07bd40b703281be58`.

The committed Swift source SHA-256 was `b653c1ff6fc71dc50202f2670a3ce18b9104b29c9e5b56422232ef40e6d0ab0e` locally and at `/tmp/parakeet-mel-stage/Sources/mel-stage-capture/main.swift`. A release build completed in 5.19 seconds. On Apple M1 Ultra, macOS, `probe-dft` and the new direct `probe-dft-core` mode each captured 512 frames. This was CPU-only Accelerate execution. It did not load a CoreML model or use ANE or GPU execution.

The real-DFT manifest SHA-256 is `4c24b59e0eff5c9a6a65ef372f789005b1fdda6fbf83070c0132c40359a769a1`; the direct complex-core manifest SHA-256 is `850d6f48b863eebac1dc0c758a9c2cfa07277e3338217bb901307c981344ba6a`. The preserved general-probe manifest is `94e84e95d5fc44ce49b358a6a413b7d5e1d1c8bdf471ab81a927c08ae052d2fd`. The certified 3001-frame stage manifest is `67ecd46384c33040869093cde9b0615e0a0e7aaaa8450212c46a366405c61a14`. `comparison.json` contains the complete signed-ULP histograms and validates every manifest before scoring.

## Geometry result

The direct 256-point complex-core comparison contains 262,144 real-or-imaginary float32 values:

| Candidate | Numeric mismatches | Exact impulse rows | Maximum absolute difference | Row-signature SHA-256 |
|---|---:|---:|---:|---|
| recovered vDSP radix-4 DIF | 0 | 512 | 0 | `5f70bf18a086007016e948b04aed3b82103a36bea41755b6cddfaf10ace3c6ef` |
| radix-4 DIT | 135,136 | 16 | 2.980232e-7 | `69302653144fcd83610a0d42a7c6238065512743f36e62be86faef380d688dc2` |
| split-radix DIT | 135,408 | 8 | 2.980232e-7 | `84438788ed5b8d98b7392e1a24ba421956b03adcc51193e15d75c27eac522792` |
| radix-4 DIF | 144,544 | 16 | 2.980232e-7 | `e3d338427a0b4eb3c197e34a87597e8434b6df0fc00613bf89b9cb8a16fc7265` |

The recovered real postprocess is also exact on all 263,168 real-DFT impulse values. With the corrected postprocess applied to every candidate, radix-4 DIT has 138,924 numeric mismatches, split-radix DIT has 139,588, and ordinary radix-4 DIF has 143,106. Their row-signature hashes are respectively `dcd90ea93942da529fd5e6b5772b0b47a80ccd05ff6653a888b5433a2aa2b4f1`, `15c65698eeab362c4457015272f3f4310fd8247292a6f4d02c6bf57ebfa2f840`, and `9fae5e7cafb6f5343d0bdd10b7e949c1fe05fde36205a72fd6ed82062d0ca067`.

Among the original portable networks, radix-4 DIT was the closest candidate but did not identify the actual vDSP kernel. Split radix lost half of the exact direct-core impulse rows, while ordinary radix-4 DIF added 9,408 direct-core mismatches. A recurrence-built twiddle table was worse at 207,312 direct-core mismatches. Tested mixed-radix stage orders (4,8,8), (8,4,8), and (8,8,4) produced 145,552, 147,568, and 148,016 direct-core mismatches. The rejected candidates were removed rather than kept as dead code.

Substituting branch-specific roots recovered from vDSP output reduced the closest candidate from 135,136 to 90,192 direct-core mismatches and increased exact rows from 16 to 80. Standalone 4-, 16-, 64-, and 256-point vDSP root captures produced 90,704 mismatches. Split real/imaginary multiplication and both tested one-product-FMA orders did not improve that result. Coefficient values mattered, but captured roots alone did not recover the kernel.

## Public Accelerate observations

A separate CPU-only program compared vDSP_DFT_zop with the public legacy vDSP_fft_zip and vDSP_fft_zop APIs. Both legacy APIs are bit-identical to the DFT API on 48 sparse stage probes. On the 512-frame impulse basis they differ only at 512 axis-zero values, with maximum absolute difference 6.123234e-17. On the 13-frame general set, all 75 differences occur in the dense random frame. On the preserved 3001 dense frames, the DFT and legacy APIs differ at 87,858 of 1,536,512 complex-core scalar values, with maximum absolute difference 2.3841858e-7.

Creating legacy FFT setup tables at log2(n) from 8 through 12 and executing the same 256-point transform produces the same mismatch counts and maximum differences for every setup size. This eliminates setup-table resolution and stride as the source of the DFT-versus-legacy gap. The direct root row also fails to match every tested public libm construction: the best double-libm candidate differs at 136 of 256 complex roots, while cosf candidates differ at 225 or more. Full output is in vdsp-public-api.txt.

Two-input superposition probes use captured single-input outputs as their own coefficient baseline, so they measure arithmetic grouping instead of root error. For the same 1/3 and 2/3 values, the residual count changes with both input separation and base index; for example, separation 32 yields 0 residual values at base 0 but 236 at base 1 and 316 at base 15. The input SHA-256 is 6b0f16d9641f7e4e3c6e53aba562adead03e13f0228da27fd4682cea1678a7f3; the capture manifest SHA-256 is 9315a3525af1c6f20123021055a34456150d8d7d0d8166a586b55504161e7f8a. pair-residual-map.txt records all 48 alignments.

Targeted cancellation and small-length probes sharpen that result. The public length-4 codelet matches the reduction \((x_0+x_2)+(x_1+x_3)\) on all 16,384 tested inputs for both DC and Nyquist, while four alternate associations do not. The same radix-4 DC tree is exact through length 128 and differs on only one of 2,048 length-256 inputs. Same-architecture Swift implementations of ordinary radix-4 DIT and DIF still diverge from the public kernel at length 16, so the discrepancy begins with the first twiddled stage rather than the base butterfly.

Public runtime inspection resolved the missing mechanism. The length-16, 64, and 256 setups use the same generic entry. It applies a grouped radix-4 DIF first pass, then FMA-fused radix-4 passes of lengths 16, 64, and 256. Its coefficient table stores cosine/tangent factors so each rotation and the following butterfly share rounding: \(\cos \theta, \tan \theta, \cos 2\theta, \tan 2\theta, \cos 3\theta / \cos \theta, \tan 3\theta)\). Double-precision libm followed by a float32 cast reproduces every logical core coefficient; length 256 only permutes their storage. The recovered real postprocess uses the observed float32 grouping and five Darwin cosine-rounding corrections.

The portable translation is bit-identical to vDSP on all four independent datasets. It matches all 262,144 direct complex-core scalar outputs, all 263,168 real-DFT impulse values, all 6,682 general-probe values, and all 1,542,514 certified stage values. Every dataset reports zero numeric, signed-zero, and bit mismatches, with maximum absolute difference zero. The earlier external-source prerequisite and remaining-middle-stage hypothesis are disproved. `vdsp-kernel-recovery.txt` records the bounded experiments; no Apple framework binary or implementation code was copied into the deliverable.

## Preserved-frame comparison

The recovered candidate matches all 13 general rows and all 3,001 certified stage rows bit for bit. For comparison, radix-4 DIT has 2,097 and 414,809 numeric mismatches on those datasets; split-radix DIT has 2,093 and 415,960; ordinary radix-4 DIF has 2,387 and 416,428. `comparison.json` contains the complete metrics and row signatures.

`mel_stage_compare.py` previously computed the log candidate from its approximate mel projection, despite describing the input as certified. Commit `18248be2` now feeds certified `melproj` values into the known float32 guard-add and float64-log path. The repeated comparison reports `logmel: EXACT (384128 values)`. The full output is in `mel-stage-compare.txt`; its process exit remains 1 because Hann and mel projection still diverge.

No tolerance, reference lock, golden capture, execution policy, or release gate changed. The fixed Hann window and vDSP_dotpr accumulation remain separate frontend work.

## Commands

```bash
python3 overlay/tools/coreml/dft_geometry_probe.py make-input /tmp/coreml-dft-impulses.f32
cp overlay/tools/coreml/capture/Sources/mel-stage-capture/main.swift /tmp/parakeet-mel-stage/Sources/mel-stage-capture/main.swift
cd /tmp/parakeet-mel-stage && swift build -c release --product mel-stage-capture
cd /tmp/parakeet-mel-stage && .build/release/mel-stage-capture --mode probe-dft --waveform coreml-dft-impulses.f32 --out probe-dft-impulses-18248be2 && .build/release/mel-stage-capture --mode probe-dft-core --waveform coreml-dft-impulses.f32 --out probe-dft-core-impulses-18248be2
python3 overlay/tools/coreml/dft_geometry_probe.py compare --impulse-dump /tmp/coreml-dft-impulses-18248be2 --core-dump /tmp/coreml-dft-core-impulses-18248be2 --general-dump ~/.cache/mlx-omarchy/parakeet-reference/captures/b650695c-75aec2a/mel-stage-probes/dft --stage-dump ~/.cache/mlx-omarchy/parakeet-reference/captures/b650695c-75aec2a/mel-stage-probes/stage-capture --json-out receipts/2026-09-12-parakeet-dft-core-geometry/comparison.json
python3 overlay/tools/coreml/mel_stage_compare.py ~/.cache/mlx-omarchy/parakeet-reference/captures/b650695c-75aec2a/20260912T154759Z-librispeech/ane ~/.cache/mlx-omarchy/parakeet-reference/captures/b650695c-75aec2a/mel-stage-probes/stage-capture
/tmp/vdsp_compare /tmp/parakeet-mel-stage/coreml-dft-impulses.f32; /tmp/vdsp_compare /tmp/parakeet-mel-stage/dft-stage-pairs.f32; /tmp/vdsp_setup_resolution /tmp/parakeet-mel-stage/dft_in.f32; /tmp/vdsp_setup_resolution /tmp/parakeet-mel-stage/dft-stage-3001.f32; /tmp/vdsp_twiddle
```
