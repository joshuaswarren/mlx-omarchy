# Encoder leftover fp16 linears flip e101: 104/104 (2026-09-15)

Verdict: running the 194 leftover GPU linears with fp16 chunked accumulation
flips the ANE-encoder E2E from 101/104 to **104/104, tokens_match true,
transcript_match true** (emission 101 emits 7892, not 8029). Encoder frozen
contract bounds all pass, and the hidden-state residual improves
(rel_l2 0.024917 → 0.023044). The runner fork is staged on jwm1, not landed
in `overlay/`; `63c1d3cf` not merged.

## Change

`vulkan_encoder.py` `linear` handler: instead of one fp32 GEMM, split K into
16-wide chunks; each chunk reduces in fp32, rounds to fp16, and accumulates
in fp16 ascending — the probed fp16 datapath that matches the ANE reference
arithmetic more closely than fp32.

Staged fork: `/var/tmp/ParakeetE2EFp16linear-stage/vulkan_encoder.py`
sha256 `974fc2a966e38719449a791088449ab09bcc3442ca7effdf8819903bd76fec19`,
a copy of the pinned stage runner
`cd858ed352f6b0d25e035643cef3e5bd43ad689614fe7e396a3f7af1d4edec6d` with only
the `linear` handler replaced (repo copy:
`receipts/2026-09-15-encoder-fp16-leftover/vulkan_encoder_fp16linear.py`).

`ponytail` ceiling: the chunk loop costs ~K/16 extra dispatches per linear;
encoder wall rose 18243.9 → 22641.2 ms (vk compute dispatches 5914 → 158100
on the diag-capture basis). A reshape-batched matmul form would need its own
decode A/B before replacing the loop; not attempted.

## Attribution (probe ≡ real arm, bit-for-bit)

`EncoderResidualE99/encoder_precision_probe.py` (stage-runner fork with
per-family precision knobs) run with `--islands ABC` and no knobs reproduces
the real ANE arm's `encoder_hidden` **bit-identically**: rel_l2 vs pin
`b3d60c81` = 0.000000, and its decode reproduces the e101 = 8029 miss.
Adding only `{"linear":"fp16"}` flips the decode to 104/104. The flip is
therefore attributable to the linear fp16 datapath alone.

| islands-ABC probe arm | rel_l2 vs golden | max_abs | mean_abs | decode |
| --- | ---: | ---: | ---: | --- |
| cr (control) | 0.024918 | 0.147156 | 0.004182 | prefix 101, e101 8029 |
| linear fp16 (**landed form**) | 0.023044 | 0.175354 | 0.003755 | **104/104 match** |
| linear+silu+sigmoid fp16 | 0.023471 | 0.161621 | 0.003831 | prefix 101 |
| linear+softmax+silu+sigmoid fp16 | 0.022900 | 0.138245 | 0.003797 | 104/104 match |

All-GPU probe arms (islands off) diverge at emission 97 — the ANE islands
carry emissions 97–100; only island-ABC arms are faithful models of the E2E.
`layer_norm` fp16 variants explode (rel_l2 ≈ 1.0 vs golden) and were never
candidates; fp16 LN is rejected by measurement, not by taste.

## E2E result (jwm1-linux, real pipeline)

Same command as `receipts/2026-09-15-parakeet-e2e-ane-bnns.json` with
`--encoder-runner` pointed at the staged fork and fresh out/scratch dirs.
Lock `/tmp/m1-gpu.lock`, `flock -w 900`, never steal, never unlink; nested
`flock -n` free after. Post-run liveness: 0 workers, `/dev/accel/accel0` held.

| quantity | value |
| --- | --- |
| status | **match** |
| emissions actual / native | 104 / 104 |
| matching_prefix_length | 104 (first_divergence null) |
| e101 | **7892** (was 8029) |
| durations_match / frame_indices_match | true / true |
| transcript_match | true, both sha256 `db501a8c080380ea027ffa50a4b4956c39df77cb692c4fb78e556311a11a0790` |
| encoder_max_abs_err | 0.175354 ≤ 0.3 PASS |
| encoder_mean_abs_err | 0.003755 ≤ 0.02 PASS |
| encoder_rel_l2_err | 0.023044 ≤ 0.1 PASS |
| nan / inf | 0 / 0 PASS |
| encoder_hidden sha256 | `38c73261f29230276ed76f1fc017b76b024156d79218bd5f1347fdc7e7d43ec7` (new pin; prior `b3d60c81`) |
| mel sha256 | `5b54f4a9a2ba3434cd69b6e48e6780d3bcb6c635d9ce85cda3d85c60f2455bde` (pinned mel, unchanged) |
| encoder_mask sha256 | `d8bf6ec9a4065ce6dff4d9252edc07e2910eeeac87f8fc3c05bd949e1e7861a7` (unchanged) |
| ANE | islands ABC, 72 submits, 72 worker starts, 0 timeouts, exec 5077.361 ms |
| ops | 96 ANE / 1278 GPU / 3351 executed, `cpu_tensor_events` 0 |
| stage walls ms | audio 252.9, mel 9744.8, encoder 22641.2, decode 2899.6, pipeline 35623.8 |

Bounds are the frozen contract in `parakeet-reference.lock`
(0.3 / 0.02 / 0.1); the 0.14716 figures in older receipts are the prior
pin's measured stats, not the bound.

## Identity

- e2e runner `/var/tmp/ParakeetE2E/parakeet_e2e.py` sha256 `1623707880831c963514eb07b533a3e707bfda6ce05208932a0a4522b10f12a7`
- decoder `vulkan_decoder.py` `637f077ece00025e5be3c341db256c8933ae19655804d3439490709d9ebc60ef`, LUT `fused_lut_2026-09-15.npz` `5a7591b89185b060ccf4f9135e6460a7429fe9988fad265b035e1dc3137b4cb7` (pkg `/var/tmp/ParakeetE2EAneBnns/pkg`, 17293b51 landing)
- libmlx `0947c063b4a370df90f1242b8657baf74daeac42208fe148a929f0b8c2374640` (venv-agxgen wheel `8f6de34c`, same as the 101/104 run), `Device(gpu, 0)` Apple M1 Honeykrisp
- worker `762dd1de…`, libane `1ab9d95d…`, bundles `island-attn-a-kt` / `island-pv` / `island-select-8head-scratch417`
- audio sha256 `30885601173f96b0d8ddd020dc959b055c6c1582b85a33e3fcab8c4b08ed94c2`, matches lock
- host jwm1-linux, kernel `7.1.6-1-1-ARCH`, aarch64, Python 3.14.7

## Offline A/B harness

`decode_ab.py` (this dir is on jwm1 at
`/var/tmp/EncoderResidualE99/decode_ab.py`) runs the pinned greedy TDT
decode on any `encoder_hidden.npy`. Controls: golden → 104/104 match; pin
`b3d60c81` → prefix 101, e101 8029 (reproduces the E2E miss exactly).
No-ANE probe arms decoded: cr/linear16/fp32chain/softmax16 diverge at e97;
silu16 reproduces the pin's 101/8029; LN-fp16 arms diverge immediately.

## Artifacts

`receipts/2026-09-15-encoder-fp16-leftover/`:
`e2e-report.json` (`e48435ed…`), `decode-ab-control.jsonl` (`c98a56c9…`),
`probe-abc.jsonl` (`f5960765…`), `run-report-abc-cr.json` (`31bc616d…`),
`run-report-abc-linear16.json` (`f712c096…`),
`vulkan_encoder_fp16linear.py` (`974fc2a9…`).

## Not claimed

- Receipt was written before the code landing: `5688f8bd` puts the same
  16-wide fp16-chunk linear into `overlay/tools/coreml/vulkan_encoder.py`
  (`974fc2a9…`), replacing the jwm1-only fork.
- No perf claim: the fp16 chunk loop is slower (+4.4 s encoder wall).
- No 450 ms hardware pass; mel stage here paid a cold `glslc`.
- `conv` fp16 knob raises `EncoderRunError` on this MIL (stmt 46); unused.
- Generalization beyond this pinned fixture is untested.

resolved_model: this session (`zai/glm-5.3-flash`).
