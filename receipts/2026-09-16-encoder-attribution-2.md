# Encoder attribution 2 on 1f830a0d: the 6.75 s decomposes, and the fp16
# pointwise fusion lands −375.6 ms of it (2026-09-16)

Attribution plus one landed identical-bytes cut. Re-attribute of the
non-ANE encoder time on the `7d82ec94`/`1f830a0d` baseline (receipts/
`2026-09-16-parakeet-e2e-baseline.md`: encoder_ane 9375.2 ms, ANE worker
exec 2623.3 ms → 6.75 s of GPU + host) with a diag wheel built at
origin/main `1f830a0d`, then the single highest-value identical-bytes cut
the buckets pointed at.

## Buckets of the 6.75 s

Two phase-isolated captures of the encoder leg only (standalone
`vulkan_encoder_instr.py`, warm unprofiled + record with
`MLX_OMARCHY_GPU_PROFILE`), additive timestamps only (host buckets:
per-round input staging, batch round, output pack, final drain, whole
`run()`; GPU busy: device ticks, `period_ns=1`). Record capture: 5202 vk
compute dispatches — the fused receipt's count — GPU busy 4074.2 ms.

| bucket | ms | % of 6.75 s |
| --- | ---: | ---: |
| **GPU busy** (device ticks, record capture) | **4074.2** | **60.4%** |
| — MatmulF32Coopmat (194 disp; the 194 leftover-batched fp16 linears) | 1453.6 | 21.5% |
| — custom fused chains e428 (698 disp; LN/silu/GLU/sm + leftover chain) | 829.3 | 12.3% |
| — ElementwiseF32 (762 disp; pointwise f32 chains) | 596.3 | 8.8% |
| — MatmulF32 batched (50 disp, gy=24; the 48 FF linears' xb@wb) | 335.0 | 5.0% |
| — SliceUpdatePairF16 (605 disp; transpose/copy staging) | 320.8 | 4.7% |
| — CastF16F32 1616 + CastF32F16 615 | 221.5 | 3.3% |
| — ReduceF32 264 / CopyGeneralF16 225 / ConvF16 27 / rest | 317.7 | 4.7% |
| **host graph residual** (mx statement build between submits; run − stage − submit − pack − drain − batch_open, warm leg) | **2275.2** | **33.7%** |
| **host staging around the single batch submit** (eval-drain host cost 4300.1−4074.2 + output pack 97.8 + batch_open 23.1) | **346.8** | **5.1%** |
| **output/tail drain** (final wanted eval, warm) | **81.1** | **1.2%** |
| **total** | **6777.3** | **100.3%** |

Cross-check: warm unprofiled run() wall 9407.0 ms vs the baseline receipt's
9375.2 ms clean median (+0.3% run noise); sum above uses the 6751.9 ms
clean budget, so the +25.4 ms (+0.4%) is measurement noise, not a missing
bucket. There is no fourth bucket: the ANE batch round (2605–2630 ms host
round time; island attn-a 1178.8 + select 924.0 + pv 387.7 ms in the E2E
record capture) is the part of the encoder wall outside the 6.75 s, and
GPU busy sits inside the per-round input staging because every
`mx.eval` joins before the batch round submits.

Host note: the profiler's own per-dispatch host cost is 18.9 ms of the
5202 (`h` field); the profiled-vs-clean wall delta is ≈ +200 ms, all in
host buckets, which is why quoted host numbers come from the warm
unprofiled leg.

### What the buckets ruled out

- MatmulF32Coopmat 1453.6 ms is the largest single line, but its known
  identical-bytes attack (block-rounded fp16 coopmat linear) was measured
  slower twice and reverted (`531c48a7`, `b9b5bf69`, `7d82ec94`) — the
  coopmat linears and the batched-linear fp16 chain are rounding-pinned
  and untouchable without a new idea.
- The remaining ElementwiseF32 + boundary casts are the fusion residue:
  248 fp16 add/sub/mul and 24 sigmoid statements each lowered to
  cast-in ×2 + ElementwiseF32 + cast-out = 4 dispatches of pure glue.

## The landed cut: fused fp16 pointwise

`_pw` + `_pw_kernel` + `_sigmoid_kernel` in
`overlay/tools/coreml/vulkan_encoder.py`: one custom kernel per
(add, sub, mul) × (same-shape flat, broadcast scalar, conv-stem
row-broadcast) plus a sigmoid kernel, replacing the 4-dispatch chains.
Each kernel widens the identical fp16 operands to fp32, runs the
identical fp32 op (the sigmoid body is the documented mx.sigmoid lowering
`1.0 / (1.0 + exp(-x))`, the silu kernel without its multiply), and
applies the same single round-to-nearest-even at the fp16 boundary —
bit-identical by construction. Guards fall back to the unchanged chain
for every operand pair that does not match the three exact layouts (the
63 int32 mask adds stay on the int path). Grid convention is thread
counts per axis: oversized tensors (the 24.5 M-element conv-stem muls)
spread the flat index across y around the 65535-workgroup x limit with a
baked bounds guard.

Fused statements in the pinned program: 120 add + 128 mul + 24 sigmoid
(the 48 broadcast selects already lower to SelectF16 at 2.4 ms total and
were left alone; pad/relu were priced at <25 ms and left alone).

### Result

| quantity | before (`1f830a0d`) | after (this branch) |
| --- | ---: | ---: |
| encoder_ane median, 6 E2E runs | 9375.209 ms | **8999.640 ms (−375.6, −4.0%)** |
| encoder_ane individual runs | 9375.2 median set | 8915.2 / 8965.1 / 8968.4 / 8999.6 / 9069.2 / 10029.9 |
| vk compute dispatches (encoder leg) | 5202 | **4485 (−717)** |
| GPU busy (device ticks) | 4074.2 ms | **3736.4 ms (−337.8)** |
| — ElementwiseF32 | 762 / 596.3 ms | 523 / 301.2 ms |
| — CastF16F32 | 1616 / 165.1 ms | 1138 / 130.1 ms |
| — CastF32F16 | 615 / 56.4 ms | 376 / 37.7 ms |
| — custom kernels e428 | 698 / 829.3 ms | 937 / 831.5 ms |
| ANE batch round (host round time) | 2623.3 ms | 2599.3 ms |
| `encoder_hidden` sha256 | `38c73261…` | **`38c73261…` (pin, every run)** |
| E2E status / emissions | match, 104/104 | **match, 104/104, all 6 runs** |
| transcript sha256 | `db501a8c…` | **`db501a8c…` (every run)** |

After-cut GPU busy −337.8 ms of the −375.6 wall; the rest is dispatch and
statement host cost. The r10 run (10029.9 ms) is an outlier on an
otherwise flat 8915–9069 spread; the median absorbs it.

Bit-exactness evidence, in order: a 12-case unit harness
(`/var/tmp/enc-attr2/pwcheck.py`, release wheel) comparing every fused
path against the old dispatch chain on the exact operand shapes of the
pinned program — ALL-EXACT, bit-for-bit on uint16 views; then the
standalone encoder legs and all six full E2E runs reproducing the
`encoder_hidden` pin and transcript sha exactly.

## Identity

- Host `jwm1-linux`, aarch64, kernel `7.1.6-1-1-ARCH`, 8 cores,
  `/dev/accel/accel0`, Vulkan Mesa Honeykrisp, device Apple M1 (G13G B1),
  `timestamp_period_ns: 1`.
- Lock `/tmp/m1-gpu.lock`, inode 29 throughout, `flock -w 900`, never
  stolen, never unlinked; `flock -n` free after every battery;
  every leg announced between windows.
- Diag wheel: `mlx_omarchy-0.32.2.dev202609160555+diag.1f830a0d`
  sha256 `56ebf422d8e3e5ae9ede2b44a00f0b2fbbdefe7440000a0f1ec948ceeb3959ea`,
  built with the proven recipe (`/var/tmp/enc-attr2-build.sh`,
  worktree `/var/tmp/enc-attr2-src` at `1f830a0d`, mlx pin `1f8e74e3`,
  `MLX_OMARCHY_ANE_SOURCE_DIR` → omarchy-ane `6fa243a`,
  `-DMLX_OMARCHY_GPU_PROFILING=ON`); gate literal count in `libmlx.so`: 3.
  Diag wheel used ONLY for attribution captures; every timing and identity
  claim for the cut runs on the release baseline wheel
  `e54cb680…` (site `/var/tmp/ParakeetE2EBaseline7d82/site`).
- Instrumented stage runner `/var/tmp/enc-attr2/vulkan_encoder_instr.py`
  sha256 `86fdc32300fc40047e3a71c5252ae504beef6736a41948334c983967a9932e1d`
  (base `240e3b63…` = origin/main overlay, +30 lines timing/print only).
- Cut runner staged at `/var/tmp/enc-attr2/vulkan_encoder_cut.py`, byte
  from this branch's `overlay/tools/coreml/vulkan_encoder.py`.
- Strict worker `f171a61e…` + libane-strict `56b46234…`, bundles
  `/var/tmp/island-reexport/bundles`, `ANE_ISLAND_MODE=resident-batch`:
  1 batch submission, 1 worker start, 72 rounds, 0 timeouts on every run.
- ANE worker-side exec per island (E2E record, cut): attn-a 1178.8 +
  select 924.0 + pv 387.7 ms.
- Artifacts on jwm1: `/var/tmp/enc-attr2/` (diag wheel dist, site,
  `profile-enc-record.jsonl` before / `profile-enc-cut.jsonl` after,
  `cut-out-*` E2E reports, `pwcheck.py`, build/leg scripts).

## Not claimed

- No claim on the coopmat/matmul lines (1453.6 + 335.0 ms): untouched,
  still the largest GPU bucket, known attacks measured slower or
  rounding-unsafe.
- No claim on macOS, other fixtures, cold-SPIR-V walls, or the TDT leg.
- The host graph residual (2275.2 ms) is priced as a bucket, not attacked:
  it is the cost of the faithful MIL interpreter itself.
- `63c1d3cf` not merged, not touched.
