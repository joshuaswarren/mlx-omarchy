# Per-dispatch fixed cost on the M1 — characterization and cuts (2026-09-07, DispatchFloor2)

Repo `/tmp/mlx-release-f5-final` @ `4e116e54` plus the uncommitted overlay edits listed under "Changes".
Hardware: jwm1 Apple M1 (G13G B1), Mesa 26.1.7 Honeykrisp, Vulkan 1.4.354, glslc 2026.3.
Second driver: llvmpipe (LLVM 22.1.8) on the x86 build host (`MLX_OMARCHY_ALLOW_NON_APPLE=1`).
Bench: `overlay/tests/omarchy/bench_dispatch_floor.cpp` (target `omarchy_dispatch_floor_bench`, registered in
`overlay/tests/omarchy/CMakeLists.txt`, not a ctest). Every M1 run: `flock /home/joshuawarren/benchq/gpu.lock`,
`taskset -c 4-7`, N = 1000 nodes in one submission, best of 3 trials per row, three runs
(`dispatch-floor-before-m1.jsonl` = tree before any change, `dispatch-floor-after-m1.jsonl` = final tree).
`gpu_us` = fence-bounded device time / N; `record_us` = host recording time / N.

## Result

| dispatch (ElementwiseF16 add, 896 f16, 4 workgroups) | before | after |
|---|---|---|
| raw production path (barriers + descriptors + push constants) GPU us | 8.71 | **3.37** |
| raw floor (no barrier, no descriptor bind, no push constants) GPU us | 8.71 | 2.93 |
| raw with `params.count = 0` (launch + preamble only) GPU us | 6.35 | 2.96 |
| FillF16 reference (14-instruction shader), production path GPU us | 2.68 | 2.68 |
| encoder level (`CommandEncoder::dispatch_compute` x1000, commit, synchronize) us/dispatch | 23.08 | 7.60 (pooled) / 9.35 (push) |
| op level (`eval` of 1000 chained `add`s) us/op | 17.43 | 8.30 |
| agx shaderdb ElementwiseF16 (instrs / gprs / uniforms / preamble instrs) | 1120 / 73 / 392 / 250 | 20 / 5 / 32 / 26 |

Parent's end-to-end A/B of the intermediate tree (op specialization only, perfsnap2, 14:55): decode 70.1/58.9/42.2 -> 76.5/63.3/44.6 tok/s (+9%), token IDs equal.

## Mechanism breakdown (measured, GPU us per node, M1)

| mechanism | number |
|---|---|
| Honeykrisp launch + CDM cache flush floor (vkCmdCopyBuffer / vkCmdFillBuffer, which hk implements as compute) | 1.4 - 1.6 |
| smallest real shader (FillF16, 14 instrs, 22 preamble instrs), production path | 2.5 - 2.7 |
| ElementwiseF16 before: shader launch/preamble/uniform setup with no body work (`count = 0`) | 6.35 => ~3.7 us of shader setup |
| ElementwiseF16 before: main body on 896 elements (8.71 - 6.35) | ~2.4 |
| ElementwiseF16 after specialization: setup 2.96 (= FillF16), body ~0.4 | |
| pre+post barriers per dispatch (production 3.37 vs no-barrier 2.57 after; 8.71 vs 8.67 before) | 0 - 0.8 (hidden behind a heavy shader; visible once the shader is small) |
| single pre barrier instead of pre+post (pre-only row) | ~0.6 saved vs pre+post |
| descriptor set alloc+update+bind vs bind-only vs none (desc-* rows) | 0.0 (GPU), 0.1 host |
| push-constant size 128 vs 16 vs 0 bytes | 0.0 |
| redundant vkCmdBindPipeline | 0.0 |
| submission cost: 10 submits/1000 nodes +0.07 us/node; 1 node per submit +30 us/node; serial submit+wait ~90-200 us | decode has 7 submits/token: ~0.2 ms |
| workgroups: 1 group slower than 4-64 (grid-stride loop), 1024 groups +3 us | |

Honeykrisp emits a CDM cache flush after every compute dispatch (`hk_dispatch_with_usc_launch`, mesa-26.1.7 src/asahi/vulkan/hk_cmd_dispatch.c), so Vulkan barriers add little on this driver; the per-dispatch cost was the shader's preamble (uniform hoisting: the driver moves every uniform-only computation, including the reciprocal for each `x % uniform`, and every constant table into a per-dispatch single-thread preamble) plus its uniform count.

Host side: recording cost is 0.5-0.7 us/dispatch on the M1 P cores at full clock but 12-13 us on the E cores or a cold P core (the bench now spins the CPU 300 ms before host-timed levels; `DISPATCH_FLOOR_NO_WARM=1` shows the unwarmed number). Parent's taskset A/B: no end-to-end effect (host is not on the critical path at 585 dispatches/token), so no affinity mechanism was built. Push descriptors vs pooled sets, alternating x4, N=5000: record 0.60-0.63 vs 0.68-0.82 us/dispatch, total equal (GPU-bound).

Other decode kernels' shaderdb (M1, `dispatch-floor-shaderdb-m1.txt`, instrs/gprs/uniforms/preamble): QmmVecSubgroupF16 423/45/148/205, QmmVecQ4WordSubgroupF16 114/27/66/45, FastRopeF16 244/19/182/203, MatmulF16 216/35/130/102, SoftmaxF16 235/21/80/74, FastRmsNormF16 116/17/64/46, CopyGeneralF16 97/23/54/41, DequantF16 271/25/92/135, TakeF16 202/23/96/79. At the measured ~15 ns per preamble instruction, FastRopeF16 (48/token) is the next target (~0.15 ms/token); the bench has `rope-production` / `rope-production-count0` rows ready (llvmpipe-verified, not yet measured on the M1).

## Changes (overlay, uncommitted)

1. `shaders/elementwise.comp`: specialization constants `OPERATION` (constant_id 0, folds the 41-case op switch) and `LAYOUT` (constant_id 1; 1 = both operands dense and `count` long, no broadcast index math). Unspecialized defaults keep the runtime paths.
2. `compute.h/.cpp`: `ComputeRuntime::pipeline(kernel, params)` derives the key per kernel (`specialization()`: elementwise F32/F16/BF16 only) and caches one pipeline per (kernel, layout, operation); `pipeline(kernel)` stays the unspecialized pipeline. Ctor takes `push_descriptors` and sets `VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR`.
3. `encoder.cpp/.h`: one memory barrier ahead of every node (dispatch, copy, fill) and a device-to-host barrier closing every batch; the post-dispatch barrier is gone. Coverage is a superset of before: previously copy->copy, copy->fill and copy->host had no barrier at all in the default mode. Gated mode unchanged. Push descriptors when `Device::push_descriptors()` (`vkCmdPushDescriptorSetKHR`), pooled sets otherwise.
4. `device.h/.cpp`, `vulkan.h`: `CapabilityReport::push_descriptor`, extension enabled at device creation, `MLX_OMARCHY_NO_PUSH_DESCRIPTORS` kill switch, table entries `CmdPushDescriptorSetKHR`, `ResetDescriptorPool`.
5. `trace.h` comment, `docs/install-omarchy.md` (barrier and push-descriptor/specialization paragraphs), bench + CMake registration.

## Suites

M1 (`/home/joshuawarren/benchq/DispatchFloor2/build`, final tree, push and `MLX_OMARCHY_NO_PUSH_DESCRIPTORS=1`):
omarchy_runtime_tests 40/40, omarchy_primitive_tests 99/99, omarchy_compiled_tape_tests 11/11, omarchy_copy_offset_tests 25/25 — all SUCCESS in both modes.

llvmpipe (`/tmp/mlx-dispatch-floor-work/build`, final tree, two primitive runs plus both modes earlier):
runtime 40/40, compiled_tape 11/11, copy_offset 25/25 SUCCESS; primitive 97/98 with the same pre-existing failure as the untouched baseline run at 14:20 (`test_primitives.cpp:5743/5789`, quantize/dequantize error-message text; passes on the M1 build, so it is the local work tree's patch state, not this change). One earlier llvmpipe run of the intermediate tree (specialization, pre+post barriers) failed `test_primitives.cpp:4898` (categorical determinism) once in each descriptor mode; it passed 3/3 in isolation and 2/2 full runs after the single-barrier change, which is also the change that adds barriers between consecutive transfer nodes.

## Not done / open

- FastRopeF16 and the other preambles above (parent's extended ownership): measured and ranked, not changed.
- Push descriptors are a ~0.1 us/dispatch host win only; keep or drop is the parent's call (`MLX_OMARCHY_NO_PUSH_DESCRIPTORS=1` restores the pooled path; the layout flag makes the two paths exclusive per device).
- With push descriptors, `MLX_OMARCHY_TAPE_NO_REUSE` only affects the allocator (documented).
- Cross-run GPU timing noise on the M1 is +-0.4 us (DVFS); all rows are best-of-3 across three runs.
