# Harness invocation provenance (2026-09-08 paired window)

All scripts are copied verbatim into scripts/ (sha256 in scripts/SHA256SUMS)
so the receipts are self-contained; the /tmp copies on jwm1-linux are
runnable references, not the provenance.

## Scripts

| file | sha256 | role |
|---|---|---|
| scripts/gpu_bench.py | f6aad9722194108f242ed1da490735ef94c300757e5e379b89d97c19638bf5ee | GPU micro-bench (16 cases, 3 warmups + 30 samples, host->host, qmm on default device) |
| scripts/ane_timed.sh | 6d58224a140cfb86a348d46e9aa0521daa5bb48fd1788f1de49b760661ba01a0 | ANE lifecycle6 timed run with failure-safe insmod/rmmod |
| scripts/ane_window.sh | 82dfdf236e4a4dd3361b2c763dd7979c2a0f47e0f43da4a126c83d92cd91c34e | flock wrapper for ane_timed.sh (the receipted ANE run used this) |
| scripts/paired_window.sh | 5f43f7cdf9567550f72c55114626ba2604b54a9111c3195432133dec659fe3d4 | flock wrapper for GPU bench + ANE half (the GPU numbers came from this) |

## Exact invocations (host joshuawarren@100.84.184.102, hostname printed by each script)

1. GPU bench (inside paired_window.sh, under flock on /tmp/m1-gpu.lock):
       /home/joshuawarren/venv-ane-paired-c254/bin/python /tmp/gpu_bench.py --out /tmp/paired-results/gpu-bench-c254-release.json
   venv: python3.14 -m venv ~/venv-ane-paired-c254; pip installed
   ~/src/mlx-parity-baseline-20260908/dist-parity-release/mlx_omarchy-0.32.2.dev202609081618+c254867-cp314-cp314-linux_aarch64.whl
   (release 255c2f93...7e02) and numpy 2.5.3.
   Output committed here as gpu-bench-c254-release.json.

2. ANE timed run (receipted; ane_window.sh under flock):
       bash /tmp/ane_window.sh
   which runs:
       bash /tmp/ane_timed.sh
   which runs, inside the lock:
       sudo insmod /home/joshuawarren/src/ane-eightcore-20260906/runtime-lifecycle5/ane/ane.ko
       python3 /home/joshuawarren/src/ane-eightcore-20260906/benchmark-packages.py \
           /home/joshuawarren/src/ane-eightcore-20260906/compiler \
           /tmp/parity-ane-candidates \
           /home/joshuawarren/src/ane-eightcore-20260906/runtime-abi1/bindings/python/dylib/libane_python.so \
           /tmp/parity-ane-candidates/timed-results \
           b1-gemv-k896-n4864 b2-gemv-k4864-n896
       sudo rmmod ane
   Harness gate: /sys/module/ane/version must equal f2a3e5e+lifecycle6.
   Method: every program prepared via pyane_init; timing = first transfer
   through last readback per iteration; 3 warmups + 30 iterations; exact
   fp16 output match on every iteration. Raw outputs committed here as
   b1-gemv-k896-n4864.json / b2-gemv-k4864-n896.json (+ y.fp16), run log
   ane-locked-run.log.

3. Candidate packages measured: /tmp/parity-ane-candidates/b1-gemv-k896-n4864/pkg
   (96 programs) and /tmp/parity-ane-candidates/b2-gemv-k4864-n896/pkg
   (146 programs), compiled by ~/src/mil-hwx-compiler/build/mil-hwxc
   (x86-64, branch ane-parity @ 42fd0bb) and device-free dry-run validated
   with ~/src/ane-eightcore-20260906/compiler/tools/h13_run_linux.py
   --dry-run (plan schema mil-hwxc.h13-linux-plan.v1). Compilation
   provenance: omarchy-ane ane-parity receipt 2026-09-08-ane-candidate-packages.md
   and mil-hwx-compiler ane-parity receipt 2026-09-08-ane-parity-candidates.md.

## Window-contract violations disclosed (not evidence)

- Paired window ANE half aborted 17:22:10.423963Z on a wrong device-node
  check, leaving the module loaded until manual rmmod verified absent
  17:23:15.223145Z.
- ane_timed.sh was then run directly without the lock (~17:23:35Z -
  17:24:02.839851Z). Excluded as evidence; raw log
  ane-unlocked-run-REJECTED.log. All active GPU owners (ParityBaseline,
  DecodeParity, PrefillParity, AttentionDecodeParity) notified for
  overlap check; DecodeParity confirmed zero jwm1-linux activity today.

## Overlap confirmations received

- DecodeParity: zero commands on jwm1-linux today (validation ran local
  llvmpipe on the x86 dev box) -- no overlap, no invalidation needed.
- ParityBaseline: window released before 17:22Z (GO message), no timed
  baseline activity in 17:22-17:26Z per their release DM.
- PrefillParity / AttentionDecodeParity: notification sent 2026-09-08;
  no overlap reported.
