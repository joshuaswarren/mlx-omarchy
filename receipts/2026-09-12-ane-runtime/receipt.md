# ANE runtime acceptance receipt

## Source identity

- mlx-omarchy base: `9e22a534b9297242f360d32d26062cf2b4c2ab0d`
- runtime source commit: `382492a5896322b32d2058b20e3aa1c6c693aa4e`
- canonical libane repository: `joshuaswarren/omarchy-ane`
- canonical libane commit: `f261a6cb537aca62f267ad3d01beda0d6877544c`
- explicit ANEC compiler source: `aa688df66cbc2110e0df94f0d50fb72c7fa30a18`
- explicit package compiler binary SHA-256: `45fa6cb33e86ac9ec5e35421d338e07c9cd2cfb55b91e26242d9aa2ebeadffec`
- graph hash: `5584d0fd8d40027229890408e924e6f7930cd5f02516466a442193408482ce76`
- executing model: `openai-codex/gpt-5.6-sol`; no routing fallback

## Local source verification

The scoped build completed for `omarchy_ane_runtime_tests` and `mlx-omarchy-ane-smoke`. The scoped test command was:

```text
ctest --test-dir .work/build-ane-runtime -R '^omarchy_ane_runtime_tests$' --output-on-failure
```

It reported `1/1` passed and `0` failed. The real worker startup path on the x86 build host rejected execution before device work with the single named error:

```text
[omarchy-ane] runtime: ANE execution requires Linux aarch64 on the qualified base M1.
submitted=false
recovery=worker stopped before another submission; hardware recovery not required
```

## Bundle preparation

The repository adapter converted `receipts/fixtures/h13-explicit-chain-add-mul` to schema 3:

```text
h13_package_to_bundle: PASS programs=2 payloads=2 output=/tmp/AneRuntimeBridge-chain-bundle
```

The one-program canary is a strict schema-3 projection of program 0 from that same receipt-bound package. It retains `program-0.anec`, inputs `a` and `b`, output `sum`, compiler identity `aa688df66cbc2110e0df94f0d50fb72c7fa30a18`, and graph-source hash `5584d0fd8d40027229890408e924e6f7930cd5f02516466a442193408482ce76`. Its payload-collection identity was recomputed with the repository's canonical algorithm.

## Base M1 acceptance

The test ran on `jwm1-linux` while holding a single nonblocking `flock` on `/tmp/m1-gpu.lock`. `/tmp/m1-gpu.lease.AneRuntimeBridge` recorded the owner, PID, source commit, and start time while the lock was held.

Observed hardware identity:

```text
host=jwm1-linux
kernel=7.1.6-1-1-ARCH
machine=aarch64
driver_version=f2a3e5e+lifecycle6
driver_srcversion=DD43701FCB506056A340587
dt_compatible=apple,t8103-ane,
dt_status=okay
runtime_pm=on/active
```

The source-equivalent compact build used the committed `runtime.cpp`, `worker.cpp`, strict manifest/bundle loader, and the canonical libane `ane.c`. It was necessary because the isolated full CMake build reached the link step but `/tmp` had only 165 MiB free and `ar` reported `No space left on device`. The complete CMake build and scoped test had already passed on the integration host.

The locked acceptance command used a 30-second process deadline and a positive 10-second runtime deadline for each call. Observed results:

```text
one program, worker 3287335:
iteration 0 exact_fp16=PASS bytes=128
iteration 1 exact_fp16=PASS bytes=128
shutdown worker_pid=3287335 released_programs=1 process_released=true

two programs add then mul, worker 3287338:
iteration 0 exact_fp16=PASS bytes=128
iteration 1 exact_fp16=PASS bytes=128
shutdown worker_pid=3287338 released_programs=2 process_released=true

hardware_acceptance=pass
```

Inputs were exact fp16 `1` and `2`. The one-program expected output was exact fp16 `3`; the ordered add/mul expected output was exact fp16 `6`. Both executions were repeated in the same worker to prove buffer and program-handle reuse.

Post-run release checks reported:

```text
lease_file_released=true
lock_reacquire=pass
worker_processes_released=true
```

No GPU, CPU, simulator, or non-ANE fallback was present in either path.
