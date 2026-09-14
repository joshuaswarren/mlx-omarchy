# jw16 Mesa driver parity with jwm1: stock 26.2.2 -> Honeykrisp fork git-6f6afc8968

Date: 2026-09-14
Task: bring `jw16mbp1-linux` to the same Honeykrisp Mesa build jwm1 runs, then
re-measure the pinned Q4 decode/prefill protocol and report the delta.

## Verdict

The driver difference flagged as a confound in `receipts/2026-09-14-jw16-gpu-parity-rerun.md`
was a real and large jw16 loss. jw16 ran stock `extra/mesa 1:26.2.2-1`
(`driverInfo = Mesa 26.2.2-arch1.1`); jwm1 runs the fork package
`mesa-honeykrisp-omarchy 26.3.0.devel.hk6f6afc8-1`
(`driverInfo = Mesa 26.3.0-devel (git-6f6afc8968)`). The same package recipe was
built on jw16 from the same pinned fork commit
`6f6afc896844730f6d6c47f12a91145351cb4c28` and installed. jw16 now reports the
identical `driverInfo` string as jwm1.

Pooled medians over 6 reps per leg per driver (two independent A/B/A/B phases,
same wheel, same prompts, same lock):

- Q4 1053-token prefill: 2035.29 -> 3576.23 tok/s, x1.7571 (25.29% -> 44.43% of native M1 Max Metal)
- Q4 1053/32 decode: 86.61 -> 143.19 tok/s, x1.6532 (30.52% -> 50.45% of native)
- Q4 30-token prefill: 201.62 -> 360.99 tok/s, x1.7905 (13.29% -> 23.79% of native)
- Q4 30/32 decode: 149.17 -> 169.68 tok/s, x1.1375 (51.98% -> 59.13% of native)

Greedy identity is unchanged: all 24 legs produced the pinned native
generated-ID digests (`7fd25a869ff21678` short, `7da83f06ec9f001d` at 1053).

The dense fp16 2048-square `matmul(a,b)+bias` median is flat across the driver
change (8.6549 / 8.6567 ms stock vs 8.7056 / 8.6264 ms fork), so this is not a
general GPU throughput change: it lands on the quantized prefill/decode kernels.

A residual jw16 deficit remains after the driver is equalized. jwm1 sits at
71.25% / 67.70% of its own native base-M1 decode divisors and 60.38% of native
1053-token prefill (`receipts/2026-09-14-jwm1-gpu-parity-rerun.md`, not
remeasured here); jw16 on the same driver is at 59.13% / 50.45% / 44.43% of the
much higher native M1 Max divisors. The driver accounted for roughly half the
long-context gap, not all of it.

## Identity

- Receipt checkout (local mlx-omarchy): `b28deb1cbf7899cab71df08d432dd47cb31862ce` (`git describe --always --dirty`: `v0.3.2-4-gb28deb1c-dirty`; tree dirty from unrelated sibling work).
- Host: `jw16mbp1-linux`, `aarch64`, `Apple MacBook Pro (16-inch, M1 Max, 2021)`, Apple M1 Max T6001, GPU `Apple M1 Max (G13C C0)`, `nproc=10`.
- Kernel: `7.1.6-1-1-ARCH` (unchanged; no reboot).
- Measured wheel: `mlx-omarchy==0.32.2.dev202609122106+b41e2b74`, wheel file
  `dist/mlx_omarchy-0.32.2.dev202609122106+b41e2b74-cp314-cp314-linux_aarch64.whl`,
  SHA-256 `cb13927379ed9b9bce7be90475347b3d3291906be5411e14d784ccce03357bae`
  (same wheel before and after; the wheel was not rebuilt or reinstalled).
- Provenance on every leg: `verified=match harness=3db3cb9a-dirty core.cpython-314-aarch64-linux-gnu.so=sha256:4ad850da16c300ea libmlx.so=sha256:f2d45e601f05dd22`.
- Interpreter: `/home/joshuawarren/.local/share/mlx-omarchy-test-venv/bin/python` (Python 3.14.7); mlx-lm `0.31.3`.
- Model: `mlx-community/Qwen2.5-0.5B-Instruct-4bit`, snapshot `a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3` from the local HF cache, `HF_HUB_OFFLINE=1`.
- `scripts/bench_decode.py` SHA-256 `f5062d88f34b0845c1e59b0b35d2e33ae02180f636a0f65c543a4366ec2bef7f`; `scripts/bench_matrix.json` SHA-256 `df8eb9f3ed84182604379b7cde070ade706bfe590fbf83a35b1877cfaf43f258` (both identical to the 2026-09-13 and 2026-09-14 jw16 receipts).
- Runner `runner-jw16-hk-parity-run.py` SHA-256 `2c989dd7062111764a05a8ed2e16a3e0d91d4e64c92384adb9a5322c1546079d` (deployed as `/tmp/jw16-hk-parity-run.py`).
- Raw captures in this directory, hashes in `SHA256SUMS`.
- Agent model: `anthropic/claude-opus-5`; fallback: false.

### Driver versions, exact

| | before | after |
| --- | --- | --- |
| pacman | `mesa 1:26.2.2-1`, `vulkan-asahi 1:26.2.2-1`, `vulkan-mesa-implicit-layers 1:26.2.2-1` (all `extra/`) | `mesa-honeykrisp-omarchy 26.3.0.devel.hk6f6afc8-1` (local build) |
| `driverName` | Honeykrisp | Honeykrisp |
| `driverInfo` | `Mesa 26.2.2-arch1.1` | `Mesa 26.3.0-devel (git-6f6afc8968)` |
| Vulkan `apiVersion` | 1.4.354 | 1.4.359 |
| ICD | `/usr/share/vulkan/icd.d/asahi_icd.json` | `/usr/share/vulkan/icd.d/asahi_icd.aarch64.json` |
| `/usr/lib/libvulkan_asahi.so` SHA-256 | `765c4336b0d4b96e2576fe3243f05901550143d1996d5f3b3becd6080ff85bc9` | `8255dbbbfa51321b616b28d60647b89ae7cfba2c2071f27cc4bf29f9390caa06` |

jwm1's reference string, for the parity claim:
`Vulkan: Honeykrisp, Mesa 26.3.0-devel (git-6f6afc8968)`
(`receipts/2026-09-14-jwm1-gpu-parity-rerun.md` line 22). jw16 now reports the
same `driverName`/`driverInfo` pair from the same fork commit. No ICD override
and no `AGX_*`/`MESA_*`/`VK_*` environment variable was set in any measured run
(`vk_env` is empty in all four captures).

## Build

Built natively on jw16 from the pinned recipe already used for jwm1:
`mlx-omarchy-dattr-finish/packaging/mesa-honeykrisp-omarchy/PKGBUILD`,
SHA-256 `44d31562ae39a5afedb466ff26a573752af1813b320f6591b3497686088d487f`
(byte-identical to the file the 2026-09-08 jwm1 package receipt names).

```sh
# /home/joshuawarren/src/mesa-pkg-jw16-20260914
makepkg -s -C -f --noconfirm     # log: mesa-hk-build.log
```

- Second source file `Mesa-MLAA-License-Clarification-Email.txt` fetched from
  `AsahiLinux/PKGBUILDs` and verified against the PKGBUILD `sha512sums` entry
  (`ecdabad21a86f04...fd2a7e31`, match).
- Source clone HEAD: `6f6afc896844730f6d6c47f12a91145351cb4c28` (exact pinned commit).
- `prepare()` took the documented bindgen path: distro `rust-bindgen 0.73.2-1`
  is 0.73.x, so `cargo install --locked bindgen-cli 0.72.1` ran locally
  (log lines 20-70).
- Build: `==> Finished making: mesa-honeykrisp-omarchy 26.3.0.devel.hk6f6afc8-1 (Mon Sep 14 09:41:53 2026)`, `real 1m39.374s`, `EXIT=0`.
- Package: `mesa-honeykrisp-omarchy-26.3.0.devel.hk6f6afc8-1-aarch64.pkg.tar.xz`,
  SHA-256 `cad0718c3cb4bee066df21e73654d3b0568a98a6470004d58c2fa14630bcdee7`,
  11055120 bytes, at `~/src/mesa-pkg-jw16-20260914/`.
- The package bytes differ from the jwm1 package
  (`cef58afea49c39852dc023bf0eb0f7179166119f53c6d4b7db29be50d7ef0519`,
  `receipts/2026-09-08-honeykrisp-package.json`). Same source commit and same
  PKGBUILD, different build host and toolchain: jw16 built with
  `gcc 16.1.1+r12+g301eb08fa2c5-1`, `llvm 22.1.8-2`, `clang 22.1.8-1`,
  `rust 1:1.98.1-1`, `meson 1.12.0-1`, `ninja 1.13.2-3`. Parity is claimed on
  the driver source commit and the reported `driverInfo`, not on package bytes.
- `strings libvulkan_asahi.so` in the built package contains
  `Mesa 26.3.0-devel (git-6f6afc8968)`; the installed
  `/usr/lib/libvulkan_asahi.so` hashes to the package member.

## Install and rollback

jw16's stock layout differs from jwm1's: on jw16 the Vulkan Asahi driver and the
device-select layer ship as separate `extra/` packages (`vulkan-asahi`,
`vulkan-mesa-implicit-layers`), and the fork package neither conflicts with nor
replaces those names, so it file-conflicts with them. The install removes those
three stock packages, then installs the fork package:

```sh
sudo pacman -Rdd --noconfirm mesa vulkan-asahi vulkan-mesa-implicit-layers
sudo pacman -U --noconfirm .../mesa-honeykrisp-omarchy-26.3.0.devel.hk6f6afc8-1-aarch64.pkg.tar.xz
```

`install-hk.sh` (in this directory, deployed at
`~/src/mesa-pkg-jw16-20260914/install-hk.sh`) does exactly that and rolls back
automatically if either step fails.

Rollback artifacts, staged before any system change, in
`~/src/xbuild-drivers-jw16/stock-26.2.2/`:

| artifact | SHA-256 |
| --- | --- |
| `mesa-1:26.2.2-1-aarch64.pkg.tar.xz` | `038526d27535aa153fd76b556185c93e0957a23348898bb46860a92652fd5a26` |
| `vulkan-asahi-1:26.2.2-1-aarch64.pkg.tar.xz` | `f90b16a251666921eb67fdf4ee34c23b55e07e81e6c92351dbc125d05d2317ec` |
| `vulkan-mesa-implicit-layers-1:26.2.2-1-aarch64.pkg.tar.xz` | `f2f05995e01c3facd7b434e2e1e4980153acac4c11cddcbf7930f624d9b04244` |
| stock `libvulkan_asahi.so` | `765c4336b0d4b96e2576fe3243f05901550143d1996d5f3b3becd6080ff85bc9` |
| stock `asahi_icd.json` | `47accccf13966115c6a128ba9f99e5eb219bfc2d44dae4c380ed98171f9eabba` |

The rollback was **executed and verified**, not just written down: between the
two measurement phases jw16 was returned to stock and the restored
`/usr/lib/libvulkan_asahi.so` hashed back to `765c4336...` with
`driverInfo = Mesa 26.2.2-arch1.1`, then the fork package was installed again.

`pacman -U` alone cannot roll back: with `--noconfirm` it answers N to
"remove mesa-honeykrisp-omarchy?" and aborts with
`error: unresolvable package conflicts detected` (observed). The working
rollback removes the fork package first:

```sh
sudo pacman -Rdd --noconfirm mesa-honeykrisp-omarchy
sudo pacman -U --noconfirm \
  ~/src/xbuild-drivers-jw16/stock-26.2.2/mesa-1:26.2.2-1-aarch64.pkg.tar.xz \
  ~/src/xbuild-drivers-jw16/stock-26.2.2/vulkan-asahi-1:26.2.2-1-aarch64.pkg.tar.xz \
  ~/src/xbuild-drivers-jw16/stock-26.2.2/vulkan-mesa-implicit-layers-1:26.2.2-1-aarch64.pkg.tar.xz
```

That is `rollback-stock.sh` in this directory. Current jw16 state: fork package
installed (`mesa-honeykrisp-omarchy 26.3.0.devel.hk6f6afc8-1`).

Every package transaction on jw16 today, from `/var/log/pacman.log`:

```text
09:37:13 installed eglexternalplatform 1.2.1-1, egl-wayland 4:1.1.22-1, libomxil-bellagio 0.9.3-5
09:37:16 installed python-markupsafe 3.0.3-1, python-mako 1.3.12-1, spirv-llvm-translator 22.1.6-1
09:37:17 installed libclc 22.1.8-2, libmicrohttpd 1.0.10-1, debuginfod 0.196-1, valgrind 3.25.1-5,
         directx-headers 1:1.619.5-1, python-pyaml 26.7.0-1
09:37:20 installed rust 1:1.98.1-1, rust-bindgen 0.73.2-1
09:46:22 removed mesa 1:26.2.2-1, vulkan-asahi 1:26.2.2-1, vulkan-mesa-implicit-layers 1:26.2.2-1
09:46:23 installed mesa-honeykrisp-omarchy 26.3.0.devel.hk6f6afc8-1
09:47:25 removed mesa-honeykrisp-omarchy          (rollback verification)
09:47:26 installed mesa 1:26.2.2-1, vulkan-mesa-implicit-layers 1:26.2.2-1, vulkan-asahi 1:26.2.2-1
09:47:54 removed mesa/vulkan-asahi/vulkan-mesa-implicit-layers, installed mesa-honeykrisp-omarchy
09:54:37 removed mesa-honeykrisp-omarchy, installed the three stock packages
                                                 (coopmat capability probe, below)
09:54:45 removed the three stock packages, installed mesa-honeykrisp-omarchy   (final state)
```

The `makedepends` at 09:37 were installed by `makepkg -s` during a first build
attempt whose output was not captured (a `tee` into a not-yet-created log
directory); that attempt was stopped and its `src/` discarded before any
package was produced. The logged build in `mesa-hk-build.log` is a full
`makepkg -s -C -f` from a clean `src/` that found those makedepends already
satisfied, which is why it installs nothing and still rebuilds everything from
a clean tree (ninja reports 1625 edges; `real 1m39.374s`, `user 11m37.132s` on
10 cores). Those build-time packages were left installed; they are additive and
do not affect the runtime driver.

## Protocol

Identical to `receipts/2026-09-14-jw16-gpu-parity-rerun.md`: `MLX_DISABLE_COMPILE=1`,
`HF_HUB_OFFLINE=1`, greedy `--temp 0.0`, `--seed 0`, EOS suppressed,
`--tokens 32`, `--warmup-tokens 4`, decode over 31 inter-token gaps,
`bench_matrix.prompt_text` for `short` (2 UTF-8 bytes, 30 chat-template tokens)
and `ctx1024` (4759 UTF-8 bytes, 1053 chat-template tokens), fresh subprocess
per leg, `--wheel` provenance gate armed on every leg.

Changed from that receipt, deliberately: each phase runs **3 reps per leg with
the legs alternating** (short, ctx1024, short, ctx1024, short, ctx1024) and the
phase sequence is A/B/A/B (stock, fork, stock, fork) so the delta does not rest
on one warm run or on one driver ordering. Quoted per-driver numbers are the
median of the 6 reps pooled across the two same-driver phases.

Each phase is one outer lock holding both legs and the matmul:

```sh
ssh jw16mbp1-linux 'timeout -k 10s 1800s flock -w 60 /tmp/m1-gpu.lock \
  /home/joshuawarren/.local/share/mlx-omarchy-test-venv/bin/python \
  /tmp/jw16-hk-parity-run.py <phase-label> 3'
```

The 2048-square fp16 `matmul(a,b)+bias` check runs inside the same lock after
the decode legs: `mx.gpu`, seed `20260913`, 3 warmups, 20 measured synchronized
evaluations, median, TFLOPS from `(2*N^3+N^2)/elapsed`. Its exactness fixture is
this runner's own (`[[1,2],[3,4]]@[[5,6],[7,8]] + [1,1]`, expected
`[[20,23],[44,51]]`, matched in all four phases) and is **not** the
`[[20.0, 21.0], [43.5, 49.5]]` fixture of the earlier receipts, so the matmul
exactness lines are not comparable across receipts; the medians are.

Phase timestamps and wall times (`started_iso`, seconds):
`before-A 2026-09-14T09:35:20-0500 14s`, `after-A 09:46:40 12s`,
`before-B 09:47:34 15s`, `after-B 09:48:06 12s`. The build finished 09:41:53 and
the machine was otherwise idle; `llm-inference.service` was `inactive` in all
four captures.

## Results

Native M1 Max Metal divisors are the precise 2026-09-10 16m1mbp 12-rep Metal
medians used by the prior jw16 receipts: 286.96 / 283.79 tok/s decode and
1517.55 / 8048.42 tok/s prefill. jwm1 columns are quoted from
`receipts/2026-09-14-jwm1-gpu-parity-rerun.md` against its own base-M1
divisors and were **not** remeasured here.

| Metric | jw16 stock 26.2.2 (median of 6) | jw16 Honeykrisp 6f6afc8968 (median of 6) | fork/stock | stock % native | fork % native | jwm1 % native (base-M1) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q4 short decode, 30/32 | 149.1707 tok/s | 169.6769 tok/s | x1.1375 | 51.98% | 59.13% | 71.25% |
| Q4 short prefill, 30 tokens | 201.6151 tok/s | 360.9891 tok/s | x1.7905 | 13.29% | 23.79% | 112.56% |
| Q4 1K-context decode, 1053/32 | 86.6097 tok/s | 143.1857 tok/s | x1.6532 | 30.52% | 50.45% | 67.70% |
| Q4 1K-context prefill, 1053 tokens | 2035.2917 tok/s | 3576.2252 tok/s | x1.7571 | 25.29% | 44.43% | 60.38% |

All six reps per driver, in run order (A phase then B phase):

```text
short decode   stock: 149.1506 149.1908 150.1334 | 148.8602 149.6487 143.9514
short decode   fork : 169.4364 170.2156 169.8483 | 169.5056 170.4748 166.6511
short prefill  stock: 201.8023 201.4279 205.6507 | 205.6714 201.0701 201.2192
short prefill  fork : 364.1385 360.1814 367.6599 | 357.6938 361.7968 358.0063
1053 decode    stock:  86.6538  86.5655  87.3766 |  86.8127  85.5886  84.6686
1053 decode    fork : 143.6736 144.2785 138.9351 | 142.6979 145.5036 140.8158
1053 prefill   stock: 2032.1229 2041.9852 2064.2161 | 2036.2244 2034.3589 2033.4957
1053 prefill   fork : 3600.8673 3590.3021 3562.1482 | 3601.7529 3560.0705 3556.8247
```

The stock and fork distributions do not overlap on any of the four metrics, and
each driver's two phases agree with each other, so the delta is the driver, not
drift or thermals.

Dense fp16 2048-square matmul+add, same lock, per phase:

| phase | driver | median ms | TFLOPS |
| --- | --- | ---: | ---: |
| before-A | Mesa 26.2.2-arch1.1 | 8.654942 | 1.9854625817 |
| after-A | Mesa 26.3.0-devel (git-6f6afc8968) | 8.705622 | 1.9739041608 |
| before-B | Mesa 26.2.2-arch1.1 | 8.656743 | 1.9850495143 |
| after-B | Mesa 26.3.0-devel (git-6f6afc8968) | 8.626430 | 1.9920250305 |

Flat within 0.9%, and consistent with the 2026-09-13/14 jw16 medians
(8.712577 / 8.717198 / 8.680269 ms). The driver change moves the quantized
decode/prefill path only.

Generated-ID digests, all 24 legs:

```text
short:   7fd25a869ff21678, n=32, first=9707,0,2585 last=646,387,7881
1053:    7da83f06ec9f001d, n=32, first=13060,498,369 last=3897,553,279
```

Both match the pinned native macOS digests, on both drivers, so the fork driver
changes rates without changing greedy output.

### Raw bench_decode stdout, one leg per driver

Stock, 1053:

```text
provenance: mlx-omarchy 0.32.2.dev202609122106+b41e2b74 mx=0.32.2.dev202609122106+b41e2b74 verified=match harness=3db3cb9a-dirty core.cpython-314-aarch64-linux-gnu.so=sha256:4ad850da16c300ea libmlx.so=sha256:f2d45e601f05dd22
decode 86.65 tok/s over 31 tokens (32 requested, EOS suppressed)
prefill 0.518s (reported separately, excluded from decode)
prompt_tokens 1053
decode mean per-token 11.5 ms
generated_ids sha256:7da83f06ec9f001d n=32 first=13060,498,369 last=3897,553,279
{"decode_tps": 86.6538, "device": "Apple M1 Max (G13C C0)", "engine": "bench_decode", "generated": 32, "ids_first": [13060, 498, 369], "ids_last": [3897, 553, 279], "ids_sha256_16": "7da83f06ec9f001d", "prefill_s": 0.518177, "prefill_tps": 2032.1229, "prompt_tokens": 1053}
```

Fork, 1053:

```text
provenance: mlx-omarchy 0.32.2.dev202609122106+b41e2b74 mx=0.32.2.dev202609122106+b41e2b74 verified=match harness=3db3cb9a-dirty core.cpython-314-aarch64-linux-gnu.so=sha256:4ad850da16c300ea libmlx.so=sha256:f2d45e601f05dd22
decode 143.67 tok/s over 31 tokens (32 requested, EOS suppressed)
prefill 0.292s (reported separately, excluded from decode)
prompt_tokens 1053
decode mean per-token 7.0 ms
generated_ids sha256:7da83f06ec9f001d n=32 first=13060,498,369 last=3897,553,279
{"decode_tps": 143.6736, "device": "Apple M1 Max (G13C C0)", "engine": "bench_decode", "generated": 32, "ids_first": [13060, 498, 369], "ids_last": [3897, 553, 279], "ids_sha256_16": "7da83f06ec9f001d", "prefill_s": 0.29243, "prefill_tps": 3600.8673, "prompt_tokens": 1053}
```

Full stdout for all 24 legs is in the four `capture-*.json` files.

## Mechanism: stock has no cooperative matrix at all

Measured directly on jw16 by toggling the package (fork -> stock -> fork, all
three states verified by `driverInfo`), no ICD override:

| | stock `Mesa 26.2.2-arch1.1` | fork `Mesa 26.3.0-devel (git-6f6afc8968)` |
| --- | --- | --- |
| `vulkaninfo \| grep -ci VK_KHR_cooperative_matrix` | 0 | 2 (`extension revision 2`) |
| `mx.device_info()["cooperative_matrix_f32_8"]` | 0 | 1 |
| `api_version` | 1.4.354 | 1.4.359 |

So the stock driver does not expose `VK_KHR_cooperative_matrix` on this device,
and mlx-omarchy never selects `QmmPrefillCoopmatF16` there: every stock number
in this receipt and in the earlier jw16 receipts is a non-coopmat fallback path,
not a slower coopmat kernel. That is the mechanism behind the x1.66-x1.79 jump
on the three quantized legs and behind the flat dense fp16 matmul, which does
not use that extension. `QmmPrefillOpt` reports the same reading from the
2026-09-12 agx-qmm-codegen receipt on jwm1.

Both `architecture` and `driverName` read `honeykrisp` on stock too: the Asahi
Vulkan driver is named Honeykrisp upstream, so the driver *name* alone never
distinguished the two builds. Only `driverInfo` and the coopmat capability do.

## Consequence for the prior jw16 receipts

`receipts/2026-09-13-jw16-q4-decode/`, `receipts/2026-09-14-jw16-gpu-parity-refresh.md`
and `receipts/2026-09-14-jw16-gpu-parity-rerun.md` were all measured on stock
`Mesa 26.2.2-arch1.1`. Their jw16-vs-jwm1 rows compared two different drivers
and understate jw16 by 1.14x to 1.79x depending on the leg. The stock numbers
here reproduce them (short decode 149.17 vs 147.28/150.12, 1053 decode 86.61 vs
87.45/87.10, 1053 prefill 2035.29 vs 2072.03/2067.98), so those receipts are
valid measurements of the stock driver, not of the project's driver.

## Lock and hardware safety

- `/tmp/m1-gpu.lock` inode 12 in all four phases, taken with `flock -w 60`, never stolen, never unlinked; the lock was free before each phase.
- Nested `flock -n /tmp/m1-gpu.lock -c true` returned 1 inside every phase.
- The build and the package install ran outside the lock and touched no GPU job.
- Power on AC throughout (`macsmc-ac/online=1`, battery `Full`, `tps6598x-source-psy-0-003a/online=1`).
- No reboot, no device-tree change, no `1x896`, no SET write, no ANE access. `ane` is loaded with refcount 0 (`ane 65536 0`) after this work and was never loaded, unloaded, or opened by it; `accel0` was not touched.
- `llm-inference.service` `inactive` in all four captures and after the run; after the last phase `flock -n /tmp/m1-gpu.lock` succeeded, inode 12 still present, and no `bench_decode`, `deqp`, `llama` or leftover mlx process was running.
- Nothing was run on jwm1: every jwm1 number here is quoted from an existing receipt.
