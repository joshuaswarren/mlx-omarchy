# Community report: MacBook Air 13-inch 2024 (M3, `apple,t8122`)

Submitted by a contributor by hand (collector archive, redacted by the
collector: 24 home-path replacements, no names or addresses). Wheel v0.3.8
(`f5ba1c8`), collector at source `5516b28c`, Python 3.14.7, kernel
`7.1.13-401.asahi.fc44.aarch64+16k`, Omarchy boot entry `test-fedora-7.1.13`,
16 GB, 8 cores online.

## What it shows

- **No Apple GPU is exposed to Vulkan on this machine.** `vulkaninfo` lists
  only `llvmpipe (Mesa 26.1.8)`. The kernel command line carries
  `modprobe.blacklist=appledrm module_blacklist=appledrm`, so the display/GPU
  driver is off and Honeykrisp has no device. That is the current M3 state
  under this Omarchy Mac kernel, not a fault in the report.
- **mlx-omarchy behaved as designed:** every probe stopped with the named
  refusal `device 'llvmpipe ...' is not an Apple GPU running Omarchy
  Honeykrisp` instead of silently running on the CPU. 0/6 correctness ops,
  benchmark and profile unavailable, all recorded with the refusal text.
- **The collector had one bug, now fixed:** `mlx.__file__` is `None`
  (namespace package), so the probe raised `TypeError` and never found the
  shipped `mlx-omarchy-info`, which is why `capabilities` is null and the
  deep profile section reports the tool as `not-found`.
- **The run ended without saying whether anything was uploaded**, which is
  why the submitter was unsure it worked. The collector now prints an explicit
  closing line: done, nothing uploaded, how to submit.

## What it changes

- M3 is recorded as: Vulkan device absent on Omarchy Mac's current kernel;
  mlx-omarchy cannot run there until the Asahi GPU driver is enabled for
  `t8122`. README "Hardware" already limits support to the M1.
- `scripts/collect_quick.py`: locate the info tool from `mlx.core.__file__`.
- `scripts/collect_deep.py`: no contradictory "nothing written" line when
  `--out` was given; explicit closing status; default endpoint named.
