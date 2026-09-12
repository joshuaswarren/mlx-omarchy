# h13-conv-relu-boundary fixtures (real, byte-exact)

Payloads for the plan §46 known-H13 graph and the §62 bundle-contract
boundary demonstration. Copied verbatim (read-only) from the root
qualification run of `scripts/verify-ane-compiler.sh all`
(mlx-omarchy main `a12eafd994af881dbc2b39c0b9807523f27b171a`):

- package: `/dev/shm/mlx-compiler-review-20260912/ane-compiler/h13-qualification/conv_relu`
  (4097 programs + `manifest.json`, schema `mil-hwxc.h13-anec-package.v1`,
  target H13, artifactFormat anec)
- compiler: `mil-hwx-compiler` `a0ce354cf800011a84420da4e12013eb8140b2a5`,
  extracted from lock-verified archive
  `12bfbcd0049849dba18ee5fd3556762bf3313459c8ca5a65ae962f507a5b6d64`

Provenance chain: `ane-compiler.lock` pin -> archive SHA-256 check ->
pinned source tree -> `tests/fixtures/conv_relu.mil` -> emitted package.

| file | sha256 | bytes |
|---|---|---|
| `conv_relu.mil` | `cc204fd0252b16fe514970fcd82abde15600303f8355718984ae8ba4ddab7cf8` | 947 |
| `program-0.anec` | `63462dece46ebcd100b3785be09979786c969d7add75b212902344418fcb7123` | 12928 |
| `program-1.anec` | `3d73787c1fc04842da72bb6e1a1930ffa1d3c7b7280ab7d838add2da2edfba22` | 4736 |

`program-0.anec` is the `conv` program (encoder `apple-parity-conv`, W baked
into the kernel section at `constantOffset` 640, `constantBytes` 8192).
`program-1.anec` is the first of the 4096 `maximum` slice programs (encoder
`h13-source-qualified`, 2 sources, kernel_size 0). Together they cover both
program shapes in the package, which the all-program sweep in
`receipts/2026-09-12-h13-bundle-adapter-blocker/demonstrate_blocker.py`
verifies mechanically.

No device was accessed to produce these files, and no ANE execution is
claimed anywhere in this fixture set.
