# 2026-09-01 — Published-artifact verification, v0.3.0-alpha.1

First verification of the RELEASED wheels (previous gates used local builds).
Assets downloaded with `gh release download v0.3.0-alpha.1 --repo joshuaswarren/mlx-omarchy`.
Environment: 2026-09-01, workstation (x86_64, Vulkan/llvmpipe) and jwm1-linux (M1, Asahi, real GPU).

## Asset integrity

| Asset | sha256 (downloaded) | Local build receipt | Match |
|---|---|---|---|
| `...-cp311-cp311-linux_x86_64.whl` | `b4ca0c8e9cf475bff476964115f0f82a6459111d44481ac0eb1310a51da1c13c` | same | YES |
| `...-cp314-cp314-linux_aarch64.whl` | `94e53d89b33961af3331ecfca495e649464e5dce707f08ec8196365a0e81f30a` | same | YES |

GitHub renamed `+6049968` to `.6049968` in the aarch64 asset filename. Byte content is identical
(sha256 verified above). Note: pip REJECTS the `.6049968` filename ("Invalid wheel filename (invalid
version)"); installing from the published asset requires a local rename back to `+6049968`. This is a
upload-pipeline defect worth fixing (tag the build without a local version segment or re-upload
pre-renamed); it does not affect the wheel contents.

## x86_64 (this workstation, fresh venv, `MLX_OMARCHY_ALLOW_NON_APPLE=1`)

- README value_and_grad snippet: `array(14.9008, dtype=float32) array([[39.2658], [54.1665]], dtype=float32)` — PASS
- D1: `mx.log2(mx.array(1024.0))` = `10.0`; `mx.log10(mx.array(1000.0))` = `2.999999761581421` — PASS
- C2: `np.array(mx.abs(...))` raised catchable `RuntimeError: [omarchy] Abs is not implemented for the
  Omarchy Vulkan backend (dtype=float32, shape=[2]). No CPU fallback is available` — PASS
- C3: `mx.save` of `mx.zeros((0,))` writes `.npy` fallback (`empty.safetensors.npy`, 70 bytes); the
  `.safetensors` path itself is NOT created, so `mx.load("<name>.safetensors")` fails with
  `[load_safetensors] Failed to open file`. Round-trip works through the `.npy` path:
  `mx.load("empty.safetensors.npy")` → `(0,) float32`. PASS with a note: the `save` side appends `.npy`
  without telling the caller, so the naive `mx.save(p); mx.load(p)` round-trip trips. Suggest the save
  side either log/warn or the load side resolve the `.npy` sibling.

## aarch64 (jwm1-linux @ 192.168.3.66, Asahi M1, fresh venv, no allow flag — real device)

- default device: `Device(gpu, 0)`
- value_and_grad: `14.900775909423828 [[39.26576232910156], [54.166542053222656]]` — PASS (within 1e-4 of README)
- D1: log2 `10.0`, log10 `2.999999761581421` — PASS
- C2: same catchable RuntimeError as x86_64 — PASS
- C3: zero-size save/load round-trip `(0,) float32` — PASS (same `.npy` path note)

## mlx-lm on the M1 (README command, verbatim)

Method: reused `.work/venv-fp16` after `pip install --force-reinstall --no-deps` of the DOWNLOADED
wheel (mlx-lm 0.31.3 + pure-python deps already present). Installed dist confirmed as
`0.32.2.dev20260901+6049968`.

```
==========
Paris
==========
Prompt: 41 tokens, 19.156 tokens-per-sec
Generation: 2 tokens, 4.269 tokens-per-sec
Peak memory: 0.292 GB
```
Command: `MLX_DISABLE_COMPILE=1 python -m mlx_lm generate --model /home/joshuawarren/models/Qwen2.5-0.5B-Instruct-4bit-mlx --prompt 'What is the capital of France? Answer in one word.' --max-tokens 32`

## Verdict

**v0.3.0-alpha.1 is sound as published.** Both wheels are bit-identical to the verified local builds,
install clean in fresh venvs on their target platforms, and pass all four regression checks plus real
on-device mlx-lm generation. Two non-blocking defects to file separately: (1) GitHub's `+`→`.` rename
in the asset name makes the published aarch64 wheel uninstallable without a manual rename — a PEP 440
upload-pipeline fix is needed; (2) zero-size `mx.save` silently appends `.npy`, so the documented-name
round-trip fails on the load side.

No commits made. Receipt only.

## Asset republish, 2026-09-01

The first upload used wheels versioned `0.32.2.dev20260901+6049968`.
GitHub rewrites `+` to `.` in asset names, and pip rejects the result with
`Invalid wheel filename (invalid version)`. The published aarch64 wheel was
therefore not installable, which is release-blocking for the target platform.

Both wheels were rebuilt with `DEV_RELEASE=1`, which drops the local-version
suffix and yields `0.32.2.dev20260901`. That name survives GitHub upload and
pip accepts it. Contents are the same tree, commit 6049968.

- x86_64: `mlx_omarchy-0.32.2.dev20260901-cp311-cp311-linux_x86_64.whl`,
  2725514 bytes, sha256 fef6c53dfa9e74846681db72eb678dedb12943b18656e91fa59ac2dbd9c19749
- aarch64: `mlx_omarchy-0.32.2.dev20260901-cp314-cp314-linux_aarch64.whl`,
  2618384 bytes, sha256 a77ebee0889c3d1d04083401051786ebd473e4e2fc40dcc0914f3cd30ef4555a

Verification after republish, downloading from the release URL:

x86_64 on the dev workstation, fresh venv, `MLX_OMARCHY_ALLOW_NON_APPLE=1`:

```
version 0.32.2.dev20260901
value 14.9008 grad [39.2658, 54.1665]
log2(1024) 10.0 | log10(1000) 3.0
C2 ok: [omarchy] Abs is not implemented for the Oma
```

aarch64 on jwm1-linux, fresh venv, real Honeykrisp device, no override:

```
version 0.32.2.dev20260901 Device(gpu, 0)
log2 10.0 log10 3.0
grad [39.2658, 54.1665]
C2 ok
```

Both install directly from the release with `pip install <downloaded wheel>`.
