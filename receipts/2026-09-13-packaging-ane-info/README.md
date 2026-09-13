# Receipt: Phase 10 packaging groundwork (§73, §81-84)

Host-only. Not a hardware pass. No GPU/ANE execution, no module load, no
fleet access.

## Source

- Repository: `joshuaswarren/mlx-omarchy`
- Branch: `feature/packaging-ane-info`
- Parent: `0d4eec2a224b5f4bf592b072a8b451799fd14b88` (`origin/main`)

## What landed

- `scripts/installed_state.py` — FDT `apple,*-ane` probe (reuses
  `collect_quick._ane_devicetree`), `/dev/accel/accel0` character-device
  check, `/sys/module/ane` presence, Core ML frontend/cache presence.
  `MLX_OMARCHY_SYSROOT` / `--sysroot` injects a fake tree.
- `overlay/tools/mlx-omarchy-info/main.cpp` — same ANE and Core ML fields
  on the installed `mlx-omarchy-info` reporter (`--json` and text).
- `install.sh` — GPU smoke still always runs and refuses on failure. ANE
  smoke runs only when `${MLX_OMARCHY_ACCEL_DEV:-/dev/accel/accel0}` is a
  character device, and refuses if that node exists without an FDT ANE
  node and loaded `ane` module. No `kmod-ane` package install. Uninstall
  still removes `mlx-omarchy`, `mlx-omarchy-demo`, `mlx-omarchy-info`,
  the desktop entry, and `$PREFIX`.
- Omarchy-mac follow-up (separate repo/branch, does not touch PR 425):
  `docs/ane-kmod-gate.md` on `docs/ane-kmod-gate` from `b18809aa`.

## Host checks

Command:

```sh
python3 -m unittest tests.test_installed_state tests.test_install_sh_contract tests.test_release_assets
```

Output:

```text
..................
----------------------------------------------------------------------
Ran 18 tests in 0.347s

OK
```

Working directory:
`~/.config/superpowers/worktrees/mlx-omarchy/PackagingPrep`

`/dev/null` is used only as a `S_ISCHR` positive control. That is not an
ANE device and is not a hardware result.

## Not claimed

- C++ `mlx-omarchy-info` was not compiled (needs the MLX overlay build).
- No `/dev/accel/accel0` on this host; ANE smoke did not run against a
  real accelerator.
- Menu label remains `MLX (Apple GPU)`.
- `kmod-ane` is still omarchy-pkgs / Omarchy M platform work.
