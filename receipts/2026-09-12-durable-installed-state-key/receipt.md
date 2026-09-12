# Durable MLX installed-state key

The installer creates `~/.local/bin/mlx-omarchy-info` and removes it on
uninstall. It resolves the wheel's executable through the installed `mlx`
namespace once, using isolated Python, then writes a quoted native launcher.
The existing interpreter, optional demo, GPU label, and installation prefix
are unchanged. No ANE capability is added.

## Verification

- `python3 -m unittest tests.test_install_sh_contract -v`: 2 tests passed.
  Real installer commands refuse off-target installation before writes and
  remove owned artifacts without removing an unrelated file.
- `ruff check tests/test_install_sh_contract.py`, `bash -n install.sh`, and
  `shellcheck install.sh` passed.
- [Real-wheel smoke](smoke.json): the generated launcher executes the same
  native reporter as the installed x86 wheel and returns its same off-target
  `backend unavailable` result. This is launcher verification, not Apple GPU
  qualification or a complete installation test.
- The proposed upstream migration adds the reporter to an existing install
  without requiring the optional demo, preserves existing interpreter bytes,
  does not rewrite an existing reporter, and leaves no reporter when the MLX
  package is missing. It also succeeds without writes when no install exists.
  All actions used a disposable HOME; no live installation was changed.

## Upstream dependency

The menu and platform installation commands belong to `omacom/omarchy-mac`,
whose default branch is `quattro`. This account has pull permission but no
push permission. The platform change therefore requires a fork pull request;
local source and smoke results do not constitute upstream adoption.

The earlier patch export and source-text tests were removed. The original
patch omitted existing-install migration and pinned an unpublished installer;
that is not a valid installed-state cutover.
