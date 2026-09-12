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
push permission. Local source and smoke results do not constitute upstream
adoption.

Submitted as [upstream PR 425](https://github.com/omacom/omarchy-mac/pull/425),
head `729079d4da6b37b402542b177f5fa99ea3e3361a`, targeting `quattro`. Both
platform commands pin published installer `0545dc7f8678c9d25800d37f96ab7262cb07fc2b`
and SHA-256 `993bf0222c998367c7a82732f03477723aa7d5b1a44e89edb56ee1fa977444e8`.
The full platform install-guard script passed without skipped fixture cases.
The actual migration controller writes its completion marker after success
but not after a missing-package failure; the real menu predicate changes from
absent to present. Menu bytes differ only in the two installed-state predicates.
The graphical menu and a complete supported M1 installation remain untested.

## Review and CI outcome

Independent review found no blocking bugs. Independent QA passed all 10 MLX
guard checks without skips. Its broader local suite failed on this host, but
the sorted failure list was byte-identical on the base branch. The integrated
MLX host command `python3 -m pytest tests/coreml/ overlay/tests/omarchy/coreml/
tests/test_install_sh_contract.py -q` passed 73 tests and 3 subtests in 9.55 s.
No graphical acceptance run was possible: this host lacks a desktop session,
the required Hyprland/Quickshell tools, QEMU, and `/dev/kvm`.

At PR head `729079d`, upstream syntax/shellcheck, `tests/all`, and `test/all`
passed. The [ARM install job](https://github.com/omacom/omarchy-mac/actions/runs/34712326831/job/103603204099)
failed at its final metadata check for `omarchy-nvim-refresh` and
`omarchy-nvim-setup`. CI installed `omarchy-nvim-2026.8.13-1`. The workflow
pins `omacom/omarchy-pkgs` at `6d27290193109c07b0134360d382e64790ae5dda`;
its PKGBUILD installs the setup script and a refresh symlink, without the
required summary metadata. In a temporary directory, both the base and PR
versions of `bin/omarchy commands --check` emitted the same two failures
against that unchanged package source. Neither Neovim command was executed.
This is an upstream package dependency, not a green CI result. The diagnosis
and review boundaries are [posted on PR 425](https://github.com/omacom/omarchy-mac/pull/425#issuecomment-5648115917).

Review also identified a pre-broken-install edge: resolving the missing
wheel fails the migration, stopping the update without a completion marker.
The new menu key can hide removal until migration succeeds; direct
`omarchy-remove-ai-mlx` remains available. Errors are not swallowed and no
optional-demo compatibility key was added.

Upstream adoption, the package CI fix, and supported-device graphical
acceptance remain open. Host launcher checks do not close those gates.

The earlier patch export and source-text tests were removed. The original
patch omitted existing-install migration and pinned an unpublished installer;
that is not a valid installed-state cutover.
