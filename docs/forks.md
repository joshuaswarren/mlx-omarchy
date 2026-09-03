# Forks and the backport flow

mlx-omarchy patches three upstream repositories. Each upstream has a fork under
`joshuaswarren` with an `omarchy` branch. Work lands in the fork first.
Upstreaming a change is a separate, later decision.

## Upstream to fork map

| Upstream | Fork | Branch | Base and carried work |
| --- | --- | --- | --- |
| `ml-explore/mlx` | `joshuaswarren/omarchy-mlx` | `omarchy` | MLX 0.32.2 commit `1f8e74e3f12f31365464a6867c6579f0e9b29d85`, the `mlx.lock` pin, plus the six patches from `patches/`, one commit per patch |
| `eiln/ane` | `joshuaswarren/omarchy-ane` | `omarchy` | upstream `main` tip at fork time, `0dcea99`, plus the six-commit libane series applied with `git am` |
| `allbilly/libane` | `joshuaswarren/omarchy-libane` | `omarchy` | upstream head `1e0afd8` plus `local-kmd-changes.patch` as one commit |

`allbilly/ane` is documentation and register research with examples. We depend
on it as an upstream reference. We do not patch it, so it has no fork.

## omarchy-libane is not a GitHub fork object

GitHub allows one repository per account per fork network. `eiln/ane` and
`allbilly/libane` share one network, and `joshuaswarren/omarchy-ane` occupies
the slot. GitHub refused a second fork and a transfer with:
`joshuaswarren already has a repository in the eiln/ane network`.

`joshuaswarren/omarchy-libane` is therefore a plain repository with the same
history and remotes as a fork. `DeckardAndFriends/omarchy-libane` is a real
fork whose recorded parent is `allbilly/libane`. If GitHub lifts the limit, or
one of the two network repositories is later deleted or detached, transfer that
fork to `joshuaswarren` and it becomes the canonical one.

## Why we fork

Owner decision, 2026-09-03: maintain forks, backport from upstream into them,
and add our fixes there. We do not wait on upstream review to land work.
Whether to upstream a change stays open per change. Two standing rules:
never push to an upstream remote, and never force-push.

## Backport flow

Example for `omarchy-mlx`. The same steps work in every fork clone:

1. `git fetch upstream`
2. `git checkout omarchy`
3. `git merge upstream/main` for a full sync, or `git cherry-pick <commit>` for a targeted backport
4. `git push origin omarchy`

Rules:

- Upstream history is never rewritten. Merge or cherry-pick. Never rebase the shared `omarchy` branch onto upstream, because that would demand a force-push.
- Never force-push.
- All pushes go to `origin`. `upstream` is read-only in practice.

`omarchy-mlx` starts at the `mlx.lock` pin, not at the upstream tip. A merge of
`upstream/main` also brings everything after the pin. To ride one upstream fix
on the pinned base, cherry-pick that commit instead.

## Deliberately not wired up

The build still consumes the pinned upstream archive:

- `mlx.lock` pins MLX 0.32.2 at commit `1f8e74e3f12f31365464a6867c6579f0e9b29d85`, with the archive URL and SHA-256.
- `scripts/prepare-mlx.sh` downloads the archive and applies the six patches from `patches/`.
- Nothing in the build reads `joshuaswarren/omarchy-mlx`.

Switching the build to the fork is an owner decision that has not been made.
It would change the release story: a release would name a fork commit instead
of an upstream archive plus patch files, the patch set would live as commits
instead of `patches/*.patch`, and provenance would point at `joshuaswarren`
instead of `ml-explore`.

## ane.dtbo

`allbilly/libane` ships `ane.dtbo`. A full static device tree that carries an
ANE node cost this project seven of eight CPU cores. An overlay applied to the
live tree is the correct mechanism. Read `docs/boot-and-kernel.md` before
kernel work that touches the device tree.
