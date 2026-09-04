# Forks and the backport flow

mlx-omarchy patches three upstream lineages. Each has a fork under
`joshuaswarren` carrying our work on branches. Work lands in the fork first.
Upstreaming a change is a separate, later decision.

## Upstream to fork map

| Upstream | Fork | Branch | Base and carried work |
| --- | --- | --- | --- |
| `ml-explore/mlx` | `joshuaswarren/omarchy-mlx` | `omarchy` | MLX 0.32.2 commit `1f8e74e3f12f31365464a6867c6579f0e9b29d85`, the `mlx.lock` pin, plus the six patches from `patches/`, one commit per patch |
| `eiln/ane` and `allbilly/libane` | `joshuaswarren/omarchy-ane` | `omarchy`, `omarchy-kmd` | `omarchy`: eiln/ane tip `0dcea99` plus the six-commit libane series from `~/keep/eiln-ane-series/`, applied with `git am`. `omarchy-kmd`: allbilly/libane head `1e0afd8` plus the jwm1 debug instrumentation from `~/keep/ane-kmd-local/local-kmd-changes.patch` |
| `AsahiLinux/linux` | `joshuaswarren/linux` | `ane-dt-t8103` | Asahi tag `asahi-7.1.6-1`, the exact source of the Arch kernel `7.1.6-1-1-ARCH` on the M1 test machine, plus two commits `326d6033059d18a1f47833ba7ad3a3ee2c4eb443` and `b1cb024a1`: the ANE device-tree node, `status = "disabled"` in `t8103.dtsi` and enabled in `t8103-j293.dts`. After hostile review found that an enabled ANE DART node would let the in-tree `apple-dart` driver and the out-of-tree driver fight over the same MMIO window and IRQ 417, the second commit keeps the ANE DART node disabled everywhere and drops the `iommus` phandle. |

`eiln/ane` and `allbilly/libane` are one lineage. `allbilly/libane` is a fork
of `eiln/ane` that stays ahead of it, and it carries both the kernel module in
`ane/` and the userspace library in `libane/`. GitHub allows one repository
per account per fork network, so one fork serves both. The clone carries two
upstream remotes: `upstream` for `eiln/ane` and `allbilly` for
`allbilly/libane`. Do not try to fork `allbilly/libane` separately; GitHub
refuses a second repository in the same network.

The instrumentation on `omarchy-kmd` is captured from an uncommitted working
tree on jwm1. It is not a finished fix, and the module currently resets the
machine when loaded.

`allbilly/ane` is documentation and register research with examples. We depend
on it as an upstream reference. We do not patch it, so it has no fork.

The former `joshuaswarren/omarchy-libane` and
`DeckardAndFriends/omarchy-libane` repositories are archived. Each README
points at `joshuaswarren/omarchy-ane` as the live repository for this lineage.

## Why we fork

Owner decision, 2026-09-03: maintain forks, backport from upstream into them,
and add our fixes there. We do not wait on upstream review to land work.
Whether to upstream a change stays open per change. Two standing rules:
never push to an upstream remote, and never force-push.

## Backport flow

Example for `omarchy-ane`. The same steps work in every fork clone:

1. `git fetch upstream` for `eiln/ane`, or `git fetch allbilly` for `allbilly/libane`
2. `git checkout <branch>`
3. `git merge <remote>/main` for a full sync, or `git cherry-pick <commit>` for a targeted backport
4. `git push origin <branch>`

Rules:

- Upstream history is never rewritten. Merge or cherry-pick. Never rebase a shared branch onto upstream, because that would demand a force-push.
- Never force-push.
- All pushes go to `origin`. The upstream remotes are read-only in practice.

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

## Kernel device tree fork

`joshuaswarren/linux` exists so the ANE device-tree work has a reviewable
home. Mesa will not take user-space driver support until a kernel driver is
upstream, so kernel-side work is prepared here first. The branch
`ane-dt-t8103` carries the device-tree patch that lets the ANE node reach
the kernel through the normal m1n1 payload path instead of a GRUB
`devicetree` override, which freezes m1n1's per-boot patching and cost the
test machine seven of its eight CPU cores. See `docs/boot-and-kernel.md`
for the failure and the verification checklist.

Status of that branch: it has never been booted. Before anyone installs
it, the checklist in the commit message must pass on a disposable machine.
The branch ships no ANE driver. The out-of-tree driver in
`joshuaswarren/omarchy-ane` must stay blacklisted; loading it hard-reset
the M1 on 2026-09-03.
