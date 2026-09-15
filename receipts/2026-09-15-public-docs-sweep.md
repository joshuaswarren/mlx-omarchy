# 2026-09-15 — Public docs sweep (hostname / homelab scrub)

Scope: public READMEs and README-linked docs across the ANE-related repos.
Evidence files are untouched; hostname-bearing receipt filenames in README text
were replaced with generic `receipts/` pointers instead of renaming evidence.
AGENTS.md files left alone per assignment. omarchy-linux README is the stock
kernel README (clean, untouched).

Verification: `grep -rE 'jwm1|jw16|jw16mbp1|macstudio|omp-studio-local|/var/tmp|tailscale|192\.168\.|thewarrens'` over every edited file → 0 matches.

## mlx-omarchy — branch `docs/public-readme-sweep`, commit `448370b6569c1cc4f8ba61d421ab4efa3cc33cc4`
- README.md: "M1 (`jwm1`)" / "M1 Max (`jw16`)" perf labels → hardware names;
  hostname-bearing receipt links → generic `receipts/` links (Performance and
  Neural Engine sections).
- docs/2026-09-13-encoder-parity-harness.md: "proven on jwm1" → "proven on
  hardware".
- docs/known-defects.md: two "on jwm1"/"jwm1-linux" mentions → hardware names.
- docs/CONTRIBUTOR-GUIDE.md: rejected-idea table "on jwm1" → "on M1".

## ane-linux-experiments — branch `docs/public-readme-sweep`, commit `ea08a576cb83c8d7d59a98f720c5121621396ece`
- README.md Current status: hostnames (jwm1, jw16, jw16mbp1-linux),
  `jwm1-ane.service`, and the `mlx-omarchy/receipts/2026-09-13-jw16-*` path
  replaced with hardware names (T8103/M1, T6001/M1 Max) and generic receipt
  pointers. Technical content (commit SHAs, channel layout, SET semantics)
  preserved verbatim.
- README.md Qwen reference workflow: machine-local capture command
  (`~/src/ANEForge`, `$HOME/src/llama.cpp`, `/tmp`, `~/.local/bin/uv`)
  rewritten as a portable macOS command using the in-repo
  `tools/aneforge-qwen-reference.py`.
- docs/fresh-hwx-usage.md: "proven on jwm1" → "proven on T8103 hardware".
- docs/omarchy-ane-out-of-box-plan.md: jwm1/jw16 mentions and the
  jw16-named mlx-omarchy receipt path → hardware names / generic receipt
  reference.

## mil-hwx-compiler — branch `docs/public-readme-sweep`, commit `eff0faae87f253798bf748fb097275d4ededdca2`
- README.md: two `--host macstudio` macOS re-mint examples → `--host
  <macos-host>` (the flag is an SSH target; placeholder keeps the meaning).
- docs/ane/README.md: already clean.

## omarchy-linux — branch `docs/public-readme-sweep`, commit `89cf2774434f78ea8a512bd69b1e9acbfbb46e9a`
- ANE-STAGING-NOTE.md (tracked, repo root): three "jwm1" mentions → "the M1
  host" / M1 hardware. Note is not README-linked; cleaned as root-level
  user-facing doc.

## omarchy-ane
- README.md already clean (0 matches); no change, no commit.

## Not landed
Branches are local; hand to LandHostLeaves for landing/PR.
