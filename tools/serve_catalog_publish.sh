#!/bin/sh
# Publish a refreshed serve catalog as a metadata-only PR.
#
# Invariants (Main review 2026-09-20):
# - The publish branch always starts from the CURRENT HEAD where the
#   refresh ran, never from a stale remote tip, so the generated diff is
#   preserved and every PR regenerates fully from current main.
# - The push is --force-with-lease against the fetched remote tip: a
#   concurrent foreign update to the branch fails the push instead of
#   being silently overwritten.
# - PR creation queries the existing PR explicitly and either creates one
#   or reports the existing one; gh failures are LOUD (auth/network
#   problems must fail the job, never masquerade as "already open").
# - Retry-safe: a previous run whose PR query failed leaves the diff on
#   the publish branch; this run re-detects branch-vs-base divergence and
#   still ensures the PR exists.
#
# Usage: serve_catalog_publish.sh <catalog-path> <branch> [base]
# Requires: git, gh (GH_TOKEN set). Exit 0 = nothing to publish or PR
# ensured; non-zero = real failure.

set -eu
# POSIX sh: no pipefail. No pipeline in this script needs it; every gh
# call is a single command whose failure set -e propagates.

catalog="$1"
branch="$2"
base="${3:-main}"

git diff --quiet -- "$catalog" || local_diff=1
local_diff="${local_diff:-0}"

if git ls-remote --exit-code --heads origin "$branch" >/dev/null 2>&1; then
    git fetch origin "$branch"
    has_remote=1
else
    has_remote=0
fi

if [ "$local_diff" = 0 ] && [ "$has_remote" = 0 ]; then
    echo "No catalog diff and no publish branch; nothing to publish."
    exit 0
fi

# Branch from the CURRENT HEAD: the generated diff survives the checkout
# (same-commit checkout keeps the worktree), and the branch is never
# seeded from a stale remote tip.
git checkout -B "$branch"

if [ "$local_diff" = 1 ]; then
    git add -- "$catalog"
    git commit -m "catalog: availability refresh (metadata-only, automated)"
fi

# A publish is needed iff the branch's catalog differs from base's.
if git diff --quiet "$base" "$branch" -- "$catalog"; then
    echo "Publish branch matches $base; nothing to publish."
    exit 0
fi

git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
git push --force-with-lease origin "$branch"

pr_body='Automated availability refresh: size_bytes / refreshed_at /
generated_at only. The refresher cannot write qualification,
recommended, revision, or priority; the structural pre-write guard and
tests/test_serve_catalog_refresh.py verify it. Any upstream-drift
warnings are in the job log. Human merge required; nothing auto-merges.'

existing="$(gh pr list --head "$branch" --base "$base" --state open \
    --json number --jq 'length')"
case "$existing" in
    0)
        printf '%s\n' "$pr_body" > pr-body.md
        gh pr create --head "$branch" --base "$base" \
            --title "serve catalog: availability refresh (metadata-only)" \
            --body-file pr-body.md
        ;;
    1)
        echo "PR already open for $branch; pushed update to it."
        ;;
    *)
        echo "ERROR: $existing open PRs for $branch; refusing to guess." >&2
        exit 1
        ;;
esac
