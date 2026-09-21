#!/bin/sh
# privacy-check.sh v2 — fail-closed per-commit private-content guard.
# Contains ONLY generic patterns. Site inventory lives in an OPTIONAL
# untracked file (.private-patterns, gitignored) and is never published.
# Scans EACH introduced commit's added lines, new filenames, and commit
# message (catches add-then-remove across a range). Any git failure is
# fatal (fail-closed).
# Usage: privacy-check.sh <base>..<head>     exit 0 clean / 1 blocked
set -u
fail(){ echo "privacy-check: $*" >&2; exit 1; }
REPO="${1:-}"
[ -d "$REPO" ] && cd "$REPO" || { [ $# -ge 2 ] || fail "usage: privacy-check.sh [repo] <base>..<head>"; }
RANGE="${2:-${1:-}}"
[ -n "$RANGE" ] || fail "usage: privacy-check.sh <base>..<head>"
case "$RANGE" in *..*) ;; *) fail "range must contain .. (got: $RANGE)";; esac
base_part="${RANGE%%..*}"; head_part="${RANGE#*..}"
[ -n "$base_part" ] && [ -n "$head_part" ] || fail "bad range: $RANGE"
git rev-parse -q --verify "$base_part" >/dev/null 2>&1 || fail "bad range base: $base_part"
git rev-parse -q --verify "$head_part" >/dev/null 2>&1 || fail "bad range head: $head_part"
[ "$base_part" != "$head_part" ] || : 

PATTERNS=$(mktemp) || fail "mktemp failed"
ALLOW=$(mktemp) || fail "mktemp failed"
trap 'rm -f "$PATTERNS" "$ALLOW"' EXIT
cat >>"$PATTERNS" <<'PAT'
/home/[a-z0-9_.-]+/
/Users/[A-Za-z0-9_.-]+/
100\.(6[4-9]|[7-9][0-9]|1[0-1][0-9]|12[0-7])\.[0-9]{1,3}\.[0-9]{1,3}
10\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}
172\.(1[6-9]|2[0-9]|3[01])\.[0-9]{1,3}\.[0-9]{1,3}
192\.168\.[0-9]{1,3}\.[0-9]{1,3}
PAT
for f in "${PRIVACY_PATTERNS:-.private-patterns}"; do
  [ -f "$f" ] && cat "$f" >>"$PATTERNS"
done
for f in "${PRIVACY_ALLOW:-.privacy-allow}"; do
  [ -f "$f" ] && cat "$f" >"$ALLOW"
done

scan(){ # scan <content> <label>; sets SCAN_HIT=1 on block
  _content="$1"; _label="$2"; SCAN_HIT=0
  while IFS= read -r pat; do
    [ -n "$pat" ] || continue
    n=$(printf '%s\n' "$_content" | grep -icE "$pat") || n=0
    [ "$n" -gt 0 ] || continue
    blocked=1
    if [ -s "$ALLOW" ]; then
      while IFS= read -r apat; do
        [ -n "$apat" ] || continue
        an=$(printf '%s\n' "$_content" | grep -icE "$apat") || an=0
        if [ "$an" -ge "$n" ]; then blocked=0; break; fi
      done < "$ALLOW"
    fi
    if [ "$blocked" -eq 1 ]; then
      echo "privacy-check: BLOCKED $_label pattern '$pat' ($n line(s))" >&2
      SCAN_HIT=1
    fi
  done < "$PATTERNS"
}

rc=0
for c in $(git rev-list --first-parent "$RANGE"); do
  msg=$(git log -1 --format=%B "$c") || fail "cannot read commit $c"
  scan "$msg" "message $c" && rc=1
  names=$(git diff-tree --no-commit-id --name-status -r "$c") || fail "diff-tree failed on $c"
  addednames=$(printf '%s\n' "$names" | awk '$1=="A"{print $2}')
  scan "$addednames" "new filenames $c" && rc=1
  added=$(git diff-tree -p --no-commit-id "$c" | grep -E '^\+[^+]') || true
  scan "$added" "added lines $c" && rc=1
done
if [ $rc -eq 0 ]; then echo "privacy-check: clean ($RANGE)"; fi
exit $rc
