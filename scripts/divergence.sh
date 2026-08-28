#!/usr/bin/env bash
# Report how far this fork has drifted from Cloudflare's pingora.
#
# Run it before a rebase, after one, and any time the sbproxy lockfile moves
# to a new fork rev. The numbers it prints belong in sbproxy's Cargo.toml,
# in the comment above `[patch.crates-io]`.
#
# Why this exists: for a long time the fork's fetch refspec was
# `+refs/tags/0.8.0:refs/tags/0.8.0`, one tag and nothing else, so
# `origin/main` never resolved and nobody could answer "how far behind are
# we" without noticing the refspec first. This script fixes the refspec if
# it finds it narrowed, so the answer stays available.
#
# Note on tags: `0.8.1` is NOT an ancestor of `main`. Cloudflare cuts
# releases on a release branch, so a tag carries commits main does not have
# and vice versa. This fork tracks `main`. Rebasing it onto a release tag
# would move it to a different line, not an older point on the same one.

set -euo pipefail

UPSTREAM_REMOTE="${UPSTREAM_REMOTE:-origin}"
UPSTREAM_BRANCH="${UPSTREAM_BRANCH:-main}"
FORK_BRANCH="${FORK_BRANCH:-sbproxy-0.8.0}"

cd "$(dirname "$0")/.."

# A refspec that fetches only a tag leaves `origin/main` unresolvable, which
# is the state this script exists to stop recurring.
if ! git config --get-all "remote.${UPSTREAM_REMOTE}.fetch" | grep -q 'refs/heads/\*'; then
    echo "note: ${UPSTREAM_REMOTE} was not fetching branches; widening its refspec" >&2
    git config --unset-all "remote.${UPSTREAM_REMOTE}.fetch" || true
    git config --add "remote.${UPSTREAM_REMOTE}.fetch" \
        "+refs/heads/*:refs/remotes/${UPSTREAM_REMOTE}/*"
    git config --add "remote.${UPSTREAM_REMOTE}.fetch" '+refs/tags/*:refs/tags/*'
fi

git fetch --quiet "$UPSTREAM_REMOTE"

upstream="${UPSTREAM_REMOTE}/${UPSTREAM_BRANCH}"

# Prefer the pushed branch over the local one. A local checkout left behind
# a push reports a smaller divergence than is real, which is the direction
# that lets a drift go unnoticed.
FORK_REMOTE="${FORK_REMOTE:-sb}"
fork="${FORK_REMOTE}/${FORK_BRANCH}"
if git rev-parse --verify --quiet "$fork" >/dev/null; then
    git fetch --quiet "$FORK_REMOTE"
    if git rev-parse --verify --quiet "$FORK_BRANCH" >/dev/null &&
        [ "$(git rev-parse "$FORK_BRANCH")" != "$(git rev-parse "$fork")" ]; then
        echo "note: local ${FORK_BRANCH} differs from ${fork}; reporting ${fork}" >&2
    fi
else
    fork="$FORK_BRANCH"
fi

base="$(git merge-base "$upstream" "$fork")"
read -r behind ahead <<<"$(git rev-list --left-right --count "${upstream}...${fork}")"
files="$(git diff --name-only "${upstream}...${fork}" | wc -l | tr -d ' ')"
src="$(git diff --name-only "${upstream}...${fork}" | grep -cv '^\.github/' || true)"

cat <<EOF
fork:        ${fork}
upstream:    ${upstream} at $(git rev-parse --short "$upstream")
merge base:  $(git rev-parse --short "$base")  ($(git log -1 --format=%ad --date=short "$base"))

behind:      ${behind} upstream commits not in the fork
ahead:       ${ahead} commits of ours not upstream
files:       ${files} differ (${src} outside .github/)

ours:
EOF
git log --format='  %h  %s' "${upstream}..${fork}"

echo
echo "files we carry, outside CI config:"
git diff --name-only "${upstream}...${fork}" | grep -v '^\.github/' | sed 's/^/  /'
