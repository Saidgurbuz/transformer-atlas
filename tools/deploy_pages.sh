#!/usr/bin/env bash
# Build the static site and publish it to the `site` branch of this repo.
# (Not `gh-pages`: GitHub auto-publishes a branch with that name, even for private repos.)
#   tools/deploy_pages.sh            update the site branch (nothing becomes public by itself)
#   tools/deploy_pages.sh --enable   also turn on GitHub Pages for the branch: the site goes
#                                    public at https://<user>.github.io/transformer-atlas/
#                                    (works for this private repo on the owner's plan)
# To unpublish: delete the Pages site in the repo settings (or gh api -X DELETE repos/<repo>/pages).
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
BRANCH=site
SHA=$(git rev-parse --short HEAD)

tools/build_pages.sh

SITE="$ROOT/.pages"
if [ ! -d "$SITE/.git" ] && [ ! -f "$SITE/.git" ]; then
  rm -rf "$SITE"
  if git ls-remote --exit-code --heads origin "$BRANCH" >/dev/null 2>&1; then
    git fetch -q origin "$BRANCH"
    git worktree add -q -B "$BRANCH" "$SITE" "origin/$BRANCH"
  else
    git worktree add -q --detach "$SITE"
    (cd "$SITE" && git checkout -q --orphan "$BRANCH" && git rm -rq --cached . && git clean -fdxq)
  fi
fi

# fresh copy of the app (no tests, tools, reference code or docs)
find "$SITE" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
cp -r dist/pages/. "$SITE/"

cd "$SITE"
git add -A
if git diff --cached --quiet; then
  echo "$BRANCH already up to date with $SHA"
else
  git commit -q -m "deploy: site from main@$SHA"
  git push -q -u origin "$BRANCH"
  echo "$BRANCH updated from main@$SHA"
fi

if [ "${1:-}" = "--enable" ]; then
  if gh api "repos/$REPO/pages" >/dev/null 2>&1; then
    echo "Pages already enabled"
  else
    gh api -X POST "repos/$REPO/pages" -f "source[branch]=$BRANCH" -f 'source[path]=/' >/dev/null
    echo "Pages enabled"
  fi
  gh api "repos/$REPO/pages" --jq '.html_url'
fi
