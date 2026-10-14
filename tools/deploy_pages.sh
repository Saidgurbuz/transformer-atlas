#!/usr/bin/env bash
# Redeploy the published main branch through GitHub Actions.
# Normal pushes to main already deploy automatically.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -ne 0 ]]; then
  echo 'Usage: tools/deploy_pages.sh (push changes to main first)' >&2
  exit 2
fi

gh workflow run pages.yml --ref main
echo 'Requested deployment of main. Follow it with: gh run list --workflow pages.yml'
